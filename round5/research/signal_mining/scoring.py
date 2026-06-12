from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def signal_id(prefix: str, parts: Sequence[Any]) -> str:
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"SIG_{prefix}_{digest}"


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 30 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if math.isnan(value) else value


def rank_corr(x, y) -> float:
    xs = pd.Series(x).rank(method="average").to_numpy()
    ys = pd.Series(y).rank(method="average").to_numpy()
    return safe_corr(xs, ys)


def hit_rate(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    centered = xv - np.nanmean(xv)
    nonzero = np.sign(centered) != 0
    if nonzero.sum() < 30:
        return 0.0
    return float(np.mean(np.sign(centered[nonzero]) == np.sign(yv[nonzero])))


def monotonicity_score(x, y, bins: int = 5) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 100 or data["x"].nunique() < bins:
        return 0.0
    try:
        data["bin"] = pd.qcut(data["x"], q=bins, duplicates="drop")
    except ValueError:
        return 0.0
    means = data.groupby("bin", observed=True)["y"].mean().to_numpy()
    if len(means) < 3:
        return 0.0
    diffs = np.diff(means)
    if np.all(diffs >= 0) or np.all(diffs <= 0):
        return float(abs(means[-1] - means[0]))
    return float(abs(safe_corr(np.arange(len(means)), means)) * abs(means[-1] - means[0]))


def mi_score(x, y, max_samples: int = 5000) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 100 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    if len(x) > max_samples:
        idx = np.linspace(0, len(x) - 1, max_samples).astype(int)
        x = x[idx]
        y = y[idx]
    try:
        return float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0])
    except Exception:
        return 0.0


def day_metric(x, y, include_mi: bool = False) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    xv = x[mask]
    yv = y[mask]
    ic = safe_corr(xv, yv)
    ric = rank_corr(xv, yv) if len(xv) <= 20000 else safe_corr(pd.Series(xv).rank().to_numpy(), pd.Series(yv).rank().to_numpy())
    return {
        "ic": ic,
        "rank_ic": ric,
        "hit_rate": hit_rate(xv, yv),
        "monotonicity": monotonicity_score(xv, yv),
        "mi": mi_score(xv, yv) if include_mi else 0.0,
        "sign": 1 if ic > 0 else -1 if ic < 0 else 0,
        "n": int(len(xv)),
    }


def aggregate_days(base: Dict[str, Any], day_metrics: Dict[int, Dict[str, Any]], min_abs_ic: float = 0.01) -> Dict[str, Any]:
    out = dict(base)
    for day in [2, 3, 4]:
        metric = day_metrics.get(day, {"ic": 0.0, "rank_ic": 0.0, "hit_rate": 0.0, "monotonicity": 0.0, "mi": 0.0, "sign": 0, "n": 0})
        out[f"score_day{day}"] = metric["ic"]
        out[f"rank_ic_day{day}"] = metric["rank_ic"]
        out[f"hit_day{day}"] = metric["hit_rate"]
        out[f"mono_day{day}"] = metric["monotonicity"]
        out[f"mi_day{day}"] = metric["mi"]
        out[f"sign_day{day}"] = metric["sign"]
        out[f"n_day{day}"] = metric["n"]
    signs = [out[f"sign_day{d}"] for d in [2, 3, 4]]
    oos_scores = [abs(float(out[f"score_day{d}"])) for d in [3, 4]]
    train_score = abs(float(out["score_day2"]))
    sign_stable = signs[0] != 0 and signs[0] == signs[1] == signs[2]
    min_n = min(int(out[f"n_day{d}"]) for d in [2, 3, 4])
    stability_score = (
        min(train_score, *oos_scores)
        + 0.25 * min(abs(float(out["rank_ic_day2"])), abs(float(out["rank_ic_day3"])), abs(float(out["rank_ic_day4"])))
        + 0.05 * max(0.0, min(float(out["hit_day3"]), float(out["hit_day4"])) - 0.5)
    )
    out["sign_stable"] = sign_stable
    out["min_abs_oos_ic"] = min(oos_scores)
    out["min_n"] = min_n
    out["stability_score"] = stability_score
    if sign_stable and min_n >= 100 and train_score >= min_abs_ic and min(oos_scores) >= min_abs_ic:
        out["tier"] = "confirmed"
    elif min_n >= 100 and train_score >= min_abs_ic and max(oos_scores) >= min_abs_ic:
        out["tier"] = "promising"
    elif train_score >= min_abs_ic:
        out["tier"] = "unstable"
    else:
        out["tier"] = "noisy"
    return out


def _future_return(group: pd.DataFrame, horizon: int) -> pd.Series:
    return group["mid_price"].shift(-horizon) - group["mid_price"]


def score_feature_signals(
    features: pd.DataFrame,
    feature_cols: List[str],
    family_map: Dict[str, List[str]],
    horizons: List[int],
    include_mi: bool = False,
    max_features: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    feature_cols = feature_cols[:max_features] if max_features else feature_cols
    family_by_feature = {feature: family for family, cols in family_map.items() for feature in cols}
    grouped = {key: grp for key, grp in features.groupby(["day", "product"], sort=False)}
    for product in sorted(features["product"].unique()):
        product_groups = {day: grouped.get((day, product)) for day in [2, 3, 4]}
        for feature in feature_cols:
            if feature not in features.columns:
                continue
            family = family_by_feature.get(feature, "feature")
            for horizon in horizons:
                metrics = {}
                for day, grp in product_groups.items():
                    if grp is None or len(grp) < horizon + 30:
                        continue
                    target = _future_return(grp, horizon)
                    metrics[day] = day_metric(grp[feature].to_numpy(), target.to_numpy(), include_mi=include_mi)
                base = {
                    "signal_id": signal_id("FEAT", [product, feature, horizon]),
                    "scope": "single_product",
                    "family": family,
                    "product": product,
                    "feature": feature,
                    "horizon": horizon,
                }
                rows.append(aggregate_days(base, metrics))
    return rows


PAIR_FEATURES_DEFAULT = [
    "ret_1",
    "ret_3",
    "ret_10",
    "imbalance_l1",
    "weighted_imbalance",
    "microprice_edge_rel",
    "microprice_edge_z_50",
    "spread_z_50",
    "total_depth_delta_1",
    "volatility_50",
]

REFINED_PAIR_FEATURES = [
    "imbalance_l1",
    "weighted_imbalance",
    "microprice_edge_rel",
    "microprice_edge_z_20",
    "microprice_edge_z_50",
    "microprice_edge_z_100",
    "spread_z_20",
    "spread_z_50",
    "depth_z_50",
    "total_depth_delta_1",
]


def score_pair_signals(
    features: pd.DataFrame,
    products: List[str],
    horizons: List[int],
    pair_features: Optional[List[str]] = None,
    max_products: Optional[int] = None,
) -> List[Dict[str, Any]]:
    products = products[:max_products] if max_products else products
    pair_features = [f for f in (pair_features or PAIR_FEATURES_DEFAULT) if f in features.columns]
    rows: List[Dict[str, Any]] = []
    grouped = {key: grp for key, grp in features.groupby(["day", "product"], sort=False)}
    for leader in products:
        for follower in products:
            if leader == follower:
                continue
            for feature in pair_features:
                for horizon in horizons:
                    metrics = {}
                    for day in [2, 3, 4]:
                        leader_grp = grouped.get((day, leader))
                        follower_grp = grouped.get((day, follower))
                        if leader_grp is None or follower_grp is None:
                            continue
                        merged = leader_grp[["timestamp", feature]].merge(
                            follower_grp[["timestamp", "mid_price"]], on="timestamp", how="inner"
                        )
                        if len(merged) < horizon + 30:
                            continue
                        target = merged["mid_price"].shift(-horizon) - merged["mid_price"]
                        metrics[day] = day_metric(merged[feature].to_numpy(), target.to_numpy(), include_mi=False)
                    base = {
                        "signal_id": signal_id("PAIR", [leader, follower, feature, horizon]),
                        "scope": "ordered_pair",
                        "family": "lead_lag",
                        "leader": leader,
                        "follower": follower,
                        "feature": feature,
                        "horizon": horizon,
                    }
                    rows.append(aggregate_days(base, metrics))
    return rows


def score_residual_pair_signals(
    features: pd.DataFrame,
    products: List[str],
    horizons: List[int],
    pair_features: Optional[List[str]] = None,
    max_products: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Score lead-lag after removing same-timestamp marketwide common factors.

    This is deliberately stricter than raw pair scoring. It subtracts the
    day/timestamp cross-sectional mean from both leader feature and follower
    future return, which prevents one broad market move from creating thousands
    of duplicate pair edges.
    """
    products = products[:max_products] if max_products else products
    pair_features = [f for f in (pair_features or REFINED_PAIR_FEATURES) if f in features.columns]
    work = features[features["product"].isin(products)].copy()
    rows: List[Dict[str, Any]] = []

    residual_feature_cols = []
    for feature in pair_features:
        resid_col = f"{feature}__xresid"
        work[resid_col] = work[feature] - work.groupby(["day", "timestamp"])[feature].transform("mean")
        residual_feature_cols.append((feature, resid_col))

    for horizon in horizons:
        target_col = f"target_{horizon}"
        target_resid_col = f"{target_col}__xresid"
        work[target_col] = work.groupby(["day", "product"])["mid_price"].shift(-horizon) - work["mid_price"]
        work[target_resid_col] = work[target_col] - work.groupby(["day", "timestamp"])[target_col].transform("mean")

        grouped = {key: grp for key, grp in work.groupby(["day", "product"], sort=False)}
        for leader in products:
            for follower in products:
                if leader == follower:
                    continue
                for feature, resid_feature in residual_feature_cols:
                    metrics = {}
                    for day in [2, 3, 4]:
                        leader_grp = grouped.get((day, leader))
                        follower_grp = grouped.get((day, follower))
                        if leader_grp is None or follower_grp is None:
                            continue
                        merged = leader_grp[["timestamp", resid_feature]].merge(
                            follower_grp[["timestamp", target_resid_col]], on="timestamp", how="inner"
                        )
                        if len(merged) < horizon + 30:
                            continue
                        metrics[day] = day_metric(merged[resid_feature].to_numpy(), merged[target_resid_col].to_numpy(), include_mi=False)
                    base = {
                        "signal_id": signal_id("RPAIR", [leader, follower, feature, horizon]),
                        "scope": "ordered_pair_residual",
                        "family": "residual_lead_lag",
                        "leader": leader,
                        "follower": follower,
                        "feature": feature,
                        "horizon": horizon,
                    }
                    rows.append(aggregate_days(base, metrics))
    return rows


def score_spread_and_mirror(features: pd.DataFrame, products: List[str], horizons: List[int], max_products: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    products = products[:max_products] if max_products else products
    spread_rows: List[Dict[str, Any]] = []
    mirror_rows: List[Dict[str, Any]] = []
    grouped = {key: grp for key, grp in features.groupby(["day", "product"], sort=False)}
    for i, a in enumerate(products):
        for b in products[i + 1 :]:
            for horizon in horizons:
                spread_metrics = {}
                mirror_metrics = {}
                for day in [2, 3, 4]:
                    ga = grouped.get((day, a))
                    gb = grouped.get((day, b))
                    if ga is None or gb is None:
                        continue
                    merged = ga[["timestamp", "mid_price"]].rename(columns={"mid_price": "a"}).merge(
                        gb[["timestamp", "mid_price"]].rename(columns={"mid_price": "b"}), on="timestamp", how="inner"
                    )
                    if len(merged) < horizon + 30:
                        continue
                    spread = merged["a"] - merged["b"]
                    spread_z = (spread - spread.shift(1).rolling(50, min_periods=5).mean()) / spread.shift(1).rolling(50, min_periods=5).std(ddof=0)
                    target = spread.shift(-horizon) - spread
                    # Mean reversion means spread z should predict opposite future spread change.
                    spread_metrics[day] = day_metric((-spread_z).fillna(0.0).to_numpy(), target.to_numpy())
                    mirror_metrics[day] = day_metric(merged["a"].pct_change(1).fillna(0.0).to_numpy(), (-merged["b"].pct_change(horizon).shift(-horizon)).to_numpy())
                spread_rows.append(
                    aggregate_days(
                        {
                            "signal_id": signal_id("SPREAD", [a, b, horizon]),
                            "scope": "pair",
                            "family": "spread_mean_reversion",
                            "product_a": a,
                            "product_b": b,
                            "feature": "negative_spread_z_50",
                            "horizon": horizon,
                        },
                        spread_metrics,
                    )
                )
                mirror_rows.append(
                    aggregate_days(
                        {
                            "signal_id": signal_id("MIRROR", [a, b, horizon]),
                            "scope": "pair",
                            "family": "mirror_anticorrelation",
                            "product_a": a,
                            "product_b": b,
                            "feature": "return_vs_negative_return",
                            "horizon": horizon,
                        },
                        mirror_metrics,
                    )
                )
    return spread_rows, mirror_rows


def score_interactions_and_regimes(feature_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    interaction_rows = [row for row in feature_rows if row.get("family") == "interaction"]
    regime_rows = [row for row in feature_rows if row.get("family") in {"volatility_regime", "spread"}]
    return interaction_rows, regime_rows


def build_signal_catalog(*row_groups: List[Dict[str, Any]], top_k: int = 500) -> List[Dict[str, Any]]:
    rows = []
    for group in row_groups:
        rows.extend(group)
    rows = sorted(rows, key=lambda r: (r.get("tier") == "confirmed", r.get("tier") == "promising", float(r.get("stability_score", 0.0))), reverse=True)
    catalog = []
    for row in rows[:top_k]:
        catalog.append(
            {
                "signal_id": row.get("signal_id"),
                "tier": row.get("tier"),
                "family": row.get("family"),
                "scope": row.get("scope"),
                "stability_score": row.get("stability_score"),
                "horizon": row.get("horizon"),
                "product": row.get("product", ""),
                "leader": row.get("leader", row.get("product_a", "")),
                "follower": row.get("follower", row.get("product_b", "")),
                "feature": row.get("feature"),
                "score_day2": row.get("score_day2"),
                "score_day3": row.get("score_day3"),
                "score_day4": row.get("score_day4"),
                "sign_day2": row.get("sign_day2"),
                "sign_day3": row.get("sign_day3"),
                "sign_day4": row.get("sign_day4"),
                "min_n": row.get("min_n"),
            }
        )
    return catalog

