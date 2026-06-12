from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


HORIZONS = [1, 3, 5, 10, 25, 50, 100]
ROLL_WINDOWS = [5, 20, 50, 100]


def _safe_div(num, den):
    den = den.replace(0, np.nan) if hasattr(den, "replace") else den
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _group_roll(series: pd.Series, window: int, func: str, min_periods: int = 2) -> pd.Series:
    rolled = series.shift(1).rolling(window, min_periods=min_periods)
    if func == "mean":
        return rolled.mean()
    if func == "std":
        return rolled.std(ddof=0)
    if func == "median":
        return rolled.median()
    if func == "min":
        return rolled.min()
    if func == "max":
        return rolled.max()
    raise ValueError(func)


def add_trade_features(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        for col in ["trade_count", "trade_qty", "avg_trade_size", "trade_intensity_20"]:
            prices[col] = 0.0
        return prices
    agg = (
        trades.groupby(["day", "timestamp", "product"], as_index=False)
        .agg(trade_count=("quantity", "size"), trade_qty=("quantity", "sum"), avg_trade_size=("quantity", "mean"))
    )
    out = prices.merge(agg, on=["day", "timestamp", "product"], how="left")
    for col in ["trade_count", "trade_qty", "avg_trade_size"]:
        out[col] = out[col].fillna(0.0)
    out["trade_intensity_20"] = out.groupby(["day", "product"])["trade_count"].transform(
        lambda s: _group_roll(s, 20, "mean", min_periods=1)
    ).fillna(0.0)
    return out


def build_feature_table(prices: pd.DataFrame, trades: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    df = prices.copy()
    df = df.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)
    bid_price_cols = [c for c in ["bid_price_1", "bid_price_2", "bid_price_3"] if c in df.columns]
    ask_price_cols = [c for c in ["ask_price_1", "ask_price_2", "ask_price_3"] if c in df.columns]
    bid_vol_cols = [c for c in ["bid_volume_1", "bid_volume_2", "bid_volume_3"] if c in df.columns]
    ask_vol_cols = [c for c in ["ask_volume_1", "ask_volume_2", "ask_volume_3"] if c in df.columns]

    for col in bid_vol_cols + ask_vol_cols:
        df[col] = df[col].fillna(0).abs()

    df["spread"] = (df["ask_price_1"] - df["bid_price_1"]).astype(float)
    df["rel_spread"] = _safe_div(df["spread"], df["mid_price"].astype(float))
    df["bid_total_vol"] = df[bid_vol_cols].sum(axis=1)
    df["ask_total_vol"] = df[ask_vol_cols].sum(axis=1)
    df["total_depth"] = df["bid_total_vol"] + df["ask_total_vol"]
    df["imbalance_l1"] = _safe_div(df["bid_volume_1"] - df["ask_volume_1"], df["bid_volume_1"] + df["ask_volume_1"])
    df["imbalance_all"] = _safe_div(df["bid_total_vol"] - df["ask_total_vol"], df["total_depth"])

    for i in (1, 2, 3):
        bv = f"bid_volume_{i}"
        av = f"ask_volume_{i}"
        if bv in df.columns and av in df.columns:
            df[f"imbalance_l{i}"] = _safe_div(df[bv] - df[av], df[bv] + df[av])
    if all(c in df.columns for c in ["bid_volume_1", "bid_volume_2", "bid_volume_3", "ask_volume_1", "ask_volume_2", "ask_volume_3"]):
        weighted_bid = df["bid_volume_1"] * 3 + df["bid_volume_2"] * 2 + df["bid_volume_3"]
        weighted_ask = df["ask_volume_1"] * 3 + df["ask_volume_2"] * 2 + df["ask_volume_3"]
        df["weighted_imbalance"] = _safe_div(weighted_bid - weighted_ask, weighted_bid + weighted_ask)
        df["near_far_depth_ratio"] = _safe_div(df["bid_volume_1"] + df["ask_volume_1"], df["bid_volume_3"] + df["ask_volume_3"])
        df["depth_slope_bid"] = df["bid_volume_1"] - df["bid_volume_3"]
        df["depth_slope_ask"] = df["ask_volume_1"] - df["ask_volume_3"]
        df["depth_convexity"] = (df["bid_volume_1"] - 2 * df["bid_volume_2"] + df["bid_volume_3"]) - (
            df["ask_volume_1"] - 2 * df["ask_volume_2"] + df["ask_volume_3"]
        )

    df["missing_bid_levels"] = sum((df[c].isna() | (df[c] == 0)).astype(int) for c in bid_price_cols)
    df["missing_ask_levels"] = sum((df[c].isna() | (df[c] == 0)).astype(int) for c in ask_price_cols)
    df["liquidity_wall_bid"] = df[bid_vol_cols].max(axis=1)
    df["liquidity_wall_ask"] = df[ask_vol_cols].max(axis=1)
    df["liquidity_wall_imbalance"] = _safe_div(df["liquidity_wall_bid"] - df["liquidity_wall_ask"], df["liquidity_wall_bid"] + df["liquidity_wall_ask"])

    df["microprice"] = _safe_div(
        df["ask_price_1"] * df["bid_volume_1"] + df["bid_price_1"] * df["ask_volume_1"],
        df["bid_volume_1"] + df["ask_volume_1"],
    )
    df["microprice_edge"] = df["microprice"] - df["mid_price"]
    df["microprice_edge_rel"] = _safe_div(df["microprice_edge"], df["spread"].abs().clip(lower=1))

    g = df.groupby(["day", "product"], group_keys=False)
    df["ret_1"] = g["mid_price"].pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for h in HORIZONS:
        df[f"ret_{h}"] = g["mid_price"].pct_change(h).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[f"mid_delta_{h}"] = g["mid_price"].diff(h).fillna(0.0)

    dynamic_cols = [
        "bid_total_vol",
        "ask_total_vol",
        "total_depth",
        "spread",
        "imbalance_l1",
        "imbalance_all",
        "weighted_imbalance",
        "microprice_edge",
    ]
    for col in [c for c in dynamic_cols if c in df.columns]:
        df[f"{col}_delta_1"] = g[col].diff(1).fillna(0.0)
        df[f"{col}_delta_5"] = g[col].diff(5).fillna(0.0)

    for window in ROLL_WINDOWS:
        df[f"volatility_{window}"] = g["ret_1"].transform(lambda s, w=window: _group_roll(s, w, "std")).fillna(0.0)
        df[f"ret_mean_{window}"] = g["ret_1"].transform(lambda s, w=window: _group_roll(s, w, "mean")).fillna(0.0)
        df[f"spread_mean_{window}"] = g["spread"].transform(lambda s, w=window: _group_roll(s, w, "mean")).fillna(0.0)
        df[f"spread_std_{window}"] = g["spread"].transform(lambda s, w=window: _group_roll(s, w, "std")).fillna(0.0)
        df[f"spread_z_{window}"] = _safe_div(df["spread"] - df[f"spread_mean_{window}"], df[f"spread_std_{window}"])
        df[f"depth_mean_{window}"] = g["total_depth"].transform(lambda s, w=window: _group_roll(s, w, "mean")).fillna(0.0)
        df[f"depth_z_{window}"] = _safe_div(df["total_depth"] - df[f"depth_mean_{window}"], df[f"depth_mean_{window}"].abs())
        df[f"microprice_edge_mean_{window}"] = g["microprice_edge"].transform(lambda s, w=window: _group_roll(s, w, "mean")).fillna(0.0)
        df[f"microprice_edge_std_{window}"] = g["microprice_edge"].transform(lambda s, w=window: _group_roll(s, w, "std")).fillna(0.0)
        df[f"microprice_edge_z_{window}"] = _safe_div(
            df["microprice_edge"] - df[f"microprice_edge_mean_{window}"],
            df[f"microprice_edge_std_{window}"],
        )
        df[f"microprice_edge_demeaned_{window}"] = df["microprice_edge"] - df[f"microprice_edge_mean_{window}"]
        med = g["ret_1"].transform(lambda s, w=window: _group_roll(s, w, "median")).fillna(0.0)
        mad = g["ret_1"].transform(lambda s, w=window: (s.shift(1) - s.shift(1).rolling(w, min_periods=2).median()).abs().rolling(w, min_periods=2).median()).fillna(0.0)
        df[f"ret_mad_z_{window}"] = _safe_div(df["ret_1"] - med, mad)

    df["vol_of_vol_50"] = g["volatility_20"].transform(lambda s: _group_roll(s, 50, "std")).fillna(0.0)
    df["jump_flag"] = (df["ret_mad_z_50"].abs() > 4).astype(int)
    df["spread_widening"] = (df["spread_delta_1"] > 0).astype(int)
    df["book_thinning"] = (df["total_depth_delta_1"] < 0).astype(int)
    df["timestamp_bucket"] = pd.cut(df["timestamp"], bins=10, labels=False, include_lowest=True).fillna(0).astype(int)

    if trades is not None:
        df = add_trade_features(df, trades)
    else:
        for col in ["trade_count", "trade_qty", "avg_trade_size", "trade_intensity_20"]:
            df[col] = 0.0

    interaction_pairs = {
        "imbalance_x_spread": ("imbalance_l1", "rel_spread"),
        "imbalance_x_volatility": ("imbalance_l1", "volatility_50"),
        "spread_x_depth": ("rel_spread", "total_depth"),
        "microprice_x_spread_z": ("microprice_edge_rel", "spread_z_50"),
        "momentum_x_book_thinning": ("ret_10", "book_thinning"),
    }
    for out, (a, b) in interaction_pairs.items():
        if a in df.columns and b in df.columns:
            df[out] = df[a].fillna(0.0) * df[b].fillna(0.0)

    # Lightweight spectral/cycle proxies that avoid expensive per-row FFT.
    df["autocorr_proxy_20"] = g["ret_1"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).corr(s.shift(2))).fillna(0.0)
    df["oscillation_proxy"] = (np.sign(df["ret_1"]) != np.sign(g["ret_1"].shift(1))).astype(int)

    family_map = {
        "price": [c for c in df.columns if c.startswith("ret_") or c.startswith("mid_delta_")],
        "spread": [c for c in df.columns if "spread" in c],
        "volume_depth": [c for c in df.columns if "depth" in c or "volume" in c or "imbalance" in c or "liquidity" in c],
        "book_dynamics": [c for c in df.columns if c.endswith("_delta_1") or c.endswith("_delta_5") or c in ["book_thinning", "spread_widening"]],
        "microprice": [c for c in df.columns if "microprice" in c],
        "volatility_regime": [c for c in df.columns if "volatility" in c or "vol_of_vol" in c or c in ["jump_flag", "timestamp_bucket"]],
        "trade_flow": ["trade_count", "trade_qty", "avg_trade_size", "trade_intensity_20"],
        "robust": [c for c in df.columns if "mad_z" in c],
        "spectral_cycle": ["autocorr_proxy_20", "oscillation_proxy"],
        "interaction": list(interaction_pairs.keys()),
    }
    feature_cols = []
    for cols in family_map.values():
        feature_cols.extend([c for c in cols if c in df.columns])
    feature_cols = sorted(set(feature_cols))
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0), feature_cols, family_map

