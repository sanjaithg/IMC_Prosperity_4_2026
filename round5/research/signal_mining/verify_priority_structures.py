from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.signal_mining.data import discover_data_dir, load_prices
from analysis.signal_mining.reporting import atomic_write_csv, atomic_write_json, atomic_write_text, utc_now


DAYS = [2, 3, 4]
SNACKS = [
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
]
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify and translate high-priority structural claims.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "signal_mining_refined" / "priority_checks"))
    return parser.parse_args()


def zscore_by_day(series: pd.Series, window: int = 200) -> pd.Series:
    return series.groupby(level=0).transform(
        lambda x: (x - x.shift(1).rolling(window, min_periods=max(20, window // 5)).mean())
        / x.shift(1).rolling(window, min_periods=max(20, window // 5)).std(ddof=0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def forward_delta(mid: pd.DataFrame, product: str, horizon: int) -> pd.Series:
    return mid[product].groupby(level=0).shift(-horizon) - mid[product]


def bucket_stats(signal: pd.Series, target: pd.Series, thresholds: Iterable[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    aligned = pd.DataFrame({"signal": signal, "target": target}).replace([np.inf, -np.inf], np.nan).dropna()
    for threshold in thresholds:
        for side, mask in [
            ("long_signal", aligned["signal"] >= threshold),
            ("short_signal", aligned["signal"] <= -threshold),
        ]:
            sample = aligned[mask]
            if sample.empty:
                rows.append({"threshold": threshold, "side": side, "n": 0})
                continue
            signed = sample["target"] if side == "long_signal" else -sample["target"]
            rows.append(
                {
                    "threshold": threshold,
                    "side": side,
                    "n": int(len(sample)),
                    "mean_signed_move": float(signed.mean()),
                    "median_signed_move": float(signed.median()),
                    "hit_rate": float((signed > 0).mean()),
                    "p10": float(signed.quantile(0.10)),
                    "p90": float(signed.quantile(0.90)),
                }
            )
    return rows


def pair_reversion_rows(mid: pd.DataFrame, a: str, b: str, horizons: Iterable[int], windows: Iterable[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pair_sum = mid[a] + mid[b]
    spread = mid[a] - mid[b]
    for basis_name, basis in [("sum", pair_sum), ("spread", spread)]:
        for window in windows:
            z = zscore_by_day(basis, window)
            for horizon in horizons:
                target = basis.groupby(level=0).shift(-horizon) - basis
                # Reversion: positive z should predict negative future change in basis.
                rows.extend(
                    {
                        **row,
                        "pair_a": a,
                        "pair_b": b,
                        "basis": basis_name,
                        "window": window,
                        "horizon": horizon,
                    }
                    for row in bucket_stats(-z, target, thresholds=[1.0, 1.5, 2.0, 2.5, 3.0])
                )
    return rows


def snackpack_pressure_rows(mid: pd.DataFrame, prices: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for product in SNACKS:
        frame = prices[prices["product"].eq(product)].sort_values(["day", "timestamp"]).copy()
        frame["spread"] = frame["ask_price_1"] - frame["bid_price_1"]
        frame["imbalance_l1"] = (frame["bid_volume_1"].abs() - frame["ask_volume_1"].abs()) / (
            frame["bid_volume_1"].abs() + frame["ask_volume_1"].abs()
        ).replace(0, np.nan)
        weighted_bid = frame["bid_volume_1"].abs() * 3 + frame["bid_volume_2"].abs() * 2 + frame["bid_volume_3"].abs()
        weighted_ask = frame["ask_volume_1"].abs() * 3 + frame["ask_volume_2"].abs() * 2 + frame["ask_volume_3"].abs()
        frame["weighted_imbalance"] = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask).replace(0, np.nan)
        frame["microprice"] = (
            frame["ask_price_1"] * frame["bid_volume_1"].abs()
            + frame["bid_price_1"] * frame["ask_volume_1"].abs()
        ) / (frame["bid_volume_1"].abs() + frame["ask_volume_1"].abs()).replace(0, np.nan)
        frame["microprice_edge_rel"] = (frame["microprice"] - frame["mid_price"]) / frame["spread"].abs().clip(lower=1)
        frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame = frame.set_index(["day", "timestamp"])
        for feature in ["weighted_imbalance", "imbalance_l1", "microprice_edge_rel"]:
            for horizon in [1, 3, 5, 10]:
                target = forward_delta(mid, product, horizon)
                for row in bucket_stats(frame[feature], target, thresholds=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]):
                    rows.append({"product": product, "feature": feature, "horizon": horizon, **row})
    return rows


def pebbles_constraint_rows(mid: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    total = mid[PEBBLES].sum(axis=1)
    summary_rows.append(
        {
            "structure": "PEBBLES_ALL_SUM",
            "mean": float(total.mean()),
            "std": float(total.std(ddof=0)),
            "min": float(total.min()),
            "max": float(total.max()),
            "cv_pct": float(total.std(ddof=0) / abs(total.mean()) * 100),
        }
    )
    pair_rows = []
    for a, b in [("PEBBLES_S", "PEBBLES_XL"), ("PEBBLES_XS", "PEBBLES_XL"), ("PEBBLES_S", "PEBBLES_L")]:
        pair_rows.extend(pair_reversion_rows(mid, a, b, horizons=[1, 3, 5, 10, 25, 50], windows=[50, 100, 200]))
    return summary_rows, pair_rows


def microchip_volatility_rows(mid: pd.DataFrame, spread: pd.DataFrame, product: str = "MICROCHIP_OVAL") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prod_mid = mid[product]
    prod_spread = spread[product]
    abs_diff = prod_mid.groupby(level=0).diff().abs()
    for smooth in [1, 20, 50, 100, 200]:
        smooth_spread = (
            prod_spread.groupby(level=0).transform(lambda x: x.rolling(smooth, min_periods=max(1, min(5, smooth))).mean())
            if smooth > 1
            else prod_spread
        )
        for horizon in [25, 50, 100, 200]:
            future_abs_move = prod_mid.groupby(level=0).shift(-horizon).sub(prod_mid).abs()
            future_realized = abs_diff.groupby(level=0).transform(
                lambda x, h=horizon: x.shift(-h).rolling(h, min_periods=max(5, h // 5)).mean()
            )
            rows.append(
                {
                    "product": product,
                    "smooth_spread_window": smooth,
                    "horizon": horizon,
                    "corr_future_abs_move": float(smooth_spread.corr(future_abs_move)),
                    "corr_future_realized_absdiff": float(smooth_spread.corr(future_realized)),
                }
            )
    return rows


def write_report(
    out_dir: Path,
    snack_rows: List[Dict[str, Any]],
    snack_constraint_rows: List[Dict[str, Any]],
    pebble_pairs: List[Dict[str, Any]],
    vol_rows: List[Dict[str, Any]],
) -> None:
    snack = pd.DataFrame(snack_rows)
    snack_constraints = pd.DataFrame(snack_constraint_rows)
    pebbles = pd.DataFrame(pebble_pairs)
    vol = pd.DataFrame(vol_rows)

    best_snack = snack[(snack["n"] >= 200) & (snack["mean_signed_move"] > 0)].sort_values(
        ["mean_signed_move", "hit_rate"], ascending=False
    ).head(12)
    best_snack_constraints = snack_constraints[
        (snack_constraints["n"] >= 100) & (snack_constraints["mean_signed_move"] > 0)
    ].sort_values(["mean_signed_move", "hit_rate"], ascending=False).head(12)
    best_pebbles = pebbles[(pebbles["n"] >= 100) & (pebbles["mean_signed_move"] > 0)].sort_values(
        ["mean_signed_move", "hit_rate"], ascending=False
    ).head(12)
    best_vol = vol.sort_values("corr_future_realized_absdiff", ascending=False).head(8)

    lines = [
        "# Priority Structure Translation",
        "",
        f"- generated_at: `{utc_now()}`",
        "- scope: verified uploaded claims translated into immediate test candidates",
        "",
        "## Snackpack Pressure",
        "",
        "Best threshold rows by mean signed move:",
        "",
    ]
    for row in best_snack.to_dict("records"):
        lines.append(
            f"- `{row['product']}` `{row['feature']}` h={row['horizon']} threshold={row['threshold']} "
            f"side={row['side']} n={row['n']} mean={row['mean_signed_move']:.4f} hit={row['hit_rate']:.3f}"
        )
    lines.extend(["", "## Snackpack Constraint Reversion", ""])
    for row in best_snack_constraints.to_dict("records"):
        lines.append(
            f"- `{row['pair_a']}/{row['pair_b']}` basis={row['basis']} window={row['window']} h={row['horizon']} "
            f"threshold={row['threshold']} side={row['side']} n={row['n']} mean={row['mean_signed_move']:.4f} hit={row['hit_rate']:.3f}"
        )
    lines.extend(["", "## Pebbles Constraint Reversion", ""])
    for row in best_pebbles.to_dict("records"):
        lines.append(
            f"- `{row['pair_a']}/{row['pair_b']}` basis={row['basis']} window={row['window']} h={row['horizon']} "
            f"threshold={row['threshold']} side={row['side']} n={row['n']} mean={row['mean_signed_move']:.4f} hit={row['hit_rate']:.3f}"
        )
    lines.extend(["", "## Microchip Volatility Regime", ""])
    for row in best_vol.to_dict("records"):
        lines.append(
            f"- `{row['product']}` smooth={row['smooth_spread_window']} h={row['horizon']} "
            f"corr_realized_vol={row['corr_future_realized_absdiff']:.3f} corr_abs_move={row['corr_future_abs_move']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Suggested Ordering",
            "",
            "1. Test `SNACKPACK` pressure as take-only directional entries.",
            "2. Test `SNACKPACK_CHOCOLATE/SNACKPACK_VANILLA` constraint as pair reversion.",
            "3. Test `PEBBLES_S/PEBBLES_XL` and all-pebble sum guards as stat-arb filters.",
            "4. Use `MICROCHIP_OVAL` spread only as a volatility/regime filter unless directional tests pass separately.",
        ]
    )
    atomic_write_text(out_dir / "priority_translation_report.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = discover_data_dir(args.data_dir)
    prices = load_prices(data_dir, DAYS)
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    idx = ["day", "timestamp"]
    mid = prices.pivot_table(index=idx, columns="product", values="mid_price", aggfunc="last").sort_index()
    spread = prices.pivot_table(index=idx, columns="product", values="spread", aggfunc="last").sort_index()

    snack_rows = snackpack_pressure_rows(mid, prices)
    snack_constraint_rows = []
    snack_constraint_rows.extend(pair_reversion_rows(mid, "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", horizons=[1, 3, 5, 10, 25, 50], windows=[50, 100, 200]))
    snack_constraint_rows.extend(pair_reversion_rows(mid, "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY", horizons=[1, 3, 5, 10, 25, 50], windows=[50, 100, 200]))
    pebbles_summary, pebbles_pairs = pebbles_constraint_rows(mid)
    vol_rows = microchip_volatility_rows(mid, spread)

    atomic_write_csv(out_dir / "snackpack_pressure_thresholds.csv", snack_rows)
    atomic_write_csv(out_dir / "snackpack_constraint_reversion.csv", snack_constraint_rows)
    atomic_write_csv(out_dir / "pebbles_summary.csv", pebbles_summary)
    atomic_write_csv(out_dir / "pebbles_pair_reversion.csv", pebbles_pairs)
    atomic_write_csv(out_dir / "microchip_oval_spread_volatility.csv", vol_rows)
    atomic_write_json(
        out_dir / "run_manifest.json",
        {
            "generated_at": utc_now(),
            "data_dir": str(data_dir),
            "days": DAYS,
            "outputs": [
                "snackpack_pressure_thresholds.csv",
                "snackpack_constraint_reversion.csv",
                "pebbles_summary.csv",
                "pebbles_pair_reversion.csv",
                "microchip_oval_spread_volatility.csv",
                "priority_translation_report.md",
            ],
        },
    )
    write_report(out_dir, snack_rows, snack_constraint_rows, pebbles_pairs, vol_rows)
    print(f"wrote priority checks to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

