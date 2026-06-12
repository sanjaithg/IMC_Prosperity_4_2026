"""
Round 5 overnight alpha search and verification loop.

The runner is intentionally conservative:
- fixed baseline and folds are captured once in the manifest,
- candidate failures are logged and skipped,
- all critical artifacts are written atomically,
- a lock file prevents concurrent writers.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import pprint
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtester import (  # noqa: E402
    discover_available_days,
    discover_data_dir,
    load_day,
    load_trader,
    run_backtest_single_day,
)


BASELINE_SUBMISSION = ROOT / "Submissions_Derek" / "v1_cluster_mm.py"
TEMPLATE_SUBMISSION = ROOT / "Submissions_Derek" / "v2_alpha_sandbox.py"
RUNNER_PATH = ROOT / "analysis" / "overnight_alpha_loop.py"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
DEFAULT_SEED = 20260429
STOP_PRECEDENCE = [
    "user_stop_or_lock_removed",
    "time_cap",
    "max_validated",
    "stale_limit",
    "cycle_limit",
    "candidate_space_exhausted",
]

CLUSTER_PRODUCTS = {
    "snackpack_pistrawb": [
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ],
    "snackpack_choc_van": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
    ],
    "pebbles_body": [
        "PEBBLES_L",
        "PEBBLES_M",
        "PEBBLES_S",
        "PEBBLES_XS",
        "PEBBLES_XL",
    ],
}

ENABLED_CLUSTER_MAP = {
    "snackpack_pistrawb": ["snackpack_pistrawb", "snackpack_rasp_mirror"],
    "snackpack_choc_van": ["snackpack_choc_van"],
    "pebbles_body": ["pebbles_body", "pebbles_xl_mirror"],
}

MIRROR_PRODUCTS = {
    "SNACKPACK_RASPBERRY",
    "PEBBLES_XL",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def hash_text(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def file_hash(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def code_version_hash() -> str:
    h = hashlib.sha256()
    for path in [BASELINE_SUBMISSION, TEMPLATE_SUBMISSION, RUNNER_PATH]:
        if path.exists():
            h.update(path.name.encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    os.replace(tmp, path)


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class RunLock:
    path: Path
    token: str
    force: bool = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.force and self.path.exists():
            self.path.unlink()
        payload = {
            "pid": os.getpid(),
            "token": self.token,
            "created_at": utc_now(),
        }
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"Run lock exists at {self.path}. Refusing to start a second writer.\n"
                f"Existing lock:\n{existing}"
            ) from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    def still_owned(self) -> bool:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get("token") == self.token
        except Exception:
            return False

    def release(self) -> None:
        if self.still_owned():
            self.path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-validated", type=int, default=500)
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--no-improvement-limit", type=int, default=80)
    parser.add_argument("--turnover-cap", type=float, default=50.0)
    parser.add_argument("--objective", choices=["beat_baseline", "positive_pnl"], default="beat_baseline")
    parser.add_argument("--positive-total-margin", type=float, default=25_000.0)
    parser.add_argument("--positive-fold-margin", type=float, default=5_000.0)
    parser.add_argument("--max-drawdown-cap", type=float, default=250_000.0)
    parser.add_argument("--product-loss-cap", type=float, default=-100_000.0)
    parser.add_argument("--min-fills-per-day", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=None, help="Optional smoke-test cycle limit.")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-lock", action="store_true")
    parser.add_argument("--proxy-ic-min", type=float, default=0.01)
    parser.add_argument("--proxy-hit-min", type=float, default=0.505)
    parser.add_argument("--use-subprocess", action="store_true")
    parser.add_argument("--no-passive", action="store_true")
    return parser.parse_args()


def discover_prices_data_dir(explicit: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            ROOT / "Datasets",
            ROOT / "Round_5" / "Dataset",
            ROOT / "Dataset",
            ROOT / "Datasets" / "ROUND_5",
            ROOT / "Dataset" / "ROUND_5",
            ROOT / "Data",
            ROOT / "data",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir() and list(candidate.glob("prices_round_*_day_*.csv")):
            return candidate.resolve()
    detected = discover_data_dir()
    if detected:
        path = (ROOT / detected).resolve() if not os.path.isabs(detected) else Path(detected).resolve()
        if path.is_dir() and list(path.glob("prices_round_*_day_*.csv")):
            return path
    return None


def build_folds(days: List[int]) -> List[Dict[str, Any]]:
    sorted_days = sorted(days)
    if {2, 3, 4}.issubset(set(sorted_days)):
        return [
            {"fold_id": "fold_train_2_test_3", "train_days": [2], "test_days": [3]},
            {"fold_id": "fold_train_2_3_test_4", "train_days": [2, 3], "test_days": [4]},
        ]
    folds: List[Dict[str, Any]] = []
    for i in range(1, len(sorted_days)):
        train_days = sorted_days[:i]
        test_day = sorted_days[i]
        folds.append(
            {
                "fold_id": f"fold_train_{'_'.join(map(str, train_days))}_test_{test_day}",
                "train_days": train_days,
                "test_days": [test_day],
            }
        )
    return folds


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    return float((pnl.cummax() - pnl).max())


def metrics_from_frames(pnl_df: pd.DataFrame, fills_df: pd.DataFrame, days: List[int]) -> Dict[str, Any]:
    day_metrics: Dict[str, Dict[str, Any]] = {}
    product_metrics: Dict[str, Dict[str, Any]] = {}
    total_pnl = 0.0
    max_dd = 0.0
    total_ticks = 0
    total_abs_fills = 0.0

    for day in days:
        day_pnl = pnl_df[pnl_df["day"] == day] if not pnl_df.empty else pd.DataFrame()
        day_fills = fills_df[fills_df["day"] == day] if not fills_df.empty and "day" in fills_df.columns else pd.DataFrame()
        final_pnl = float(day_pnl["pnl"].iloc[-1]) if not day_pnl.empty else 0.0
        dd = max_drawdown(day_pnl["pnl"]) if not day_pnl.empty else 0.0
        ticks = int(len(day_pnl))
        abs_fills = float(day_fills["quantity"].abs().sum()) if not day_fills.empty and "quantity" in day_fills.columns else 0.0
        turnover = abs_fills / ticks if ticks > 0 else 0.0
        day_metrics[str(day)] = {
            "pnl": final_pnl,
            "max_drawdown": dd,
            "ticks": ticks,
            "abs_filled_qty": abs_fills,
            "turnover": turnover,
            "fills": int(len(day_fills)),
        }
        if not day_pnl.empty:
            final_row = day_pnl.iloc[-1]
            for col in [c for c in day_pnl.columns if c.startswith("ppnl_")]:
                product = col[len("ppnl_") :]
                pm = product_metrics.setdefault(
                    product,
                    {"pnl": 0.0, "abs_filled_qty": 0.0, "fills": 0, "days": {}},
                )
                product_pnl = float(final_row.get(col, 0.0) or 0.0)
                product_fills = day_fills[day_fills["product"] == product] if not day_fills.empty and "product" in day_fills.columns else pd.DataFrame()
                product_abs_fills = float(product_fills["quantity"].abs().sum()) if not product_fills.empty and "quantity" in product_fills.columns else 0.0
                pm["pnl"] += product_pnl
                pm["abs_filled_qty"] += product_abs_fills
                pm["fills"] += int(len(product_fills))
                pm["days"][str(day)] = {
                    "pnl": product_pnl,
                    "abs_filled_qty": product_abs_fills,
                    "fills": int(len(product_fills)),
                }
        total_pnl += final_pnl
        max_dd = max(max_dd, dd)
        total_ticks += ticks
        total_abs_fills += abs_fills

    return {
        "pnl": total_pnl,
        "max_drawdown": max_dd,
        "ticks": total_ticks,
        "abs_filled_qty": total_abs_fills,
        "turnover": total_abs_fills / total_ticks if total_ticks > 0 else 0.0,
        "days": day_metrics,
        "products": product_metrics,
    }


def no_trade_metrics(days: List[int], folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "pnl": 0.0,
        "max_drawdown": 0.0,
        "ticks": 0,
        "abs_filled_qty": 0.0,
        "turnover": 0.0,
        "days": {
            str(day): {
                "pnl": 0.0,
                "max_drawdown": 0.0,
                "ticks": 0,
                "abs_filled_qty": 0.0,
                "turnover": 0.0,
                "fills": 0,
            }
            for day in days
        },
        "products": {},
    }
    return {"summary": summary, "folds": fold_metrics_from_day_metrics(summary, folds)}


def subprocess_backtest(
    submission: Path,
    data_dir: Path,
    days: List[int],
    artifacts_dir: Path,
    no_passive: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp_dir = artifacts_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / f"pnl_{submission.stem}_{hash_text(str(days) + str(time.time()), 8)}.csv"
    fills_path = tmp_dir / f"fills_{submission.stem}_{hash_text(str(days) + str(time.time()), 8)}.csv"
    cmd = [
        sys.executable,
        str(ROOT / "backtester.py"),
        "--submission",
        str(submission),
        "--data-dir",
        str(data_dir),
        "--days",
        *[str(d) for d in days],
        "--out",
        str(out_path),
        "--fills-out",
        str(fills_path),
        "--quiet",
    ]
    if no_passive:
        cmd.append("--no-passive")
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Subprocess backtest failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    pnl_df = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    fills_df = pd.read_csv(fills_path) if fills_path.exists() else pd.DataFrame()
    return pnl_df, fills_df


def run_backtest(
    submission: Path,
    data_dir: Path,
    days: List[int],
    artifacts_dir: Path,
    no_passive: bool,
    force_subprocess: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if force_subprocess:
        pnl, fills = subprocess_backtest(submission, data_dir, days, artifacts_dir, no_passive)
        return pnl, fills, "subprocess"

    try:
        all_pnl: List[pd.DataFrame] = []
        all_fills: List[pd.DataFrame] = []
        with contextlib.redirect_stdout(io.StringIO()):
            for day in days:
                trader = load_trader(str(submission))
                prices, trades, round_id = load_day(str(data_dir), day)
                pnl_df, fills_df, _, _ = run_backtest_single_day(
                    trader,
                    prices,
                    trades,
                    day=day,
                    round_id=round_id,
                    passive_fills=not no_passive,
                    verbose=False,
                )
                all_pnl.append(pnl_df)
                all_fills.append(fills_df)
        pnl = pd.concat(all_pnl, ignore_index=True) if all_pnl else pd.DataFrame()
        fills = pd.concat(all_fills, ignore_index=True) if all_fills else pd.DataFrame()
        return pnl, fills, "in_process"
    except Exception:
        pnl, fills = subprocess_backtest(submission, data_dir, days, artifacts_dir, no_passive)
        return pnl, fills, "subprocess_fallback"


def load_prices_for_proxy(data_dir: Path, days: List[int]) -> pd.DataFrame:
    frames = []
    for day in days:
        paths = sorted(data_dir.glob(f"prices_round_*_day_{day}.csv"))
        if not paths:
            continue
        frames.append(pd.read_csv(paths[-1], sep=";"))
    if not frames:
        raise FileNotFoundError(f"No price files found for proxy days {days} under {data_dir}")
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)
    bid = prices["bid_volume_1"].fillna(0).abs()
    ask = prices["ask_volume_1"].fillna(0).abs()
    denom = bid + ask
    prices["imbalance"] = np.where(denom > 0, (bid - ask) / denom, 0.0)
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    return prices


def add_candidate_signal(prices: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    cluster = config["cluster"]
    products = set(config.get("trade_universe") or CLUSTER_PRODUCTS[cluster])
    df = prices[prices["product"].isin(products)].copy()
    if df.empty:
        return df

    pivot = df.pivot_table(index=["day", "timestamp"], columns="product", values="mid_price", aggfunc="last")
    cluster_cols = [p for p in CLUSTER_PRODUCTS[cluster] if p in pivot.columns]
    if not cluster_cols:
        df["signal"] = 0.0
        return df

    signed = pivot[cluster_cols].copy()
    for product in MIRROR_PRODUCTS:
        if product in signed.columns:
            signed[product] = -signed[product]
    factor = signed.mean(axis=1)
    factor.name = "cluster_factor"
    df = df.merge(factor.reset_index(), on=["day", "timestamp"], how="left")

    signed_mid = df["mid_price"].where(~df["product"].isin(MIRROR_PRODUCTS), -df["mid_price"])
    scale = df["spread"].abs().clip(lower=0.5)
    df["rv_z"] = (signed_mid - df["cluster_factor"]) / scale
    df["rv_z"] = df["rv_z"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if config["model"] == "rvz":
        signal = -df["rv_z"]
    elif config["model"] == "imb":
        signal = df["imbalance"]
    else:
        signal = config["rv_signal_weight"] * (-df["rv_z"]) + (1.0 - config["rv_signal_weight"]) * df["imbalance"]
    signal = signal.where(df["rv_z"].abs() >= config["rv_z_gate"], 0.0)
    df["signal"] = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def proxy_screen(prices: pd.DataFrame, config: Dict[str, Any], folds: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    train_days = sorted({d for fold in folds for d in fold["train_days"]})
    horizon = int(config["horizon"])
    df = add_candidate_signal(prices[prices["day"].isin(train_days)], config)
    if df.empty:
        return {"passed": False, "reason": "proxy_no_cluster_rows", "ic": 0.0, "hit_rate": 0.0, "n": 0}

    df["future_delta"] = df.groupby(["day", "product"])["mid_price"].shift(-horizon) - df["mid_price"]
    valid = df[["signal", "future_delta"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 50:
        return {"passed": False, "reason": "proxy_insufficient_samples", "ic": 0.0, "hit_rate": 0.0, "n": int(len(valid))}
    signal = valid["signal"].to_numpy()
    target = valid["future_delta"].to_numpy()
    if np.std(signal) <= 1e-12 or np.std(target) <= 1e-12:
        ic = 0.0
    else:
        ic = float(np.corrcoef(signal, target)[0, 1])
        if math.isnan(ic):
            ic = 0.0
    hit = float(np.mean(np.sign(signal) == np.sign(target)))
    passed = abs(ic) >= args.proxy_ic_min or hit >= args.proxy_hit_min
    reason = "proxy_pass" if passed else f"proxy_weak_ic_hit ic={ic:.5f} hit={hit:.5f}"
    return {"passed": passed, "reason": reason, "ic": ic, "hit_rate": hit, "n": int(len(valid))}


def sparse_universes(cluster: str) -> List[List[str]]:
    products = CLUSTER_PRODUCTS[cluster]
    if cluster == "pebbles_body":
        return [
            ["PEBBLES_XL"],
            ["PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XS"],
            ["PEBBLES_XL", "PEBBLES_L"],
            ["PEBBLES_XL", "PEBBLES_M"],
            products,
        ]
    if cluster == "snackpack_choc_van":
        return [
            ["SNACKPACK_CHOCOLATE"],
            ["SNACKPACK_VANILLA"],
            products,
        ]
    return [
        ["SNACKPACK_PISTACHIO"],
        ["SNACKPACK_STRAWBERRY"],
        ["SNACKPACK_RASPBERRY"],
        ["SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"],
        products,
    ]


def candidate_search_space(
    seed: int,
    code_hash: str,
    baseline_hash: str,
    folds: List[Dict[str, Any]],
    objective: str = "beat_baseline",
) -> Iterable[Dict[str, Any]]:
    rng = random.Random(seed)
    if objective == "positive_pnl":
        clusters = ["pebbles_body", "snackpack_choc_van", "snackpack_pistrawb"]
        models = ["blend", "rvz", "imb"]
        horizons = [3, 10, 25, 50, 100]
        alpha_mids = [0.04, 0.06, 0.10]
        alpha_imbs = [0.25, 0.35, 0.45]
        rv_tilts = [0.10, 0.25, 0.40]
        imb_takes = [0.35, 0.48, 0.65]
        z_gates = [0.5, 1.0, 1.5, 2.0]
        clips = [1, 2]
        quote_modes = ["take_only", "passive_only_when_signal", "hybrid_sparse"]
        min_signals = [0.35, 0.50, 0.75, 1.00]
        cooldowns = [0, 3, 10, 25]
    else:
        clusters = ["snackpack_pistrawb", "snackpack_choc_van", "pebbles_body"]
        models = ["rvz", "blend", "imb"]
        horizons = [1, 3, 5, 10, 25, 50, 100]
        alpha_mids = [0.06, 0.10, 0.16]
        alpha_imbs = [0.18, 0.25, 0.35]
        rv_tilts = [0.15, 0.25, 0.40]
        imb_takes = [0.35, 0.40, 0.48]
        z_gates = [0.0, 0.5, 1.0]
        clips = [2, 4]
        quote_modes = ["legacy"]
        min_signals = [0.0]
        cooldowns = [0]
    combos = []
    for cluster in clusters:
        for model in models:
            for horizon in horizons:
                for alpha_mid in alpha_mids:
                    for alpha_imb in alpha_imbs:
                        for rv_tilt in rv_tilts:
                            for imb_take in imb_takes:
                                for z_gate in z_gates:
                                    for clip in clips:
                                        for quote_mode in quote_modes:
                                            for min_signal in min_signals:
                                                for cooldown in cooldowns:
                                                    for universe in (sparse_universes(cluster) if objective == "positive_pnl" else [None]):
                                                        combos.append((cluster, model, horizon, alpha_mid, alpha_imb, rv_tilt, imb_take, z_gate, clip, quote_mode, min_signal, cooldown, universe))
    rng.shuffle(combos)

    for cluster, model, horizon, alpha_mid, alpha_imb, rv_tilt, imb_take, z_gate, clip, quote_mode, min_signal, cooldown, universe in combos:
        rv_signal_weight = 0.65 if model == "rvz" else 0.35 if model == "blend" else 0.10
        config: Dict[str, Any] = {
            "model": model,
            "cluster": cluster,
            "objective": objective,
            "horizon": horizon,
            "seed": seed,
            "pos_limit": 10,
            "lean_threshold": 6,
            "clip": clip,
            "alpha_mid": alpha_mid,
            "alpha_imb": alpha_imb,
            "alpha_clu": 0.10,
            "imb_take": imb_take,
            "rv_tilt_scale": rv_tilt if model != "imb" else 0.05,
            "rv_signal_weight": rv_signal_weight,
            "imbalance_tilt_scale": 0.50,
            "position_tilt_scale": 0.30,
            "rv_z_gate": z_gate,
            "enabled_clusters": ENABLED_CLUSTER_MAP[cluster],
            "trade_universe": universe,
            "quote_mode": quote_mode,
            "min_signal_abs": min_signal,
            "passive_signal_abs": max(min_signal, z_gate),
            "cooldown_ticks": cooldown,
            "max_orders_per_tick": 2 if objective == "positive_pnl" else 10_000,
            "feature_config": {
                "signal": model,
                "horizon": horizon,
                "causal": True,
                "normalization": "same_tick_cluster_factor_no_future",
                "purge_embargo_ticks": horizon,
            },
            "thresholds": {
                "imb_take": imb_take,
                "rv_z_gate": z_gate,
                "min_signal_abs": min_signal,
            },
            "code_version_hash": code_hash,
            "baseline_snapshot_hash": baseline_hash,
            "folds": folds,
        }
        digest = hash_text(stable_json(config), 10)
        config["candidate_id"] = f"CAND_{model}_{cluster}_{horizon}_{digest}"
        yield config


def write_candidate_submission(config: Dict[str, Any], artifacts_dir: Path) -> Path:
    out_dir = artifacts_dir / "generated_submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = config["candidate_id"]
    path = out_dir / f"{candidate_id}.py"
    config_literal = pprint.pformat(config, sort_dicts=True, width=100)
    text = (
        "from Submissions_Derek.v2_alpha_sandbox import Trader as BaseTrader\n\n"
        f"CONFIG = {config_literal}\n\n"
        "class Trader(BaseTrader):\n"
        "    CONFIG = CONFIG\n"
    )
    atomic_write_text(path, text)
    return path


def fold_metrics_from_day_metrics(metrics: Dict[str, Any], folds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for fold in folds:
        fold_days = [str(d) for d in fold["test_days"]]
        days = [metrics["days"].get(day, {}) for day in fold_days]
        pnl = sum(float(d.get("pnl", 0.0)) for d in days)
        max_dd = max([float(d.get("max_drawdown", 0.0)) for d in days] or [0.0])
        ticks = sum(int(d.get("ticks", 0)) for d in days)
        abs_fills = sum(float(d.get("abs_filled_qty", 0.0)) for d in days)
        fills = sum(int(d.get("fills", 0)) for d in days)
        out[fold["fold_id"]] = {
            "test_days": fold["test_days"],
            "pnl": pnl,
            "max_drawdown": max_dd,
            "ticks": ticks,
            "abs_filled_qty": abs_fills,
            "fills": fills,
            "turnover": abs_fills / ticks if ticks > 0 else 0.0,
        }
    return out


def evaluate_gates(
    candidate_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    folds: List[Dict[str, Any]],
    turnover_cap: float,
    leakage_passed: bool,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    fold_results: Dict[str, Any] = {}
    baseline_folds = baseline_metrics["folds"]
    candidate_folds = candidate_metrics["folds"]

    if not leakage_passed:
        reasons.append("leakage_check_failed")
    if candidate_metrics["summary"]["pnl"] <= baseline_metrics["summary"]["pnl"]:
        reasons.append(
            f"oos_pnl_not_above_baseline candidate={candidate_metrics['summary']['pnl']:.2f} "
            f"baseline={baseline_metrics['summary']['pnl']:.2f}"
        )
    baseline_dd = float(baseline_metrics["summary"]["max_drawdown"])
    allowed_dd = baseline_dd * 1.05 if baseline_dd > 0 else baseline_dd + 1e-9
    if candidate_metrics["summary"]["max_drawdown"] > allowed_dd:
        reasons.append(
            f"drawdown_worsened_gt_5pct candidate={candidate_metrics['summary']['max_drawdown']:.2f} "
            f"allowed={allowed_dd:.2f}"
        )
    if candidate_metrics["summary"]["turnover"] > turnover_cap:
        reasons.append(
            f"turnover_above_cap candidate={candidate_metrics['summary']['turnover']:.5f} cap={turnover_cap:.5f}"
        )

    for fold in folds:
        fid = fold["fold_id"]
        cand = candidate_folds[fid]
        base = baseline_folds[fid]
        fold_reasons = []
        if cand["pnl"] <= base["pnl"]:
            fold_reasons.append(f"fold_pnl_not_above_baseline candidate={cand['pnl']:.2f} baseline={base['pnl']:.2f}")
        base_dd = float(base["max_drawdown"])
        fold_allowed_dd = base_dd * 1.05 if base_dd > 0 else base_dd + 1e-9
        if cand["max_drawdown"] > fold_allowed_dd:
            fold_reasons.append(f"fold_drawdown_worsened_gt_5pct candidate={cand['max_drawdown']:.2f} allowed={fold_allowed_dd:.2f}")
        if cand["turnover"] > turnover_cap:
            fold_reasons.append(f"fold_turnover_above_cap candidate={cand['turnover']:.5f} cap={turnover_cap:.5f}")
        fold_results[fid] = {
            "passed": not fold_reasons,
            "reasons": fold_reasons,
            "candidate": cand,
            "baseline": base,
        }
        if fold_reasons:
            reasons.append(f"{fid}: " + "; ".join(fold_reasons))

    return not reasons, reasons, fold_results


def evaluate_positive_gates(
    candidate_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    no_trade: Dict[str, Any],
    folds: List[Dict[str, Any]],
    args: argparse.Namespace,
    leakage_passed: bool,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    fold_results: Dict[str, Any] = {}
    summary = candidate_metrics["summary"]
    candidate_folds = candidate_metrics["folds"]
    baseline_folds = baseline_metrics["folds"]
    no_trade_folds = no_trade["folds"]

    if not leakage_passed:
        reasons.append("leakage_check_failed")
    if summary["pnl"] <= args.positive_total_margin:
        reasons.append(
            f"oos_pnl_not_positive_margin candidate={summary['pnl']:.2f} "
            f"required>{args.positive_total_margin:.2f}"
        )
    if summary["pnl"] <= no_trade["summary"]["pnl"] + args.positive_total_margin:
        reasons.append(
            f"does_not_beat_no_trade_margin candidate={summary['pnl']:.2f} "
            f"required>{no_trade['summary']['pnl'] + args.positive_total_margin:.2f}"
        )
    if summary["pnl"] <= baseline_metrics["summary"]["pnl"]:
        reasons.append(
            f"does_not_beat_v1_baseline candidate={summary['pnl']:.2f} "
            f"baseline={baseline_metrics['summary']['pnl']:.2f}"
        )
    if summary["max_drawdown"] > args.max_drawdown_cap:
        reasons.append(
            f"drawdown_above_cap candidate={summary['max_drawdown']:.2f} "
            f"cap={args.max_drawdown_cap:.2f}"
        )
    if summary["turnover"] > args.turnover_cap:
        reasons.append(
            f"turnover_above_cap candidate={summary['turnover']:.5f} cap={args.turnover_cap:.5f}"
        )

    worst_product = None
    worst_product_pnl = 0.0
    for product, product_metrics in summary.get("products", {}).items():
        product_pnl = float(product_metrics.get("pnl", 0.0))
        if worst_product is None or product_pnl < worst_product_pnl:
            worst_product = product
            worst_product_pnl = product_pnl
    if worst_product is not None and worst_product_pnl < args.product_loss_cap:
        reasons.append(
            f"product_loss_cap_breached product={worst_product} pnl={worst_product_pnl:.2f} "
            f"cap={args.product_loss_cap:.2f}"
        )

    for fold in folds:
        fid = fold["fold_id"]
        cand = candidate_folds[fid]
        base = baseline_folds[fid]
        nt = no_trade_folds[fid]
        fold_reasons = []
        if cand["pnl"] <= args.positive_fold_margin:
            fold_reasons.append(
                f"fold_pnl_not_positive_margin candidate={cand['pnl']:.2f} "
                f"required>{args.positive_fold_margin:.2f}"
            )
        if cand["pnl"] <= nt["pnl"] + args.positive_fold_margin:
            fold_reasons.append(
                f"fold_does_not_beat_no_trade_margin candidate={cand['pnl']:.2f} "
                f"required>{nt['pnl'] + args.positive_fold_margin:.2f}"
            )
        if cand["pnl"] <= base["pnl"]:
            fold_reasons.append(
                f"fold_does_not_beat_v1_baseline candidate={cand['pnl']:.2f} baseline={base['pnl']:.2f}"
            )
        if cand["max_drawdown"] > args.max_drawdown_cap:
            fold_reasons.append(
                f"fold_drawdown_above_cap candidate={cand['max_drawdown']:.2f} cap={args.max_drawdown_cap:.2f}"
            )
        if cand["turnover"] > args.turnover_cap:
            fold_reasons.append(
                f"fold_turnover_above_cap candidate={cand['turnover']:.5f} cap={args.turnover_cap:.5f}"
            )
        min_fills = args.min_fills_per_day * max(len(fold.get("test_days", [])), 1)
        if cand.get("fills", 0) < min_fills:
            fold_reasons.append(f"fold_min_fills_not_met fills={cand.get('fills', 0)} required>={min_fills}")
        fold_results[fid] = {
            "passed": not fold_reasons,
            "reasons": fold_reasons,
            "candidate": cand,
            "baseline": base,
            "no_trade": nt,
        }
        if fold_reasons:
            reasons.append(f"{fid}: " + "; ".join(fold_reasons))

    return not reasons, reasons, fold_results


def candidate_score(summary: Dict[str, Any], baseline_summary: Dict[str, Any], folds: Dict[str, Any], objective: str) -> float:
    if objective == "positive_pnl":
        fold_pnls = [float(f.get("pnl", 0.0)) for f in folds.values()]
        instability = max(fold_pnls) - min(fold_pnls) if fold_pnls else 0.0
        return (
            float(summary["pnl"])
            - 0.50 * float(summary["max_drawdown"])
            - 1_000.0 * float(summary["turnover"])
            - 0.25 * instability
        )
    return (
        float(summary["pnl"])
        - float(baseline_summary["pnl"])
        - 0.25 * max(0.0, float(summary["max_drawdown"]) - float(baseline_summary["max_drawdown"]))
        - 0.01 * float(summary["turnover"])
    )


def initial_manifest(args: argparse.Namespace, artifacts_dir: Path, lock_token: str) -> Dict[str, Any]:
    return {
        "start_time": utc_now(),
        "last_update_time": utc_now(),
        "cycle_index": 0,
        "candidates_generated": 0,
        "candidates_screened": 0,
        "candidates_validated": 0,
        "candidates_passed": 0,
        "current_best_candidate_id": None,
        "best_metrics": {},
        "baseline": None,
        "run_config": vars(args),
        "lock_token": lock_token,
        "stop_precedence": STOP_PRECEDENCE,
        "turnover_definition": "total_abs_filled_quantity / total_backtest_ticks",
        "resume_instructions": (
            "Run `python analysis/overnight_alpha_loop.py --resume` with the same data dir and config. "
            "Existing candidate metrics/rejections will be skipped."
        ),
    }


def load_or_create_manifest(args: argparse.Namespace, artifacts_dir: Path, lock_token: str) -> Dict[str, Any]:
    path = artifacts_dir / "run_manifest.json"
    if args.resume and path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["lock_token"] = lock_token
        manifest["last_update_time"] = utc_now()
        return manifest
    return initial_manifest(args, artifacts_dir, lock_token)


def update_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    manifest["last_update_time"] = utc_now()
    atomic_write_json(path, manifest)


def write_cycle_log(
    artifacts_dir: Path,
    cycle_index: int,
    args: argparse.Namespace,
    lines: List[str],
    manifest: Dict[str, Any],
) -> None:
    text = [
        f"# Cycle {cycle_index}",
        "",
        f"- timestamp: {utc_now()}",
        f"- command_args: `{stable_json(vars(args))}`",
        f"- cycle_index: {manifest.get('cycle_index')}",
        f"- candidates_generated: {manifest.get('candidates_generated')}",
        f"- candidates_screened: {manifest.get('candidates_screened')}",
        f"- candidates_validated: {manifest.get('candidates_validated')}",
        f"- candidates_passed: {manifest.get('candidates_passed')}",
        f"- current_best_candidate_id: {manifest.get('current_best_candidate_id')}",
        "",
        "## Events",
        "",
    ]
    text.extend(f"- {line}" for line in lines)
    atomic_write_text(artifacts_dir / "cycle_logs" / f"cycle_{cycle_index}.md", "\n".join(text) + "\n")


def write_final_report(artifacts_dir: Path, manifest: Dict[str, Any], stop_reason: str) -> None:
    leaderboard = load_csv_rows(artifacts_dir / "leaderboard.csv")
    rejections = load_csv_rows(artifacts_dir / "rejections.csv")
    top = sorted(
        leaderboard,
        key=lambda row: float(row.get("score", row.get("pnl", 0)) or 0),
        reverse=True,
    )[:10]
    total_screened = int(manifest.get("candidates_screened", 0) or 0)
    total_validated = int(manifest.get("candidates_validated", 0) or 0)
    total_passed = int(manifest.get("candidates_passed", 0) or 0)
    report = [
        "# Round 5 Overnight Alpha Loop Final Report",
        "",
        f"- stop_reason: `{stop_reason}`",
        f"- objective: `{manifest.get('run_config', {}).get('objective')}`",
        f"- generated: {manifest.get('candidates_generated', 0)}",
        f"- screened: {total_screened}",
        f"- validated: {total_validated}",
        f"- passed: {total_passed}",
        f"- pass_rate_validated: {(total_passed / total_validated) if total_validated else 0.0:.4f}",
        f"- current_best_candidate_id: `{manifest.get('current_best_candidate_id')}`",
        "",
        "## Baseline",
        "",
        f"```json\n{json.dumps(manifest.get('baseline'), indent=2, sort_keys=True)}\n```",
        "",
        "## No-Trade Baseline",
        "",
        f"```json\n{json.dumps(manifest.get('no_trade_baseline'), indent=2, sort_keys=True)}\n```",
        "",
        "## Top 10 Candidates",
        "",
    ]
    if top:
        for i, row in enumerate(top, start=1):
            report.append(
                f"{i}. `{row.get('candidate_id')}` score={row.get('score')} "
                f"pnl={row.get('pnl')} max_dd={row.get('max_drawdown')} turnover={row.get('turnover')}"
            )
    else:
        report.append("No candidates passed all gates.")
    report.extend(
        [
            "",
            "## Gate Pass Rates",
            "",
            f"- proxy_or_gate_rejections: {len(rejections)}",
            f"- validated_pass_rate: {(total_passed / total_validated) if total_validated else 0.0:.4f}",
            "",
            "## Best Vs Baseline",
            "",
        ]
    )
    best = manifest.get("best_metrics") or {}
    if best:
        report.append(f"Best candidate metrics:\n\n```json\n{json.dumps(best, indent=2, sort_keys=True)}\n```")
    else:
        report.append("No passing candidate beat the fixed baseline snapshot.")
    report.extend(
        [
            "",
            "## Recommended Next Actions",
            "",
            "- If this report is blocked on missing data, place `prices_round_*_day_*.csv` in `Datasets/` or pass `--data-dir` and rerun with `--resume`.",
            "- Review `artifacts/rejections.csv` for dominant failure modes before widening the search space.",
            "- Only promote a candidate after inspecting `artifacts/metrics/<id>.json` fold-level evidence.",
            "- For `positive_pnl`, promote only candidates listed in the leaderboard because less-negative candidates remain rejected.",
        ]
    )
    atomic_write_text(artifacts_dir / "final_report.md", "\n".join(report) + "\n")


def blocked_on_data(args: argparse.Namespace, artifacts_dir: Path, manifest: Dict[str, Any], reason: str) -> None:
    cycle_lines = [reason, "No candidate validation was run."]
    manifest["blocked_reason"] = reason
    update_manifest(artifacts_dir / "run_manifest.json", manifest)
    write_cycle_log(artifacts_dir, 0, args, cycle_lines, manifest)
    write_final_report(artifacts_dir, manifest, "blocked_missing_price_data")


def append_rejection(artifacts_dir: Path, row: Dict[str, Any]) -> None:
    path = artifacts_dir / "rejections.csv"
    rows = load_csv_rows(path)
    rows.append(row)
    fieldnames = [
        "timestamp",
        "candidate_id",
        "stage",
        "reason",
        "pnl",
        "baseline_pnl",
        "max_drawdown",
        "turnover",
        "proxy_ic",
        "proxy_hit_rate",
    ]
    atomic_write_csv(path, rows, fieldnames)


def append_leaderboard(artifacts_dir: Path, row: Dict[str, Any]) -> None:
    path = artifacts_dir / "leaderboard.csv"
    rows = load_csv_rows(path)
    rows.append(row)
    rows = sorted(rows, key=lambda r: float(r.get("score", 0) or 0), reverse=True)
    fieldnames = [
        "timestamp",
        "candidate_id",
        "score",
        "pnl",
        "baseline_pnl",
        "pnl_uplift",
        "no_trade_pnl",
        "max_drawdown",
        "baseline_max_drawdown",
        "turnover",
        "proxy_ic",
        "proxy_hit_rate",
        "submission_path",
    ]
    atomic_write_csv(path, rows, fieldnames)


def processed_candidate_ids(artifacts_dir: Path) -> set:
    done = set()
    for path in (artifacts_dir / "metrics").glob("*.json"):
        done.add(path.stem)
    for row in load_csv_rows(artifacts_dir / "rejections.csv"):
        if row.get("candidate_id"):
            done.add(row["candidate_id"])
    return done


def ensure_dirs(artifacts_dir: Path) -> None:
    for sub in ["candidates", "metrics", "cycle_logs", "generated_submissions", "tmp"]:
        (artifacts_dir / sub).mkdir(parents=True, exist_ok=True)


def archive_existing_run(artifacts_dir: Path) -> Optional[Path]:
    archive_sources = [
        "run_manifest.json",
        "leaderboard.csv",
        "rejections.csv",
        "final_report.md",
        "candidates",
        "metrics",
        "cycle_logs",
        "generated_submissions",
        "tmp",
    ]
    existing = [artifacts_dir / name for name in archive_sources if (artifacts_dir / name).exists()]
    if not existing:
        return None
    archive_root = artifacts_dir / "archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    suffix = 1
    while archive_dir.exists():
        suffix += 1
        archive_dir = archive_root / f"{datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%SZ')}_{suffix}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for source in existing:
        source.rename(archive_dir / source.name)
    return archive_dir


def main() -> int:
    args = parse_args()
    if args.objective == "positive_pnl" and args.artifacts_dir == str(DEFAULT_ARTIFACTS):
        args.artifacts_dir = str(DEFAULT_ARTIFACTS / "positive_pnl_loop")
        if args.turnover_cap == 50.0:
            args.turnover_cap = 3.0
    artifacts_dir = Path(args.artifacts_dir).resolve()
    ensure_dirs(artifacts_dir)
    lock_token = hash_text(f"{os.getpid()}:{time.time()}:{random.random()}", 16)
    lock = RunLock(artifacts_dir / ".run.lock", lock_token, force=args.force_lock)

    try:
        lock.acquire()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    archived_to: Optional[Path] = None
    if not args.resume:
        archived_to = archive_existing_run(artifacts_dir)
        ensure_dirs(artifacts_dir)

    manifest = load_or_create_manifest(args, artifacts_dir, lock_token)
    if archived_to is not None:
        manifest["archived_previous_run"] = str(archived_to)
    manifest_path = artifacts_dir / "run_manifest.json"
    start_monotonic = time.monotonic()
    stop_reason = "unknown"

    try:
        data_dir = discover_prices_data_dir(args.data_dir)
        if data_dir is None:
            blocked_on_data(
                args,
                artifacts_dir,
                manifest,
                "No `prices_round_*_day_*.csv` files found in configured or default data directories.",
            )
            print("Blocked: no price data found. Artifacts written under artifacts/.")
            return 1

        days = discover_available_days(str(data_dir))
        folds = build_folds(days)
        if not folds:
            blocked_on_data(args, artifacts_dir, manifest, f"Not enough price days for OOS folds: {days}")
            return 1

        test_days = sorted({day for fold in folds for day in fold["test_days"]})
        code_hash = code_version_hash()
        manifest["data_dir"] = str(data_dir)
        manifest["available_days"] = days
        manifest["folds"] = folds
        manifest["seed"] = args.seed
        manifest["code_version_hash"] = code_hash
        manifest["baseline_submission"] = rel(BASELINE_SUBMISSION)

        if not manifest.get("baseline"):
            baseline_pnl, baseline_fills, baseline_mode = run_backtest(
                BASELINE_SUBMISSION,
                data_dir,
                test_days,
                artifacts_dir,
                args.no_passive,
                args.use_subprocess,
            )
            baseline_summary = metrics_from_frames(baseline_pnl, baseline_fills, test_days)
            baseline_folds = fold_metrics_from_day_metrics(baseline_summary, folds)
            manifest["baseline"] = {
                "submission_path": rel(BASELINE_SUBMISSION),
                "submission_hash": file_hash(BASELINE_SUBMISSION),
                "data_dir": str(data_dir),
                "days": test_days,
                "folds": folds,
                "seed": args.seed,
                "run_config": vars(args),
                "backtest_mode": baseline_mode,
                "summary": baseline_summary,
                "folds_metrics": baseline_folds,
            }
            manifest["baseline_snapshot_hash"] = hash_text(stable_json(manifest["baseline"]), 16)
        if args.objective == "positive_pnl" and not manifest.get("no_trade_baseline"):
            manifest["no_trade_baseline"] = no_trade_metrics(test_days, folds)
        baseline_metrics = {
            "summary": manifest["baseline"]["summary"],
            "folds": manifest["baseline"]["folds_metrics"],
        }
        no_trade = manifest.get("no_trade_baseline") or no_trade_metrics(test_days, folds)
        baseline_hash = manifest["baseline_snapshot_hash"]
        update_manifest(manifest_path, manifest)

        prices = load_prices_for_proxy(data_dir, sorted({d for fold in folds for d in fold["train_days"]}))
        processed = processed_candidate_ids(artifacts_dir) if args.resume else set()
        no_improvement = 0
        if manifest.get("best_metrics"):
            best_score = float(manifest["best_metrics"].get("score", float("-inf")))
        else:
            best_score = float("-inf")

        generator = candidate_search_space(args.seed, code_hash, baseline_hash, folds, args.objective)
        cycle_index = int(manifest.get("cycle_index", 0) or 0)
        batch_lines: List[str] = []
        batch_count = 0
        generated_any = False

        for config in generator:
            candidate_id = config["candidate_id"]
            if candidate_id in processed:
                continue

            generated_any = True
            manifest["candidates_generated"] = int(manifest.get("candidates_generated", 0)) + 1
            atomic_write_json(artifacts_dir / "candidates" / f"{candidate_id}.json", config)

            try:
                proxy = proxy_screen(prices, config, folds, args)
                manifest["candidates_screened"] = int(manifest.get("candidates_screened", 0)) + 1
                batch_lines.append(
                    f"{candidate_id}: proxy ic={proxy['ic']:.5f} hit={proxy['hit_rate']:.5f} n={proxy['n']}"
                )
                if not proxy["passed"]:
                    append_rejection(
                        artifacts_dir,
                        {
                            "timestamp": utc_now(),
                            "candidate_id": candidate_id,
                            "stage": "proxy",
                            "reason": proxy["reason"],
                            "proxy_ic": proxy["ic"],
                            "proxy_hit_rate": proxy["hit_rate"],
                        },
                    )
                    update_manifest(manifest_path, manifest)
                else:
                    submission_path = write_candidate_submission(config, artifacts_dir)
                    config["generated_submission_path"] = rel(submission_path)
                    config["generated_submission_hash"] = file_hash(submission_path)
                    atomic_write_json(artifacts_dir / "candidates" / f"{candidate_id}.json", config)

                    pnl_df, fills_df, mode = run_backtest(
                        submission_path,
                        data_dir,
                        test_days,
                        artifacts_dir,
                        args.no_passive,
                        args.use_subprocess,
                    )
                    summary = metrics_from_frames(pnl_df, fills_df, test_days)
                    fold_metrics = fold_metrics_from_day_metrics(summary, folds)
                    candidate_metrics = {
                        "candidate_id": candidate_id,
                        "config": config,
                        "proxy": proxy,
                        "backtest_mode": mode,
                        "summary": summary,
                        "folds": fold_metrics,
                    }
                    leakage_passed = all(
                        int(config["feature_config"]["purge_embargo_ticks"]) >= int(config["horizon"])
                        for _fold in folds
                    )
                    if args.objective == "positive_pnl":
                        passed, reasons, fold_results = evaluate_positive_gates(
                            candidate_metrics,
                            baseline_metrics,
                            no_trade,
                            folds,
                            args,
                            leakage_passed,
                        )
                    else:
                        passed, reasons, fold_results = evaluate_gates(
                            candidate_metrics,
                            baseline_metrics,
                            folds,
                            args.turnover_cap,
                            leakage_passed,
                        )
                    candidate_metrics["gate_passed"] = passed
                    candidate_metrics["gate_reasons"] = reasons
                    candidate_metrics["fold_results"] = fold_results
                    candidate_metrics["no_trade_baseline"] = no_trade
                    atomic_write_json(artifacts_dir / "metrics" / f"{candidate_id}.json", candidate_metrics)
                    manifest["candidates_validated"] = int(manifest.get("candidates_validated", 0)) + 1

                    score = candidate_score(summary, baseline_metrics["summary"], fold_metrics, args.objective)
                    if passed:
                        manifest["candidates_passed"] = int(manifest.get("candidates_passed", 0)) + 1
                        append_leaderboard(
                            artifacts_dir,
                            {
                                "timestamp": utc_now(),
                                "candidate_id": candidate_id,
                                "score": score,
                                "pnl": summary["pnl"],
                                "baseline_pnl": baseline_metrics["summary"]["pnl"],
                                "pnl_uplift": summary["pnl"] - baseline_metrics["summary"]["pnl"],
                                "no_trade_pnl": no_trade["summary"]["pnl"],
                                "max_drawdown": summary["max_drawdown"],
                                "baseline_max_drawdown": baseline_metrics["summary"]["max_drawdown"],
                                "turnover": summary["turnover"],
                                "proxy_ic": proxy["ic"],
                                "proxy_hit_rate": proxy["hit_rate"],
                                "submission_path": rel(submission_path),
                            },
                        )
                        if score > best_score:
                            best_score = score
                            no_improvement = 0
                            manifest["current_best_candidate_id"] = candidate_id
                            manifest["best_metrics"] = {
                                "candidate_id": candidate_id,
                                "score": score,
                                "summary": summary,
                                "folds": fold_metrics,
                                "proxy": proxy,
                            }
                        else:
                            no_improvement += 1
                    else:
                        no_improvement += 1
                        append_rejection(
                            artifacts_dir,
                            {
                                "timestamp": utc_now(),
                                "candidate_id": candidate_id,
                                "stage": "gates",
                                "reason": " | ".join(reasons),
                                "pnl": summary["pnl"],
                                "baseline_pnl": baseline_metrics["summary"]["pnl"],
                                "max_drawdown": summary["max_drawdown"],
                                "turnover": summary["turnover"],
                                "proxy_ic": proxy["ic"],
                                "proxy_hit_rate": proxy["hit_rate"],
                            },
                        )
                    batch_lines.append(
                        f"{candidate_id}: validated pass={passed} pnl={summary['pnl']:.2f} "
                        f"baseline={baseline_metrics['summary']['pnl']:.2f} turnover={summary['turnover']:.5f}"
                    )
                    update_manifest(manifest_path, manifest)
            except Exception as exc:
                no_improvement += 1
                reason = f"candidate_exception: {exc}"
                append_rejection(
                    artifacts_dir,
                    {
                        "timestamp": utc_now(),
                        "candidate_id": candidate_id,
                        "stage": "exception",
                        "reason": reason,
                    },
                )
                atomic_write_text(
                    artifacts_dir / "metrics" / f"{candidate_id}.error.txt",
                    traceback.format_exc(),
                )
                batch_lines.append(f"{candidate_id}: exception {exc}")
                update_manifest(manifest_path, manifest)

            batch_count += 1
            elapsed_hours = (time.monotonic() - start_monotonic) / 3600.0
            stop_reason = ""
            if not lock.still_owned():
                stop_reason = "user_stop_or_lock_removed"
            elif elapsed_hours >= args.max_hours:
                stop_reason = "time_cap"
            elif int(manifest.get("candidates_validated", 0)) >= args.max_validated:
                stop_reason = "max_validated"
            elif int(manifest.get("candidates_validated", 0)) > 0 and no_improvement >= args.no_improvement_limit:
                stop_reason = "stale_limit"

            if batch_count >= args.batch_size or stop_reason:
                cycle_index += 1
                manifest["cycle_index"] = cycle_index
                write_cycle_log(artifacts_dir, cycle_index, args, batch_lines, manifest)
                print(
                    f"cycle={cycle_index} generated={manifest['candidates_generated']} "
                    f"screened={manifest['candidates_screened']} validated={manifest['candidates_validated']} "
                    f"passed={manifest['candidates_passed']} best={manifest.get('current_best_candidate_id')}"
                )
                batch_lines = []
                batch_count = 0
                update_manifest(manifest_path, manifest)
                if args.cycles is not None and cycle_index >= args.cycles and not stop_reason:
                    stop_reason = "cycle_limit"
                if stop_reason:
                    break

        if not stop_reason:
            if batch_lines:
                cycle_index += 1
                manifest["cycle_index"] = cycle_index
                write_cycle_log(artifacts_dir, cycle_index, args, batch_lines, manifest)
            stop_reason = "candidate_space_exhausted" if generated_any else "no_unprocessed_candidates"

        write_final_report(artifacts_dir, manifest, stop_reason)
        update_manifest(manifest_path, manifest)
        print(f"Stopped: {stop_reason}")
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
