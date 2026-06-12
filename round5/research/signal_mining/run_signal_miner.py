from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.signal_mining.clustering import build_all_clusters
from analysis.signal_mining.data import data_file_hashes, discover_data_dir, discover_days, load_prices, load_trades
from analysis.signal_mining.features import HORIZONS, build_feature_table
from analysis.signal_mining.reporting import (
    RunLock,
    atomic_write_csv,
    atomic_write_json,
    ensure_artifact_dirs,
    utc_now,
    write_cycle_log,
    write_final_report,
)
from analysis.signal_mining.scoring import (
    REFINED_PAIR_FEATURES,
    build_signal_catalog,
    score_feature_signals,
    score_interactions_and_regimes,
    score_pair_signals,
    score_residual_pair_signals,
    score_spread_and_mirror,
)


DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "signal_mining"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated Round 5 signal and cluster miner.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--relationship-top-k", type=int, default=500)
    parser.add_argument("--include-heavy-models", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--timestamp-stride", type=int, default=1)
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--force-lock", action="store_true")
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--refined", action="store_true", help="Use normalized features, residual pairs, and strict clustering.")
    parser.add_argument("--residualized-pairs", action="store_true")
    parser.add_argument("--strict-clusters", action="store_true")
    parser.add_argument("--cluster-min-score", type=float, default=0.08)
    parser.add_argument("--top-edges-per-node", type=int, default=4)
    return parser.parse_args()


def token() -> str:
    return hashlib.sha256(f"{os.getpid()}:{time.time()}:{random.random()}".encode("utf-8")).hexdigest()[:16]


def top_rows(rows: List[Dict[str, Any]], n: int = 20) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: float(r.get("stability_score", r.get("cluster_score", 0.0)) or 0.0), reverse=True)[:n]


def write_signal_specs(artifact_root: Path, catalog: List[Dict[str, Any]]) -> None:
    for row in catalog:
        atomic_write_json(artifact_root / "signals" / f"{row['signal_id']}.json", row)


def refined_feature_columns(feature_cols: List[str]) -> List[str]:
    allowed_exact = {
        "imbalance_l1",
        "imbalance_all",
        "weighted_imbalance",
        "microprice_edge_rel",
        "autocorr_proxy_20",
        "oscillation_proxy",
        "trade_intensity_20",
    }
    allowed_parts = [
        "microprice_edge_z_",
        "microprice_edge_demeaned_",
        "spread_z_",
        "depth_z_",
        "ret_mad_z_",
        "volatility_",
        "total_depth_delta_",
        "imbalance_l1_delta_",
        "weighted_imbalance_delta_",
    ]
    refined = [
        col
        for col in feature_cols
        if col in allowed_exact or any(part in col for part in allowed_parts)
    ]
    return sorted(set(refined))


def main() -> int:
    args = parse_args()
    if args.refined:
        if args.artifacts_dir == str(DEFAULT_ARTIFACT_ROOT):
            args.artifacts_dir = str(ROOT / "artifacts" / "signal_mining_refined")
        args.residualized_pairs = True
        args.strict_clusters = True
    if args.quick:
        args.timestamp_stride = max(args.timestamp_stride, 10)
        args.relationship_top_k = min(args.relationship_top_k, 50)
        args.max_products = args.max_products or 12
        args.max_features = args.max_features or 35
        args.max_hours = min(args.max_hours, 0.25)

    artifact_root = Path(args.artifacts_dir).resolve()
    ensure_artifact_dirs(artifact_root)
    lock = RunLock(artifact_root / ".run.lock", token(), force=args.force_lock)
    lock.acquire()
    start = time.monotonic()

    manifest: Dict[str, Any] = {
        "start_time": utc_now(),
        "last_update_time": utc_now(),
        "run_config": vars(args),
        "days": [2, 3, 4],
        "day_protocol": {"discovery": 2, "oos_validation": [3, 4]},
        "counters": {},
        "top_confirmed_signal_ids": [],
        "top_promising_signal_ids": [],
        "resume_instructions": "Rerun the same command. This miner recomputes deterministic signal artifacts.",
    }

    try:
        events: List[str] = []
        data_dir = discover_data_dir(args.data_dir)
        days = [d for d in [2, 3, 4] if d in discover_days(data_dir)]
        if days != [2, 3, 4]:
            raise RuntimeError(f"Expected days [2, 3, 4], found {days}")
        manifest["data_dir"] = str(data_dir)
        manifest["data_hashes"] = data_file_hashes(data_dir)
        atomic_write_json(artifact_root / "run_manifest.json", manifest)

        events.append(f"Loading prices/trades from {data_dir}")
        prices = load_prices(data_dir, days, timestamp_stride=args.timestamp_stride)
        trades = load_trades(data_dir, days)
        manifest["counters"]["price_rows"] = int(len(prices))
        manifest["counters"]["trade_rows"] = int(len(trades))
        write_cycle_log(artifact_root, 1, events, manifest)

        events = ["Building causal feature table"]
        features, feature_cols, family_map = build_feature_table(prices, trades)
        products = sorted(features["product"].unique())
        if args.max_products:
            products = products[: args.max_products]
            features = features[features["product"].isin(products)].copy()
        if args.refined:
            feature_cols = refined_feature_columns(feature_cols)
            for family, cols in list(family_map.items()):
                family_map[family] = [col for col in cols if col in feature_cols]
        manifest["products"] = products
        manifest["horizons"] = HORIZONS if not args.quick else [1, 5, 25, 50]
        manifest["feature_families"] = {k: [c for c in v if c in features.columns] for k, v in family_map.items()}
        manifest["counters"]["feature_columns"] = len(feature_cols)
        atomic_write_json(artifact_root / "run_manifest.json", manifest)
        write_cycle_log(artifact_root, 2, events + [f"features={len(feature_cols)} products={len(products)}"], manifest)

        horizons = manifest["horizons"]
        include_mi = bool(args.include_heavy_models)

        events = ["Scoring single-product feature signals"]
        feature_rows = score_feature_signals(
            features,
            feature_cols,
            family_map,
            horizons,
            include_mi=include_mi,
            max_features=args.max_features,
        )
        feature_rows = sorted(feature_rows, key=lambda r: float(r.get("stability_score", 0.0)), reverse=True)
        atomic_write_csv(artifact_root / "feature_scores.csv", feature_rows)
        manifest["counters"]["feature_scores"] = len(feature_rows)
        atomic_write_json(artifact_root / "run_manifest.json", manifest)
        write_cycle_log(artifact_root, 3, events + [f"feature_scores={len(feature_rows)}"], manifest)

        if time.monotonic() - start > args.max_hours * 3600:
            raise TimeoutError("time_cap_after_feature_scores")

        events = ["Scoring ordered product-pair lead-lag signals"]
        if args.residualized_pairs:
            pair_rows = score_residual_pair_signals(
                features,
                products,
                horizons,
                pair_features=REFINED_PAIR_FEATURES,
                max_products=args.max_products,
            )
            events.append("common-factor residualization enabled")
        else:
            pair_rows = score_pair_signals(features, products, horizons, max_products=args.max_products)
        pair_rows = sorted(pair_rows, key=lambda r: float(r.get("stability_score", 0.0)), reverse=True)
        atomic_write_csv(artifact_root / "pair_scores.csv", pair_rows)
        atomic_write_csv(artifact_root / "lead_lag_edges.csv", pair_rows)
        manifest["counters"]["pair_scores"] = len(pair_rows)
        atomic_write_json(artifact_root / "run_manifest.json", manifest)
        write_cycle_log(artifact_root, 4, events + [f"pair_scores={len(pair_rows)}"], manifest)

        events = ["Scoring spread and mirror relationships"]
        spread_rows, mirror_rows = score_spread_and_mirror(features, products, horizons, max_products=args.max_products)
        spread_rows = sorted(spread_rows, key=lambda r: float(r.get("stability_score", 0.0)), reverse=True)
        mirror_rows = sorted(mirror_rows, key=lambda r: float(r.get("stability_score", 0.0)), reverse=True)
        atomic_write_csv(artifact_root / "spread_scores.csv", spread_rows)
        atomic_write_csv(artifact_root / "mirror_scores.csv", mirror_rows)
        manifest["counters"]["spread_scores"] = len(spread_rows)
        manifest["counters"]["mirror_scores"] = len(mirror_rows)
        atomic_write_json(artifact_root / "run_manifest.json", manifest)
        write_cycle_log(artifact_root, 5, events + [f"spread_scores={len(spread_rows)} mirror_scores={len(mirror_rows)}"], manifest)

        interaction_rows, regime_rows = score_interactions_and_regimes(feature_rows)
        atomic_write_csv(artifact_root / "interaction_scores.csv", interaction_rows)
        atomic_write_csv(artifact_root / "regime_scores.csv", regime_rows)
        manifest["counters"]["interaction_scores"] = len(interaction_rows)
        manifest["counters"]["regime_scores"] = len(regime_rows)

        events = ["Building relationship graphs and clusters"]
        cluster_rows, clusters_json = build_all_clusters(
            pair_rows,
            spread_rows,
            mirror_rows,
            min_score=args.cluster_min_score if args.strict_clusters else 0.01,
            top_edges_per_node=args.top_edges_per_node if args.strict_clusters else None,
        )
        atomic_write_csv(artifact_root / "cluster_scores.csv", cluster_rows)
        atomic_write_json(artifact_root / "clusters.json", clusters_json)
        manifest["counters"]["clusters"] = len(cluster_rows)
        write_cycle_log(artifact_root, 6, events + [f"clusters={len(cluster_rows)}"], manifest)

        events = ["Building signal catalog and final report"]
        catalog = build_signal_catalog(feature_rows, pair_rows, spread_rows, mirror_rows, top_k=args.relationship_top_k)
        confirmed = [row for row in catalog if row.get("tier") == "confirmed"]
        promising = [row for row in catalog if row.get("tier") == "promising"]
        unstable = [
            row for row in feature_rows + pair_rows + spread_rows + mirror_rows
            if row.get("tier") in {"unstable", "noisy"}
        ]
        atomic_write_csv(artifact_root / "signal_catalog.csv", catalog)
        atomic_write_csv(artifact_root / "unstable_signals.csv", unstable)
        write_signal_specs(artifact_root, catalog)
        manifest["counters"]["signals"] = len(catalog)
        manifest["counters"]["confirmed_signals"] = len(confirmed)
        manifest["counters"]["promising_signals"] = len(promising)
        manifest["top_confirmed_signal_ids"] = [row["signal_id"] for row in confirmed[:20]]
        manifest["top_promising_signal_ids"] = [row["signal_id"] for row in promising[:20]]
        manifest["stop_reason"] = "completed"
        manifest["last_update_time"] = utc_now()
        atomic_write_json(artifact_root / "run_manifest.json", manifest)
        write_final_report(
            artifact_root,
            manifest,
            top_rows(feature_rows, 25),
            top_rows(pair_rows, 25),
            sorted(cluster_rows, key=lambda r: float(r.get("cluster_score", 0.0)), reverse=True)[:25],
            catalog[:25],
        )
        write_cycle_log(artifact_root, 7, events + [f"catalog={len(catalog)} confirmed={len(confirmed)} promising={len(promising)}"], manifest)

        print(
            f"completed feature_scores={len(feature_rows)} pair_scores={len(pair_rows)} "
            f"clusters={len(cluster_rows)} signals={len(catalog)} confirmed={len(confirmed)} promising={len(promising)}"
        )
        return 0
    except Exception as exc:
        manifest["stop_reason"] = f"error:{exc}"
        manifest["last_update_time"] = utc_now()
        atomic_write_json(artifact_root / "run_manifest.json", manifest)
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())

