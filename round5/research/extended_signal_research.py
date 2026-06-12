"""
Extended Round 5 signal research loop.

This runner uses the refined signal-mining artifacts as a proxy layer, generates
simple sparse strategy candidates, then backtests only the highest-ranked
survivors. It writes incremental artifacts so it can be left running in the
background.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtester  # noqa: E402


ARTIFACT_ROOT = ROOT / "artifacts" / "extended_signal_research"
REFINED_ROOT = ROOT / "artifacts" / "signal_mining_refined"
SUBMISSION_PATH = ROOT / "Submissions_Derek" / "v2_secondary_signal_tests.py"
DAYS = [2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-backtests", type=int, default=160)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--top-spread-rows", type=int, default=180)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
    else:
        fieldnames = ["candidate_id"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def candidate_id(prefix: str, payload: Dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def side_from_priority(side: str) -> str:
    return "high" if side == "short_signal" else "low"


def cfg_pair(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pos_limit": int(rule.get("pos_limit", 10)),
        "clip": int(rule.get("clip", 1)),
        "cooldown_ticks": int(rule.get("cooldown_ticks", 20)),
        "max_orders_per_tick": 10,
        "pressure_rules": [],
        "lead_lag_rules": [],
        "pair_rules": [rule],
    }


def cfg_pressure(rules: List[Dict[str, Any]], pos_limit: int = 10, cooldown: int = 20) -> Dict[str, Any]:
    return {
        "pos_limit": pos_limit,
        "clip": 1,
        "cooldown_ticks": cooldown,
        "max_orders_per_tick": 10,
        "pressure_rules": rules,
        "lead_lag_rules": [],
        "pair_rules": [],
    }


def cfg_lead_lag(rule: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pos_limit": 10,
        "clip": 1,
        "cooldown_ticks": int(rule.get("cooldown_ticks", 20)),
        "max_orders_per_tick": 10,
        "pressure_rules": [],
        "lead_lag_rules": [rule],
        "pair_rules": [],
    }


def add_candidate(candidates: List[Dict[str, Any]], seen: set[str], kind: str, proxy_score: float, cfg: Dict[str, Any]) -> None:
    payload = {"kind": kind, "cfg": cfg}
    cid = candidate_id(kind.upper()[:8], payload)
    if cid in seen:
        return
    seen.add(cid)
    candidates.append(
        {
            "candidate_id": cid,
            "kind": kind,
            "proxy_score": float(proxy_score),
            "cfg": cfg,
        }
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def priority_candidates(candidates: List[Dict[str, Any]], seen: set[str]) -> None:
    for rel_path, kind, min_mean in [
        ("priority_checks/pebbles_pair_reversion.csv", "priority_pebbles", 40.0),
        ("priority_checks/snackpack_constraint_reversion.csv", "priority_snackpack", 8.0),
    ]:
        df = load_csv(REFINED_ROOT / rel_path)
        if df.empty:
            continue
        df = df[(df["n"] >= 100) & (df["mean_signed_move"] > min_mean)].copy()
        df["proxy"] = df["mean_signed_move"] * df["hit_rate"] * (df["n"].clip(upper=1000) / 1000.0)
        df = df.sort_values("proxy", ascending=False).head(80)
        for row in df.to_dict("records"):
            rule = {
                "name": f"{row['pair_a']}__{row['pair_b']}__{row['basis']}__{row['window']}__{row['threshold']}__{row['side']}",
                "a": row["pair_a"],
                "b": row["pair_b"],
                "basis": row["basis"],
                "window": int(row["window"]),
                "entry_z": float(row["threshold"]),
                "exit_z": 0.0,
                "target": 10,
                "side": side_from_priority(str(row["side"])),
            }
            add_candidate(candidates, seen, kind, float(row["proxy"]), cfg_pair(rule))


def spread_artifact_candidates(candidates: List[Dict[str, Any]], seen: set[str], top_n: int) -> None:
    df = load_csv(REFINED_ROOT / "spread_scores.csv")
    if df.empty:
        return
    df = df[df["min_n"] >= 9000].copy()
    df = df.sort_values("stability_score", ascending=False).head(top_n)
    for row in df.to_dict("records"):
        feature = str(row["feature"])
        window = 100
        for token in feature.split("_"):
            if token.isdigit():
                window = int(token)
        score_by_day = [float(row.get(f"score_day{day}", 0.0) or 0.0) for day in DAYS]
        majority_sign = 1 if sum(1 for score in score_by_day if score > 0) >= 2 else -1
        # positive score on negative_spread_z means low spread tends to be long_signal;
        # negative majority means high-spread reversion.
        sides = ["low"] if majority_sign > 0 else ["high"]
        if not bool(row.get("sign_stable", False)):
            sides.append("both")
        for side in sides:
            for entry in [2.0, 2.5, 3.0]:
                rule = {
                    "name": f"{row['product_a']}__{row['product_b']}__artifact__{window}__{entry}__{side}",
                    "a": row["product_a"],
                    "b": row["product_b"],
                    "basis": "spread",
                    "window": window,
                    "entry_z": entry,
                    "exit_z": 0.0,
                    "target": 10,
                    "side": side,
                }
                proxy = float(row["stability_score"]) * (1.0 if side != "both" else 0.75)
                add_candidate(candidates, seen, "artifact_spread", proxy, cfg_pair(rule))


def cluster_candidates(candidates: List[Dict[str, Any]], seen: set[str]) -> None:
    df = load_csv(REFINED_ROOT / "cluster_scores.csv")
    if df.empty:
        return
    for row in df.sort_values("cluster_score", ascending=False).head(12).to_dict("records"):
        products = str(row["products"]).split("|")
        for a, b in zip(products, products[1:]):
            for side in ["high", "low"]:
                rule = {
                    "name": f"cluster_{row['cluster_id']}__{a}__{b}__{side}",
                    "a": a,
                    "b": b,
                    "basis": "spread",
                    "window": 100,
                    "entry_z": 2.5,
                    "exit_z": 0.0,
                    "target": 10,
                    "side": side,
                }
                add_candidate(candidates, seen, "cluster_neighbor", float(row["cluster_score"]), cfg_pair(rule))


def pressure_candidates(candidates: List[Dict[str, Any]], seen: set[str]) -> None:
    df = load_csv(REFINED_ROOT / "feature_scores.csv")
    if df.empty:
        return
    df = df[(df["tier"] == "confirmed") & (df["family"].isin(["volume_depth", "microprice"]))].copy()
    df = df.sort_values("stability_score", ascending=False).head(40)
    for row in df.to_dict("records"):
        for direction in ["follow", "invert"]:
            rule = {
                "product": row["product"],
                "feature": row["feature"],
                "threshold": 0.20,
                "direction": direction,
                "target": 5,
            }
            proxy = float(row["stability_score"]) * (1.0 if direction == "follow" else 0.6)
            add_candidate(candidates, seen, "feature_pressure", proxy, cfg_pressure([rule], pos_limit=5, cooldown=30))


def lead_lag_candidates(candidates: List[Dict[str, Any]], seen: set[str]) -> None:
    df = load_csv(REFINED_ROOT / "pair_scores.csv")
    if df.empty:
        return
    df = df[(df["tier"] == "confirmed") & (df["family"].astype(str).str.contains("lead_lag", na=False))].copy()
    df = df.sort_values("stability_score", ascending=False).head(60)
    for row in df.to_dict("records"):
        for direction in ["follow", "invert"]:
            rule = {
                "leader": row["leader"],
                "follower": row["follower"],
                "feature": row["feature"],
                "threshold": 0.20,
                "direction": direction,
                "target": 5,
                "cooldown_ticks": 30,
            }
            proxy = float(row["stability_score"]) * (1.0 if direction == "follow" else 0.5)
            add_candidate(candidates, seen, "residual_lead_lag", proxy, cfg_lead_lag(rule))


def generate_candidates(top_spread_rows: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    seen: set[str] = set()
    priority_candidates(candidates, seen)
    spread_artifact_candidates(candidates, seen, top_spread_rows)
    cluster_candidates(candidates, seen)
    pressure_candidates(candidates, seen)
    lead_lag_candidates(candidates, seen)
    candidates.sort(key=lambda row: row["proxy_score"], reverse=True)
    return candidates


def backtest_candidate(candidate: Dict[str, Any], loaded_days: Dict[int, Any]) -> Dict[str, Any]:
    day_pnls: List[float] = []
    fills_total = 0
    max_dd = 0.0
    product_pnl: Dict[str, float] = {}
    day_rows: List[Dict[str, Any]] = []

    for day in DAYS:
        prices, trades, round_id = loaded_days[day]
        trader = backtester.load_trader(str(SUBMISSION_PATH))
        trader.cfg.update(candidate["cfg"])
        trader.cfg["candidate_id"] = candidate["candidate_id"]
        with contextlib.redirect_stdout(io.StringIO()):
            pnl_df, fills_df, _positions, per_product = backtester.run_backtest_single_day(
                trader,
                prices,
                trades,
                day=day,
                round_id=round_id,
                passive_fills=True,
                verbose=False,
            )
        pnl = float(pnl_df["pnl"].iloc[-1]) if not pnl_df.empty else 0.0
        drawdown = float((pnl_df["pnl"].cummax() - pnl_df["pnl"]).max()) if not pnl_df.empty else 0.0
        fills = int(fills_df["quantity"].abs().sum()) if not fills_df.empty else 0
        day_pnls.append(pnl)
        fills_total += fills
        max_dd = max(max_dd, drawdown)
        for product, value in per_product.items():
            product_pnl[product] = product_pnl.get(product, 0.0) + float(value)
        day_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "kind": candidate["kind"],
                "day": day,
                "pnl": round(pnl, 2),
                "fills_qty": fills,
                "max_drawdown": round(drawdown, 2),
            }
        )

    summary = {
        "candidate_id": candidate["candidate_id"],
        "kind": candidate["kind"],
        "proxy_score": round(float(candidate["proxy_score"]), 6),
        "day": "ALL",
        "pnl": round(sum(day_pnls), 2),
        "day2_pnl": round(day_pnls[0], 2),
        "day3_pnl": round(day_pnls[1], 2),
        "day4_pnl": round(day_pnls[2], 2),
        "fills_qty": fills_total,
        "positive_days": sum(1 for value in day_pnls if value > 0),
        "max_drawdown": round(max_dd, 2),
        "cfg_json": json.dumps(candidate["cfg"], sort_keys=True),
    }
    product_rows = [
        {"candidate_id": candidate["candidate_id"], "product": product, "pnl": round(value, 2)}
        for product, value in sorted(product_pnl.items(), key=lambda item: item[1])
    ]
    return {"summary": summary, "days": day_rows, "products": product_rows}


def write_markdown_report(path: Path, leaderboard: List[Dict[str, Any]], started_at: float, done: bool) -> None:
    elapsed = time.time() - started_at
    lines = [
        "# Extended Signal Research",
        "",
        f"- status: {'completed' if done else 'running'}",
        f"- elapsed_seconds: {elapsed:.1f}",
        f"- candidates_tested: {len(leaderboard)}",
        "",
        "## Top Backtests",
        "",
        "| Candidate | Kind | PnL | Day 2 | Day 3 | Day 4 | Fills | Pos Days | Max DD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(leaderboard, key=lambda r: float(r["pnl"]), reverse=True)[:25]:
        lines.append(
            f"| `{row['candidate_id']}` | {row['kind']} | {row['pnl']} | {row['day2_pnl']} | "
            f"{row['day3_pnl']} | {row['day4_pnl']} | {row['fills_qty']} | {row['positive_days']} | {row['max_drawdown']} |"
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = ARTIFACT_ROOT / ".run.lock"
    if lock_path.exists() and not args.force:
        print(f"lock exists: {lock_path}; pass --force to override")
        return 2
    lock_path.write_text(str(time.time()), encoding="utf-8")
    started_at = time.time()
    try:
        candidates = generate_candidates(args.top_spread_rows)
        atomic_write_json(
            ARTIFACT_ROOT / "candidate_manifest.json",
            {
                "generated_at": started_at,
                "candidate_count": len(candidates),
                "max_backtests": args.max_backtests,
                "max_hours": args.max_hours,
                "candidates": candidates,
            },
        )

        data_dir = backtester.discover_data_dir()
        if data_dir is None:
            raise RuntimeError("No data dir discovered")
        loaded_days = {day: backtester.load_day(data_dir, day) for day in DAYS}
        atomic_write_json(ARTIFACT_ROOT / "run_manifest.json", {"data_dir": data_dir, "days": DAYS, "started_at": started_at})

        leaderboard: List[Dict[str, Any]] = []
        day_rows: List[Dict[str, Any]] = []
        product_rows: List[Dict[str, Any]] = []

        for index, candidate in enumerate(candidates[: args.max_backtests], start=1):
            if time.time() - started_at > args.max_hours * 3600:
                break
            result = backtest_candidate(candidate, loaded_days)
            leaderboard.append(result["summary"])
            day_rows.extend(result["days"])
            product_rows.extend(result["products"])
            leaderboard.sort(key=lambda row: float(row["pnl"]), reverse=True)

            atomic_write_csv(ARTIFACT_ROOT / "leaderboard.csv", leaderboard)
            atomic_write_csv(ARTIFACT_ROOT / "day_rows.csv", day_rows)
            atomic_write_csv(ARTIFACT_ROOT / "product_pnl.csv", product_rows)
            write_markdown_report(ARTIFACT_ROOT / "summary.md", leaderboard, started_at, done=False)
            if index % args.progress_every == 0:
                best = leaderboard[0]
                print(
                    f"tested={index}/{min(len(candidates), args.max_backtests)} "
                    f"best={best['candidate_id']} pnl={best['pnl']} pos_days={best['positive_days']}",
                    flush=True,
                )

        write_markdown_report(ARTIFACT_ROOT / "summary.md", leaderboard, started_at, done=True)
        return 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

