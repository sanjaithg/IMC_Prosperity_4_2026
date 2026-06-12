"""
backtester.py — Prosperity Round 3 backtester
==============================================

Replays the provided Datasets/prices_*.csv order books against a `Trader`
class defined in a submission file, matches its orders against the book
snapshot, enforces position limits and produces a per-product PnL report.

Matching model (conservative)
-----------------------------
- A player buy (sell) order that CROSSES the best ask (bid) fills immediately,
  walking the book up (down) until it exhausts the requested quantity, the
  limit price, or hits the position cap.
- Non-crossing (passive) orders are eligible for fill on the NEXT tick using a
  "queue-lookahead": if the next tick's opposing best quote moves THROUGH our
  price, we treat the order as filled at our price. This captures most passive
  fills without being wildly optimistic.
- Position limits strictly enforced: any order whose aggregate would breach is
  truncated (NOT rejected wholesale — mirrors how a clipped request is sized).

PnL
---
Realised cash + mark-to-market (mid price). Per-product breakdown included.

Usage
-----
    python backtester.py                            # default: submission2.py on all 3 days
    python backtester.py --submission submission.py
    python backtester.py --days 0 1 2 --submission submission2.py
    python backtester.py --out results.csv
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local datamodel (Round3/datamodel.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamodel import (                                  # noqa: E402
    Listing, OrderDepth, Order, Observation, Trade, TradingState,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in
       (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)},
}

# TTE (days) at the start of each provided historical day.
# Round 4: Day 1 -> 4d; Day 2 -> 3d; Day 3 -> 2d.
DAY_TO_TTE: Dict[int, float] = {1: 4.0, 2: 3.0, 3: 2.0}


# ═══════════════════════════════════════════════════════════════════
# LOADERS
# ═══════════════════════════════════════════════════════════════════
def load_trader(path: str):
    """Import a `Trader` class from an arbitrary submission file."""
    spec = importlib.util.spec_from_file_location("submission_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.Trader()


def _detect_delim(path: str) -> str:
    with open(path, "r") as f:
        head = f.readline()
    return ";" if ";" in head else ","


def load_day(data_dir: str, day: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p_path = os.path.join(data_dir, f"prices_round_4_day_{day}.csv")
    t_path = os.path.join(data_dir, f"trades_round_4_day_{day}.csv")
    if not os.path.isfile(p_path):
        raise FileNotFoundError(p_path)
    prices = pd.read_csv(p_path, delimiter=_detect_delim(p_path))
    if os.path.isfile(t_path):
        trades = pd.read_csv(t_path, delimiter=_detect_delim(t_path))
    else:
        trades = pd.DataFrame()
    return prices, trades


# ═══════════════════════════════════════════════════════════════════
# ORDER-BOOK CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════
def build_order_depth(row) -> OrderDepth:
    """`row` may be a pd.Series, a dict, or a namedtuple from itertuples."""
    if hasattr(row, "get"):
        get = row.get
    elif isinstance(row, dict):
        get = row.get
    else:
        def get(k, default=None):
            return getattr(row, k, default)
    od = OrderDepth()
    for i in (1, 2, 3):
        bp = get(f"bid_price_{i}")
        bv = get(f"bid_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv != 0:
            od.buy_orders[int(bp)] = int(bv)
        ap = get(f"ask_price_{i}")
        av = get(f"ask_volume_{i}")
        if pd.notna(ap) and pd.notna(av) and av != 0:
            od.sell_orders[int(ap)] = -int(abs(av))
    return od


# ═══════════════════════════════════════════════════════════════════
# MATCHING
# ═══════════════════════════════════════════════════════════════════
def match_orders_immediate(product: str,
                           orders: List[Order],
                           depth: OrderDepth,
                           position: int,
                           positions: Dict[str, int],
                           cash_ref: List[float],
                           fills_log: List[dict],
                           ts: int
                           ) -> Tuple[int, List[Tuple[Order, int, int]]]:
    """
    Match crossing portions immediately; return (new_position, passive_queue).

    passive_queue holds (original_order, already_filled_qty, signed_remaining)
    for orders whose non-crossing tail should be attempted against the NEXT
    snapshot.
    """
    limit = POSITION_LIMITS.get(product, 10_000)
    pos = position
    passive: List[Tuple[Order, int, int]] = []

    for order in orders:
        qty = int(order.quantity)
        if qty == 0:
            continue

        if qty > 0:   # BUY
            # Cross against sell side (ascending price) up to limit price
            remaining = qty
            filled = 0
            for price in sorted(depth.sell_orders):
                if price > order.price or remaining <= 0:
                    break
                avail = -depth.sell_orders[price]  # stored negative
                room = limit - pos
                take = min(remaining, avail, room)
                if take <= 0:
                    break
                pos += take
                cash_ref[0] -= take * price
                remaining -= take
                filled += take
                # Mutate local copy so later orders in same tick see reduced depth
                depth.sell_orders[price] = -(avail - take)
                if depth.sell_orders[price] == 0:
                    del depth.sell_orders[price]
                fills_log.append({
                    "timestamp": ts, "product": product,
                    "side": "BUY", "price": price, "quantity": take,
                })
            if remaining > 0:
                passive.append((order, filled, remaining))
        else:         # SELL
            remaining = -qty
            filled = 0
            for price in sorted(depth.buy_orders, reverse=True):
                if price < order.price or remaining <= 0:
                    break
                avail = depth.buy_orders[price]
                room = limit + pos   # max we can sell
                take = min(remaining, avail, room)
                if take <= 0:
                    break
                pos -= take
                cash_ref[0] += take * price
                remaining -= take
                filled += take
                depth.buy_orders[price] = avail - take
                if depth.buy_orders[price] == 0:
                    del depth.buy_orders[price]
                fills_log.append({
                    "timestamp": ts, "product": product,
                    "side": "SELL", "price": price, "quantity": take,
                })
            if remaining > 0:
                passive.append((order, filled, -remaining))

    positions[product] = pos
    return pos, passive


def match_passive_against_next(product: str,
                               passive: List[Tuple[Order, int, int]],
                               next_depth: OrderDepth,
                               positions: Dict[str, int],
                               cash_ref: List[float],
                               fills_log: List[dict],
                               next_ts: int
                               ) -> None:
    """Lookahead fill: if the next snapshot's best opposing price crosses our
    passive price, consider it filled at our price."""
    if not passive or next_depth is None:
        return
    limit = POSITION_LIMITS.get(product, 10_000)
    for order, _already, signed_rem in passive:
        if signed_rem > 0:  # resting BUY at order.price
            if not next_depth.sell_orders:
                continue
            next_best_ask = min(next_depth.sell_orders)
            if next_best_ask <= order.price:
                room = limit - positions[product]
                take = min(signed_rem, room)
                if take <= 0:
                    continue
                positions[product] += take
                cash_ref[0] -= take * order.price
                fills_log.append({
                    "timestamp": next_ts, "product": product,
                    "side": "BUY_PASSIVE", "price": order.price, "quantity": take,
                })
        else:               # resting SELL
            if not next_depth.buy_orders:
                continue
            next_best_bid = max(next_depth.buy_orders)
            if next_best_bid >= order.price:
                room = limit + positions[product]
                take = min(-signed_rem, room)
                if take <= 0:
                    continue
                positions[product] -= take
                cash_ref[0] += take * order.price
                fills_log.append({
                    "timestamp": next_ts, "product": product,
                    "side": "SELL_PASSIVE", "price": order.price, "quantity": take,
                })


# ═══════════════════════════════════════════════════════════════════
# MARK-TO-MARKET
# ═══════════════════════════════════════════════════════════════════
def mid_of(depth: OrderDepth) -> Optional[float]:
    if depth.buy_orders and depth.sell_orders:
        return 0.5 * (max(depth.buy_orders) + min(depth.sell_orders))
    if depth.buy_orders:
        return float(max(depth.buy_orders))
    if depth.sell_orders:
        return float(min(depth.sell_orders))
    return None


# ═══════════════════════════════════════════════════════════════════
# BACKTEST CORE
# ═══════════════════════════════════════════════════════════════════
def run_backtest_single_day(trader,
                            prices: pd.DataFrame,
                            trades: pd.DataFrame,
                            day: int,
                            passive_fills: bool = True,
                            verbose: bool = False
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, float]]:
    """Returns (pnl_df, fills_df, final_positions, final_per_product_pnl)."""
    # Pre-group prices by timestamp → list of (product, row-as-series)
    prices_sorted = prices.sort_values(["timestamp", "product"])
    grouped: Dict[int, List[Tuple[str, pd.Series]]] = defaultdict(list)
    for row in prices_sorted.itertuples(index=False):
        grouped[row.timestamp].append((row.product, row))

    # Pre-group market trades by timestamp (coarse: bucket into same ts)
    market_trades_by_ts: Dict[int, List[Trade]] = defaultdict(list)
    if not trades.empty:
        # CSV uses `symbol` and has no `day` column; trades within each day's file
        for row in trades.itertuples(index=False):
            try:
                tr = Trade(symbol=str(row.symbol),
                           price=float(row.price),
                           quantity=int(row.quantity),
                           buyer=str(getattr(row, "buyer", "")),
                           seller=str(getattr(row, "seller", "")),
                           timestamp=int(row.timestamp))
                market_trades_by_ts[tr.timestamp].append(tr)
            except Exception:
                continue

    timestamps = sorted(grouped.keys())
    positions: Dict[str, int] = defaultdict(int)
    cash_ref = [0.0]
    realized_cash_by_product: Dict[str, float] = defaultdict(float)
    fills_log: List[dict] = []
    pnl_rows: List[dict] = []
    trader_data = ""

    # Set TTE for this day if the trader exposes the knob
    tte = DAY_TO_TTE.get(day)
    if tte is not None and hasattr(trader, "INITIAL_TTE_DAYS"):
        # Mutate the CLASS attribute so new reads pick it up
        type(trader).INITIAL_TTE_DAYS = float(tte)

    prev_own_trades: Dict[str, List[Trade]] = {}
    pending_passive: Dict[str, List[Tuple[Order, int, int]]] = {}

    t_start = time.time()
    for i, ts in enumerate(timestamps):
        snapshot = grouped[ts]

        # Build OrderDepth dict and Listings
        order_depths: Dict[str, OrderDepth] = {}
        listings: Dict[str, Listing] = {}
        for prod, row in snapshot:
            order_depths[prod] = build_order_depth(row)
            listings[prod] = Listing(prod, prod, "XIRECS")

        # Match yesterday's passive queue against today's book snapshot
        if passive_fills and pending_passive:
            for prod, queue in pending_passive.items():
                if prod in order_depths:
                    # track cash delta by product
                    cash_before = cash_ref[0]
                    pos_before = positions[prod]
                    match_passive_against_next(prod, queue, order_depths[prod],
                                               positions, cash_ref, fills_log, ts)
                    realized_cash_by_product[prod] += (cash_ref[0] - cash_before)
            pending_passive = {}

        # Build own_trades from last iteration's fills
        own_trades: Dict[str, List[Trade]] = prev_own_trades
        market_trades = {prod: [t for t in market_trades_by_ts.get(ts, [])
                                if t.symbol == prod]
                         for prod in order_depths}

        state = TradingState(
            traderData=trader_data,
            timestamp=int(ts),
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=dict(positions),
            observations=Observation({}, {}),
        )

        orders_dict, conversions, trader_data = trader.run(state)
        if not isinstance(trader_data, str):
            trader_data = ""
        if trader_data and len(trader_data) > 50_000:
            trader_data = trader_data[:50_000]

        # Match orders against CURRENT book (we mutate local copies of depths)
        new_own_trades: Dict[str, List[Trade]] = defaultdict(list)
        passive_queue: Dict[str, List[Tuple[Order, int, int]]] = {}
        for prod, order_list in (orders_dict or {}).items():
            if not order_list:
                continue
            if prod not in order_depths:
                continue
            cash_before = cash_ref[0]
            pos_before = positions[prod]
            _, passive = match_orders_immediate(
                prod, order_list, order_depths[prod], positions[prod],
                positions, cash_ref, fills_log, int(ts))
            realized_cash_by_product[prod] += (cash_ref[0] - cash_before)
            if passive:
                passive_queue[prod] = passive
            # derive synthetic Trade objects for next iteration's own_trades
            # (we approximate by scanning new fills with current ts)
        pending_passive = passive_queue if passive_fills else {}

        # Build own_trades for next tick from fills just logged at ts
        tick_fills = [f for f in reversed(fills_log) if f["timestamp"] == ts]
        for f in tick_fills:
            buyer = "SUBMISSION" if f["side"].startswith("BUY") else ""
            seller = "SUBMISSION" if f["side"].startswith("SELL") else ""
            new_own_trades[f["product"]].append(
                Trade(symbol=f["product"], price=f["price"],
                      quantity=f["quantity"], buyer=buyer,
                      seller=seller, timestamp=int(ts)))
        prev_own_trades = dict(new_own_trades)

        # Mark-to-market
        mtm = cash_ref[0]
        per_product_mtm = {}
        for prod, pos in positions.items():
            if pos == 0:
                per_product_mtm[prod] = realized_cash_by_product.get(prod, 0.0)
                continue
            d = order_depths.get(prod)
            if d is None:
                per_product_mtm[prod] = realized_cash_by_product.get(prod, 0.0)
                continue
            m = mid_of(d)
            if m is None:
                per_product_mtm[prod] = realized_cash_by_product.get(prod, 0.0)
                continue
            mtm += pos * m
            per_product_mtm[prod] = realized_cash_by_product.get(prod, 0.0) + pos * m

        row: Dict[str, float] = {
            "day": day, "timestamp": int(ts),
            "pnl": mtm,
            "cash": cash_ref[0],
            **{f"pos_{k}": v for k, v in positions.items()},
            **{f"ppnl_{k}": v for k, v in per_product_mtm.items()},
        }
        pnl_rows.append(row)

        if verbose and i % 2000 == 0 and i > 0:
            elapsed = time.time() - t_start
            print(f"  day {day} | tick {i}/{len(timestamps)}  "
                  f"pnl={mtm:>10.0f}  cash={cash_ref[0]:>10.0f}  "
                  f"({elapsed:.1f}s)")

    pnl_df = pd.DataFrame(pnl_rows)
    fills_df = pd.DataFrame(fills_log)
    final_per_product = {
        prod: (realized_cash_by_product.get(prod, 0.0)
               + positions[prod] * (
                   mid_of(order_depths[prod]) if prod in order_depths and mid_of(order_depths[prod]) is not None
                   else 0.0))
        for prod in set(list(realized_cash_by_product.keys()) + list(positions.keys()))
    }
    return pnl_df, fills_df, dict(positions), final_per_product


# ═══════════════════════════════════════════════════════════════════
# DRIVER
# ═══════════════════════════════════════════════════════════════════
def summarise(pnl_all: pd.DataFrame, fills_all: pd.DataFrame,
              per_product_all: Dict[int, Dict[str, float]]) -> None:
    print("\n" + "═" * 68)
    print("SUMMARY")
    print("═" * 68)
    if pnl_all.empty:
        print("No PnL rows produced.")
        return
    for d, grp in pnl_all.groupby("day"):
        final = grp.iloc[-1]["pnl"]
        peak = grp["pnl"].max()
        trough = grp["pnl"].min()
        ret_series = grp["pnl"].diff().dropna()
        sharpe = (ret_series.mean() / ret_series.std()) * math.sqrt(len(ret_series)) \
            if ret_series.std() > 0 else float("nan")
        n_fills = (fills_all["timestamp"].between(
            grp["timestamp"].min(), grp["timestamp"].max()).sum()
                   if not fills_all.empty else 0)
        print(f"\nDay {int(d)}:")
        print(f"  Final PnL   : {final:>12,.1f}")
        print(f"  Peak / trough: {peak:>12,.1f}  / {trough:>12,.1f}")
        print(f"  Max DD      : {(peak - grp['pnl'].cummax().iloc[-1] + grp['pnl'].iloc[-1] - peak):>12,.1f}"
              if False else f"  Max DD      : {(grp['pnl'].cummax() - grp['pnl']).max():>12,.1f}")
        print(f"  Per-tick Sharpe (approx): {sharpe:>6.3f}")
        print(f"  Fills on-day: {int(n_fills):>6d}")
        pp = per_product_all.get(int(d), {})
        if pp:
            print("  Per-product contribution (realized + MTM):")
            for k, v in sorted(pp.items(), key=lambda x: -abs(x[1])):
                if abs(v) < 1e-6:
                    continue
                print(f"     {k:<22s} {v:>12,.1f}")
    # Overall
    totals = pnl_all.groupby("day")["pnl"].last()
    print(f"\nTotal across {len(totals)} day(s): {totals.sum():>12,.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="final_submission_Round3.py")
    parser.add_argument("--data-dir", default="Dataset/ROUND_4")
    parser.add_argument("--days", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--out", default=None,
                        help="Write full PnL CSV to this path")
    parser.add_argument("--fills-out", default=None,
                        help="Write fills CSV to this path")
    parser.add_argument("--no-passive", action="store_true",
                        help="Disable 1-tick lookahead passive fills")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sub_path = os.path.abspath(args.submission)
    data_dir = os.path.abspath(args.data_dir)

    print(f"Loading trader from: {sub_path}")
    trader = load_trader(sub_path)

    all_pnl: List[pd.DataFrame] = []
    all_fills: List[pd.DataFrame] = []
    per_product_all: Dict[int, Dict[str, float]] = {}

    for day in args.days:
        # Fresh trader instance per day → mirrors the live environment where
        # each day is its own 10k-tick simulation starting from flat positions.
        trader = load_trader(sub_path)
        print(f"\n── Backtesting day {day} …")
        prices, trades = load_day(data_dir, day)
        print(f"   {len(prices):,} price rows, {len(trades):,} market trades")
        t0 = time.time()
        pnl_df, fills_df, _, per_product = run_backtest_single_day(
            trader, prices, trades, day=day,
            passive_fills=not args.no_passive,
            verbose=not args.quiet,
        )
        dt = time.time() - t0
        print(f"   done in {dt:.1f}s  — final PnL {pnl_df['pnl'].iloc[-1]:,.1f}")
        all_pnl.append(pnl_df)
        all_fills.append(fills_df)
        per_product_all[day] = per_product

    pnl_all = pd.concat(all_pnl, ignore_index=True) if all_pnl else pd.DataFrame()
    fills_all = pd.concat(all_fills, ignore_index=True) if all_fills else pd.DataFrame()

    summarise(pnl_all, fills_all, per_product_all)

    if args.out:
        pnl_all.to_csv(args.out, index=False)
        print(f"\nPnL written → {args.out}")
    if args.fills_out:
        fills_all.to_csv(args.fills_out, index=False)
        print(f"Fills written → {args.fills_out}")


if __name__ == "__main__":
    main()
