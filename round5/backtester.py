"""
backtester.py — Prosperity backtester (round-aware)
===================================================================

Replays the provided prices/trades CSV order books against a `Trader`
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
    python backtester.py                            # default: submission2.py on detected days
    python backtester.py --submission submission.py
    python backtester.py --days 1 2 3 --submission submission2.py
    python backtester.py --out results.csv
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local datamodel mirrored from the IMC platform.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamodel import (                                  # noqa: E402
    Listing, OrderDepth, Order, Observation, Trade, TradingState,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
ROUND5_PRODUCTS = [
    "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES", "SLEEP_POD_SUEDE",
    "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER", "SLEEP_POD_NYLON",
    "SLEEP_POD_COTTON", "MICROCHIP_CIRCLE", "MICROCHIP_OVAL",
    "MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES", "ROBOT_LAUNDRY",
    "ROBOT_IRONING", "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
    "UV_VISOR_RED", "UV_VISOR_MAGENTA", "TRANSLATOR_SPACE_GRAY",
    "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
    "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_VOID_BLUE", "PANEL_1X2",
    "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
]

POSITION_LIMITS: Dict[str, int] = {product: 10 for product in ROUND5_PRODUCTS}


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


def _extract_round_from_path(path: str) -> Optional[int]:
    m = re.search(r"round_(\d+)_day", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1))


def _find_existing_path(data_dir: str, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        path = os.path.join(data_dir, pat)
        if os.path.isfile(path):
            return path
    return None


def discover_available_days(data_dir: str) -> List[int]:
    days = set()
    for path in glob.glob(os.path.join(data_dir, "prices_round_*_day_*.csv")):
        m = re.search(r"day_(-?\d+)\.csv$", os.path.basename(path))
        if m:
            days.add(int(m.group(1)))
    return sorted(days)


def discover_data_dir() -> Optional[str]:
    candidates = [
        "Dataset/ROUND_5",
        "Round_5/Dataset",
        "Datasets/ROUND_5",
        "Dataset",
        "Datasets",
        "Data",
        "data",
        ".",
    ]
    for c in candidates:
        if os.path.isdir(c) and discover_available_days(c):
            return c
    return None


def load_day(data_dir: str, day: int) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int]]:
    price_matches = sorted(glob.glob(os.path.join(data_dir, f"prices_round_*_day_{day}.csv")))
    p_path = price_matches[-1] if price_matches else None
    if p_path is None:
        raise FileNotFoundError(
            f"No prices file found for day {day} under {data_dir} "
            f"(expected prices_round_*_day_{day}.csv)"
        )
    round_id = _extract_round_from_path(p_path)
    prices = pd.read_csv(p_path, delimiter=_detect_delim(p_path))

    t_path = None
    if round_id is not None:
        cand = os.path.join(data_dir, f"trades_round_{round_id}_day_{day}.csv")
        if os.path.isfile(cand):
            t_path = cand
    if t_path is None:
        trade_matches = sorted(glob.glob(os.path.join(data_dir, f"trades_round_*_day_{day}.csv")))
        t_path = trade_matches[-1] if trade_matches else None

    if t_path and os.path.isfile(t_path):
        trades = pd.read_csv(t_path, delimiter=_detect_delim(t_path))
    else:
        trades = pd.DataFrame()
    return prices, trades, round_id


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
                           ts: int,
                           day: Optional[int] = None
                           ) -> Tuple[int, List[Tuple[Order, int, int]]]:
    """
    Match crossing portions immediately; return (new_position, passive_queue).

    passive_queue holds (original_order, already_filled_qty, signed_remaining)
    for orders whose non-crossing tail should be attempted against the NEXT
    snapshot.
    """
    limit = POSITION_LIMITS.get(product, 10)
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
                    "day": day,
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
                    "day": day,
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
                               next_ts: int,
                               day: Optional[int] = None
                               ) -> None:
    """Lookahead fill: if the next snapshot's best opposing price crosses our
    passive price, consider it filled at our price."""
    if not passive or next_depth is None:
        return
    limit = POSITION_LIMITS.get(product, 10)
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
                    "day": day,
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
                    "day": day,
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
                            round_id: Optional[int] = None,
                            passive_fills: bool = True,
                            verbose: bool = False
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, float]]:
    """Returns (pnl_df, fills_df, final_positions, final_per_product_pnl)."""
    def _clean_party(val) -> str:
        if val is None:
            return ""
        try:
            if pd.isna(val):
                return ""
        except Exception:
            pass
        s = str(val).strip()
        return "" if s.lower() == "nan" else s

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
                symbol = getattr(row, "symbol", None)
                if symbol is None:
                    symbol = getattr(row, "product", None)
                if symbol is None:
                    continue
                buyer = _clean_party(getattr(row, "buyer", ""))
                seller = _clean_party(getattr(row, "seller", ""))
                tr = Trade(symbol=str(symbol),
                           price=float(row.price),
                           quantity=int(row.quantity),
                           buyer=buyer,
                           seller=seller,
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
                                               positions, cash_ref, fills_log, ts, day=day)
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
                positions, cash_ref, fills_log, int(ts), day=day)
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
    if not fills_df.empty and "day" not in fills_df.columns:
        fills_df["day"] = day
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
        if not fills_all.empty:
            if "day" in fills_all.columns:
                n_fills = int((fills_all["day"] == int(d)).sum())
            else:
                n_fills = int(fills_all["timestamp"].between(
                    grp["timestamp"].min(), grp["timestamp"].max()).sum())
        else:
            n_fills = 0
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
    parser.add_argument("--submission", default="submission2.py")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--days", nargs="+", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="Write full PnL CSV to this path")
    parser.add_argument("--fills-out", default=None,
                        help="Write fills CSV to this path")
    parser.add_argument("--no-passive", action="store_true",
                        help="Disable 1-tick lookahead passive fills")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sub_path = os.path.abspath(args.submission)
    if args.data_dir:
        data_dir = os.path.abspath(args.data_dir)
    else:
        detected = discover_data_dir()
        if not detected:
            raise FileNotFoundError(
                "Could not auto-detect data dir. Pass --data-dir pointing to folder "
                "containing prices_round_*_day_*.csv."
            )
        data_dir = os.path.abspath(detected)

    print(f"Loading trader from: {sub_path}")
    trader = load_trader(sub_path)

    all_pnl: List[pd.DataFrame] = []
    all_fills: List[pd.DataFrame] = []
    per_product_all: Dict[int, Dict[str, float]] = {}

    days = args.days if args.days else discover_available_days(data_dir)
    if not days:
        raise FileNotFoundError(f"No prices_round_*_day_*.csv files found in {data_dir}")

    for day in days:
        # Fresh trader instance per day → mirrors the live environment where
        # each day is its own 10k-tick simulation starting from flat positions.
        trader = load_trader(sub_path)
        print(f"\n── Backtesting day {day} …")
        prices, trades, round_id = load_day(data_dir, day)
        print(f"   {len(prices):,} price rows, {len(trades):,} market trades")
        t0 = time.time()
        pnl_df, fills_df, _, per_product = run_backtest_single_day(
            trader, prices, trades, day=day, round_id=round_id,
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
