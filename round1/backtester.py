"""
IMC Prosperity 4 — Local Backtester
=====================================
Simulates the IMC engine locally: feeds your Trader class market data tick by
tick, matches your orders against the order book, tracks position & PnL.

Usage:
    python backtester.py TUTORIAL_ROUND_1/trader_sma.py --data TUTORIAL_ROUND_1/Data
    python backtester.py TUTORIAL_ROUND_1/trader_l3fair.py --data TUTORIAL_ROUND_1/Data --day -1
    python backtester.py TUTORIAL_ROUND_1/trader_sma.py --data TUTORIAL_ROUND_1/Data --verbose

Matching logic (mirrors IMC engine):
    - Your buy orders fill against the book's sell_orders (asks)
    - Your sell orders fill against the book's buy_orders (bids)
    - You get filled at the BOOK price (not your limit price) — price improvement
    - Position limits enforced: partial fills if you'd exceed the limit
    - Orders do NOT persist across ticks (no resting orders in the book)
    - The book resets from historical data each tick (no market impact)
"""

import os
import sys
import re
import json
import glob
import importlib.util
import argparse
from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Add parent dir to path so datamodel can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datamodel import OrderDepth, TradingState, Order, Trade, Listing


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_data(data_dir):
    pframes, tframes = [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "prices_*.csv"))):
        pframes.append(pd.read_csv(f, delimiter=";"))
    for f in sorted(glob.glob(os.path.join(data_dir, "trades_*.csv"))):
        df = pd.read_csv(f, delimiter=";")
        m = re.search(r"day_([-\d]+)", os.path.basename(f))
        if m:
            df["day"] = int(m.group(1))
        tframes.append(df)

    prices = pd.concat(pframes, ignore_index=True) if pframes else pd.DataFrame()
    trades = pd.concat(tframes, ignore_index=True) if tframes else pd.DataFrame()
    return prices, trades


def load_trader(trader_path):
    """Dynamically load a Trader class from a .py file."""
    spec = importlib.util.spec_from_file_location("trader_module", trader_path)
    module = importlib.util.module_from_spec(spec)

    # Make datamodel available to the trader module
    import datamodel
    sys.modules["datamodel"] = datamodel

    spec.loader.exec_module(module)
    return module.Trader()


# ═══════════════════════════════════════════════════════════════════
# ORDER MATCHING ENGINE
# ═══════════════════════════════════════════════════════════════════

POSITION_LIMITS = {
    "EMERALDS": 80,
    "TOMATOES": 80,
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}


def build_order_depth(row) -> OrderDepth:
    """Build an OrderDepth from a prices CSV row."""
    od = OrderDepth()
    for lvl in [1, 2, 3]:
        bp = row.get(f"bid_price_{lvl}")
        bv = row.get(f"bid_volume_{lvl}")
        ap = row.get(f"ask_price_{lvl}")
        av = row.get(f"ask_volume_{lvl}")
        if pd.notna(bp) and pd.notna(bv):
            od.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av):
            od.sell_orders[int(ap)] = int(-abs(av))  # sell_orders are negative
    return od


def match_orders(orders: List[Order], order_depth: OrderDepth,
                 market_trades: List[Trade],
                 position: int, product: str) -> Tuple[List[Trade], int]:
    """
    Match trader's orders against the order book AND market trades.
    Returns list of fills and new position.

    Rules:
    1. Aggressive: Buy orders fill against resting asks if buy_price >= ask_price
    2. Aggressive: Sell orders fill against resting bids if sell_price <= bid_price
    3. Passive: Buy orders fill if a market trade occurs at price P <= buy_price
    4. Passive: Sell orders fill if a market trade occurs at price P >= sell_price
    5. Fill at the BOOK price for aggressive, or at the TRADE price for passive.
    6. Position limits enforced.
    """
    fills: List[Trade] = []
    limit = POSITION_LIMITS.get(product, 50)
    
    # Track available volume in the book for this tick
    resting_asks = dict(order_depth.sell_orders)
    resting_bids = dict(order_depth.buy_orders)
    
    # Track available "passive" volume from market trades for this tick
    # (We assume we can only be part of a market trade up to its volume)
    available_trade_vol = {} 
    for mt in market_trades:
        available_trade_vol[mt.price] = available_trade_vol.get(mt.price, 0) + mt.quantity

    if orders:
        has_large = any(abs(o.quantity) > 50 for o in orders)
        if has_large:
            print(f"  [Engine] {product} Orders: {[str(o) for o in orders]}")
            
    for order in orders:
        if order.quantity > 0:
            # --- 1. Aggressive Buying (cross the spread) ---
            for ask_price in sorted(resting_asks.keys()):
                if order.price >= ask_price:
                    print(f"  [Engine] {product} AGGRESSIVE BUY FILL at {ask_price}")
                    available = -resting_asks[ask_price]
                    if available <= 0: continue
                    
                    max_buy = limit - position
                    if max_buy <= 0: break
                    
                    fill_qty = min(order.quantity, available, max_buy)
                    if fill_qty > 0:
                        fills.append(Trade(product, ask_price, fill_qty, "SUBMISSION", "", 0))
                        position += fill_qty
                        resting_asks[ask_price] += fill_qty
                        order.quantity -= fill_qty
                if order.quantity <= 0 or limit - position <= 0: break

            # --- 2. Passive Buying (get hit by market trades) ---
            if order.quantity > 0 and (limit - position > 0):
                for t_price in sorted(available_trade_vol.keys()):
                    if t_price <= order.price:
                        available = available_trade_vol[t_price]
                        if available <= 0: continue
                        
                        max_buy = limit - position
                        if max_buy <= 0: break
                        
                        fill_qty = min(order.quantity, available, max_buy)
                        if fill_qty > 0:
                            fills.append(Trade(product, t_price, fill_qty, "SUBMISSION", "MARKET", 0))
                            position += fill_qty
                            available_trade_vol[t_price] -= fill_qty
                            order.quantity -= fill_qty
                    if order.quantity <= 0 or limit - position <= 0: break

        elif order.quantity < 0:
            sell_qty = abs(order.quantity)
            # --- 1. Aggressive Selling (cross the spread) ---
            for bid_price in sorted(resting_bids.keys(), reverse=True):
                if order.price <= bid_price:
                    available = resting_bids[bid_price]
                    if available <= 0: continue
                    
                    max_sell = limit + position
                    if max_sell <= 0: break
                    
                    fill_qty = min(sell_qty, available, max_sell)
                    if fill_qty > 0:
                        fills.append(Trade(product, bid_price, fill_qty, "", "SUBMISSION", 0))
                        position -= fill_qty
                        resting_bids[bid_price] -= fill_qty
                        sell_qty -= fill_qty
                if sell_qty <= 0 or limit + position <= 0: break

            # --- 2. Passive Selling (get hit by market trades) ---
            if sell_qty > 0 and (limit + position > 0):
                for t_price in sorted(available_trade_vol.keys(), reverse=True):
                    if t_price >= order.price:
                        available = available_trade_vol[t_price]
                        if available <= 0: continue
                        
                        max_sell = limit + position
                        if max_sell <= 0: break
                        
                        fill_qty = min(sell_qty, available, max_sell)
                        if fill_qty > 0:
                            fills.append(Trade(product, t_price, fill_qty, "MARKET", "SUBMISSION", 0))
                            position -= fill_qty
                            available_trade_vol[t_price] -= fill_qty
                            sell_qty -= fill_qty
                    if sell_qty <= 0 or limit + position <= 0: break

    return fills, position


# ═══════════════════════════════════════════════════════════════════
# PNL CALCULATION
# ═══════════════════════════════════════════════════════════════════

def calculate_pnl(cash: Dict[str, float], positions: Dict[str, int],
                  mid_prices: Dict[str, float]) -> Dict[str, float]:
    """Mark-to-market PnL per product."""
    pnl = {}
    for product in set(list(cash.keys()) + list(positions.keys())):
        c = cash.get(product, 0)
        pos = positions.get(product, 0)
        mid = mid_prices.get(product, 0)
        pnl[product] = c + pos * mid
    return pnl


# ═══════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════

def run_backtest(trader, prices, trades, day, verbose=False):
    """
    Run a full backtest for one day.
    Returns a DataFrame of per-tick results.
    """
    day_prices = prices[prices["day"] == day].sort_values("timestamp")
    timestamps = sorted(day_prices["timestamp"].unique())

    # Filter trades for this day
    day_trades = pd.DataFrame()
    if not trades.empty and "day" in trades.columns:
        day_trades = trades[trades["day"] == day]

    positions: Dict[str, int] = {}
    cash: Dict[str, float] = {}
    trader_data = ""
    own_trades_history: Dict[str, List[Trade]] = {}

    results = []
    log_activities = []
    all_trades = []
    
    mid_prices: Dict[str, float] = {} # Persistent across ticks for PnL consistency

    # Pre-group trades by timestamp for performance
    trades_by_ts = {}
    if not day_trades.empty:
        for ts, group in day_trades.groupby("timestamp"):
            trades_by_ts[ts] = group.to_dict("records")

    # Pre-group prices by timestamp for performance
    prices_by_ts = {}
    for ts, group in day_prices.groupby("timestamp"):
        prices_by_ts[ts] = group.to_dict("records")

    for tick_idx, ts in enumerate(timestamps):
        tick_rows = prices_by_ts.get(ts, [])
        if not tick_rows:
            continue

        # Build order depths for each product
        order_depths: Dict[str, OrderDepth] = {}

        for row in tick_rows:
            product = row["product"]
            order_depths[product] = build_order_depth(row)
            mid_prices[product] = row["mid_price"]

        market_trades: Dict[str, List[Trade]] = {}
        if ts in trades_by_ts:
            for tr in trades_by_ts[ts]:
                sym = tr["product"] if "product" in tr else tr.get("symbol", "")
                if sym not in market_trades:
                    market_trades[sym] = []
                market_trades[sym].append(Trade(
                    symbol=sym, price=tr["price"],
                    quantity=int(tr["quantity"]),
                    buyer=str(tr.get("buyer", "")),
                    seller=str(tr.get("seller", "")),
                    timestamp=ts))

        # Build TradingState
        state = TradingState(
            traderData=trader_data,
            timestamp=ts,
            listings={p: Listing(p, p, "XIRECS") for p in order_depths},
            order_depths=order_depths,
            own_trades=own_trades_history,
            market_trades=market_trades,
            position=dict(positions),
            observations=None,
        )

        # Run trader
        try:
            trader_result, conversions, trader_data = trader.run(state)
        except Exception as e:
            print(f"  ERROR at tick {ts}: {e}")
            trader_result = {}
            trader_data = trader_data or ""

        # Match orders and update state
        own_trades_history = {}
        tick_fills = []

        for product, orders in trader_result.items():
            if not orders:
                continue

            # Get the order depth and market trades for matching
            od = order_depths.get(product)
            ticker_trades = market_trades.get(product, [])
            if not od: continue

            fills, new_pos = match_orders(orders, od, ticker_trades, positions.get(product, 0), product)

            if fills:
                own_trades_history[product] = fills
                positions[product] = new_pos

                for fill in fills:
                    fill.timestamp = ts
                    if fill.buyer == "SUBMISSION":
                        cash[product] = cash.get(product, 0) - fill.price * fill.quantity
                    else:
                        cash[product] = cash.get(product, 0) + fill.price * fill.quantity
                    tick_fills.append(fill)
                    all_trades.append(fill)

        # Calculate PnL
        pnl = calculate_pnl(cash, positions, mid_prices)
        total_pnl = sum(pnl.values())

        # Log for DataFrame
        for product in order_depths:
            results.append({
                "timestamp": ts,
                "tick": tick_idx,
                "product": product,
                "mid_price": mid_prices.get(product, 0),
                "position": positions.get(product, 0),
                "cash": cash.get(product, 0),
                "pnl": pnl.get(product, 0),
                "total_pnl": total_pnl,
                "n_fills": len([f for f in tick_fills if f.symbol == product]),
                "n_orders": len(trader_result.get(product, [])),
            })
            
            # Log for visualizer.py
            od = order_depths[product]
            log_activities.append({
                "timestamp": ts,
                "product": product,
                "bid_price_1": list(od.buy_orders.keys())[0] if od.buy_orders else 0,
                "bid_volume_1": list(od.buy_orders.values())[0] if od.buy_orders else 0,
                "bid_price_2": list(od.buy_orders.keys())[1] if len(od.buy_orders) > 1 else 0,
                "bid_volume_2": list(od.buy_orders.values())[1] if len(od.buy_orders) > 1 else 0,
                "bid_price_3": list(od.buy_orders.keys())[2] if len(od.buy_orders) > 2 else 0,
                "bid_volume_3": list(od.buy_orders.values())[2] if len(od.buy_orders) > 2 else 0,
                "ask_price_1": list(od.sell_orders.keys())[0] if od.sell_orders else 0,
                "ask_volume_1": list(od.sell_orders.values())[0] if od.sell_orders else 0,
                "ask_price_2": list(od.sell_orders.keys())[1] if len(od.sell_orders) > 1 else 0,
                "ask_volume_2": list(od.sell_orders.values())[1] if len(od.sell_orders) > 1 else 0,
                "ask_price_3": list(od.sell_orders.keys())[2] if len(od.sell_orders) > 2 else 0,
                "ask_volume_3": list(od.sell_orders.values())[2] if len(od.sell_orders) > 2 else 0,
                "mid_price": mid_prices.get(product, 0),
                "pnl": pnl.get(product, 0),
                "total_pnl": total_pnl
            })

    return pd.DataFrame(results), log_activities, all_trades


# ═══════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════

def print_summary(results, day):
    """Print a clean summary of backtest results."""
    if results.empty:
        print("  No results.")
        return

    last_tick = results[results["timestamp"] == results["timestamp"].max()]

    print(f"\n{'='*60}")
    print(f"  DAY {day} RESULTS")
    print(f"{'='*60}")

    total_pnl = 0
    for _, row in last_tick.iterrows():
        p = row["product"]
        pnl = row["pnl"]
        pos = row["position"]
        total_pnl += pnl
        pos_str = f"LONG {pos}" if pos > 0 else (f"SHORT {abs(pos)}" if pos < 0 else "FLAT")
        pnl_color = "\033[92m" if pnl >= 0 else "\033[91m"
        print(f"  {p:>12}:  PnL = {pnl_color}{pnl:>+10.1f}\033[0m  |  Position = {pos_str}")

    total_fills = results["n_fills"].sum()
    total_orders = results["n_orders"].sum()

    pnl_color = "\033[92m" if total_pnl >= 0 else "\033[91m"
    print(f"  {'':>12}   {'─'*40}")
    print(f"  {'TOTAL':>12}:  PnL = {pnl_color}{total_pnl:>+10.1f}\033[0m  |  Fills = {total_fills:.0f}  |  Orders = {total_orders:.0f}")
    print()


def save_results(results, trader_name, day, output_dir="."):
    """Save results as CSV for visualization."""
    fname = os.path.join(output_dir, f"backtest_{trader_name}_day{day}.csv")
    results.to_csv(fname, index=False)
    print(f"  Results saved: {fname}")
    return fname


def save_log(activities, trades, trader_name, day, output_dir="."):
    """Save results as .log and .json for visualizer.py."""
    log_name = f"backtest_{trader_name}_day{day}.log"
    json_name = f"backtest_{trader_name}_day{day}.json"
    
    # Building activitiesLog string
    log_header = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss"
    log_rows = [log_header]
    for act in activities:
        row = f"{day};{act['timestamp']};{act['product']};" \
              f"{act['bid_price_1']};{act['bid_volume_1']};{act['bid_price_2']};{act['bid_volume_2']};{act['bid_price_3']};{act['bid_volume_3']};" \
              f"{act['ask_price_1']};{act['ask_volume_1']};{act['ask_price_2']};{act['ask_volume_2']};{act['ask_price_3']};{act['ask_volume_3']};" \
              f"{act['mid_price']};{act['pnl']}"
        log_rows.append(row)
    
    # Full log data
    log_data = {
        "submissionId": f"backtest_{trader_name}",
        "activitiesLog": "\n".join(log_rows),
        "tradeHistory": [
            {
                "symbol": t.symbol,
                "price": int(t.price),
                "quantity": int(t.quantity),
                "buyer": t.buyer,
                "seller": t.seller,
                "timestamp": int(t.timestamp)
            } for t in trades
        ]
    }
    
    with open(log_name, "w") as f:
        json.dump(log_data, f)
        
    # Companion JSON for profit/positions
    total_pnl = activities[-1]["total_pnl"] if activities else 0
    companion = {
        "profit": total_pnl,
        "positions": {} # Could extract if needed
    }
    with open(json_name, "w") as f:
        json.dump(companion, f)
        
    print(f"  Log generated: {log_name}")
    return log_name


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IMC Prosperity 4 — Local Backtester")
    parser.add_argument("trader", help="Path to trader .py file")
    parser.add_argument("--data", "-d", required=True, help="Path to data directory")
    parser.add_argument("--day", type=int, default=None, help="Day to backtest (e.g. -1). Default: all days")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every fill")
    parser.add_argument("--save", "-s", action="store_true", help="Save results CSV")
    args = parser.parse_args()

    if not os.path.exists(args.trader):
        print(f"Trader not found: {args.trader}")
        sys.exit(1)

    trader_name = os.path.splitext(os.path.basename(args.trader))[0]
    print(f"Backtesting: {trader_name}")
    print(f"Data: {args.data}")

    # Load
    trader = load_trader(args.trader)
    prices, trades = load_data(args.data)

    if prices.empty:
        print("No price data found.")
        sys.exit(1)

    days = sorted(prices["day"].unique())
    if args.day is not None:
        if args.day not in days:
            print(f"Day {args.day} not found. Available: {days}")
            sys.exit(1)
        days = [args.day]

    print(f"Days: {days}")
    print(f"Products: {list(prices['product'].unique())}")
    print()

    grand_total = 0
    for day in days:
        print(f"Running Day {day}...")
        # Reload trader for each day (fresh state)
        trader = load_trader(args.trader)

        results, log_activities, log_trades = run_backtest(trader, prices, trades, day, verbose=args.verbose)
        print_summary(results, day)

        if not results.empty:
            last = results[results["timestamp"] == results["timestamp"].max()]
            grand_total += last["pnl"].sum()

        if args.save:
            save_results(results, trader_name, day)
        
        save_log(log_activities, log_trades, trader_name, day)

    if len(days) > 1:
        pnl_color = "\033[92m" if grand_total >= 0 else "\033[91m"
        print(f"{'='*60}")
        print(f"  GRAND TOTAL:  {pnl_color}{grand_total:>+10.1f}\033[0m")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
