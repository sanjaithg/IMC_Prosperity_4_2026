"""
SUBMISSION 01: Naive Buy-and-Hold (Top 10 Strongest Trends)
=============================================================
Simplest strategy: pin position to ±10 in the trend direction
for the 10 products with the strongest 3-day directional moves.

Trends from Day 2-4 backtest data (% net change):
  +60.7%  PEBBLES_XL              LONG
  -44.8%  MICROCHIP_OVAL          SHORT
  -39.6%  PEBBLES_XS              SHORT
  +38.9%  OXYGEN_SHAKE_GARLIC     LONG
  +36.3%  MICROCHIP_SQUARE        LONG
  +34.6%  GALAXY_SOUNDS_BLACK_HOLES LONG
  -28.7%  UV_VISOR_AMBER          SHORT
  +23.5%  PANEL_2X4               LONG
  -21.7%  ROBOT_IRONING           SHORT
  -20.6%  MICROCHIP_TRIANGLE      SHORT
"""
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List

POSITION_LIMIT = 10

DIRECTION = {
    "PEBBLES_XL":                 +1,
    "MICROCHIP_OVAL":             -1,
    "PEBBLES_XS":                 -1,
    "OXYGEN_SHAKE_GARLIC":        +1,
    "MICROCHIP_SQUARE":           +1,
    "GALAXY_SOUNDS_BLACK_HOLES":  +1,
    "UV_VISOR_AMBER":             -1,
    "PANEL_2X4":                  +1,
    "ROBOT_IRONING":              -1,
    "MICROCHIP_TRIANGLE":         -1,
}


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, direction in DIRECTION.items():
            depth = state.order_depths.get(product)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue

            pos = state.position.get(product, 0)
            target = direction * POSITION_LIMIT
            need = target - pos

            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            orders: List[Order] = []

            if need > 0:
                # Need to buy
                ask_avail = -depth.sell_orders[best_ask]
                qty = min(need, ask_avail)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
            elif need < 0:
                # Need to sell
                bid_avail = depth.buy_orders[best_bid]
                qty = min(-need, bid_avail)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

            if orders:
                result[product] = orders

        return result, 0, ""
