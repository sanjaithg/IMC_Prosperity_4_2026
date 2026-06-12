"""
SUBMISSION 04: Top 20 Directional
==================================
Hold ±10 in trend direction for the top 20 strongest 3-day movers.
Same idea as #1 but expanded to 20 names so we capture more alpha.
"""
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List

POSITION_LIMIT = 10

DIRECTION = {
    "PEBBLES_XL":                  +1,  # +60.7%
    "MICROCHIP_OVAL":              -1,  # -44.8%
    "PEBBLES_XS":                  -1,  # -39.6%
    "OXYGEN_SHAKE_GARLIC":         +1,  # +38.9%
    "MICROCHIP_SQUARE":            +1,  # +36.3%
    "GALAXY_SOUNDS_BLACK_HOLES":   +1,  # +34.6%
    "UV_VISOR_AMBER":              -1,  # -28.7%
    "PANEL_2X4":                   +1,  # +23.5%
    "ROBOT_IRONING":               -1,  # -21.7%
    "MICROCHIP_TRIANGLE":          -1,  # -20.6%
    "SLEEP_POD_POLYESTER":         +1,  # +19.7%
    "PEBBLES_S":                   -1,  # -19.3%
    "SLEEP_POD_SUEDE":             +1,  # +18.0%
    "ROBOT_VACUUMING":             -1,  # -17.3%
    "UV_VISOR_RED":                +1,  # +17.2%
    "ROBOT_MOPPING":               +1,  # +15.9%
    "TRANSLATOR_SPACE_GRAY":       -1,  # -15.7%
    "TRANSLATOR_VOID_BLUE":        +1,  # +15.6%
    "UV_VISOR_MAGENTA":            +1,  # +15.3%
    "SLEEP_POD_COTTON":            +1,  # +14.1%
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
                ask_avail = -depth.sell_orders[best_ask]
                qty = min(need, ask_avail)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
            elif need < 0:
                bid_avail = depth.buy_orders[best_bid]
                qty = min(-need, bid_avail)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))

            if orders:
                result[product] = orders

        return result, 0, ""
