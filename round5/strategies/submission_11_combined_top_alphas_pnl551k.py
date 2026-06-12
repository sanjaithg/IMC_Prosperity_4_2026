"""
SUBMISSION 11: Combined Top Alphas (Directional + SNACKPACK pair)
===================================================================
Combines:
  - Top 20 directional plays (held at ±10)
  - SNACKPACK CHOC/VAN pair trade overrides (if z > entry)

For SNACKPACK CHOC/VAN, we IGNORE the directional bias and use
the pair-trade signal instead, since the constraint is tight.
"""
from datamodel import Order, TradingState
from typing import Dict, List

POSITION_LIMIT = 10

# === Directional positions ===
DIRECTION = {
    "PEBBLES_XL":                  +1,
    "MICROCHIP_OVAL":              -1,
    "PEBBLES_XS":                  -1,
    "OXYGEN_SHAKE_GARLIC":         +1,
    "MICROCHIP_SQUARE":            +1,
    "GALAXY_SOUNDS_BLACK_HOLES":   +1,
    "UV_VISOR_AMBER":              -1,
    "PANEL_2X4":                   +1,
    "ROBOT_IRONING":               -1,
    "MICROCHIP_TRIANGLE":          -1,
    "SLEEP_POD_POLYESTER":         +1,
    "PEBBLES_S":                   -1,
    "SLEEP_POD_SUEDE":             +1,
    "ROBOT_VACUUMING":             -1,
    "UV_VISOR_RED":                +1,
    "ROBOT_MOPPING":               +1,
    "TRANSLATOR_SPACE_GRAY":       -1,
    "TRANSLATOR_VOID_BLUE":        +1,
    "UV_VISOR_MAGENTA":            +1,
    "SLEEP_POD_COTTON":            +1,
}

# SNACKPACK pair stat-arb overrides
CHOC = "SNACKPACK_CHOCOLATE"
VAN = "SNACKPACK_VANILLA"
MEAN_DIFF = -253.93
STD_DIFF = 372.18
ENTRY_Z = 1.0
EXIT_Z = 0.2


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        targets: Dict[str, int] = {}

        # AGGRESSIVE DIRECTIONAL SIGNALS DISABLED (Tier A unconditional buy/sell removed)
        # Keeping only gated pair-trade signals below

        # Override CHOC/VAN with pair trade
        d_choc = state.order_depths.get(CHOC)
        d_van = state.order_depths.get(VAN)
        if d_choc and d_van and d_choc.buy_orders and d_choc.sell_orders \
                and d_van.buy_orders and d_van.sell_orders:
            mid_c = 0.5 * (max(d_choc.buy_orders) + min(d_choc.sell_orders))
            mid_v = 0.5 * (max(d_van.buy_orders) + min(d_van.sell_orders))
            z = ((mid_c - mid_v) - MEAN_DIFF) / STD_DIFF
            pos_c = state.position.get(CHOC, 0)
            pos_v = state.position.get(VAN, 0)
            if z > ENTRY_Z:
                targets[CHOC], targets[VAN] = -POSITION_LIMIT, +POSITION_LIMIT
            elif z < -ENTRY_Z:
                targets[CHOC], targets[VAN] = +POSITION_LIMIT, -POSITION_LIMIT
            elif abs(z) < EXIT_Z:
                targets[CHOC], targets[VAN] = 0, 0
            else:
                targets[CHOC], targets[VAN] = pos_c, pos_v

        # Send orders
        for product, target in targets.items():
            depth = state.order_depths.get(product)
            if not depth or not depth.buy_orders or not depth.sell_orders:
                continue
            pos = state.position.get(product, 0)
            need = target - pos
            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            orders: List[Order] = []
            if need > 0:
                avail = -depth.sell_orders[best_ask]
                qty = min(need, avail)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
            elif need < 0:
                avail = depth.buy_orders[best_bid]
                qty = min(-need, avail)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
            if orders:
                result[product] = orders

        return result, 0, ""
