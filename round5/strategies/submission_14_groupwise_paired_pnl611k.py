"""
SUBMISSION 14: Group-Wise Paired Directional + SNACKPACK Pair
================================================================
For each group, take BOTH the strongest LONG and strongest SHORT
to create a self-hedging "long-short" pair within the group.
This naturally exploits the within-group anti-correlations.

Plus SNACKPACK CHOC/VAN stat-arb on top.
"""
from datamodel import Order, TradingState
from typing import Dict, List

POSITION_LIMIT = 10

# Group-paired longs/shorts (pick strongest in each group on both sides)
DIRECTION = {
    # PEBBLES: XL long vs XS short (strongest +60 vs strongest -39)
    "PEBBLES_XL":  +1,
    "PEBBLES_XS":  -1,
    "PEBBLES_S":   -1,

    # MICROCHIP: SQUARE long, OVAL/TRIANGLE short
    "MICROCHIP_SQUARE":   +1,
    "MICROCHIP_OVAL":     -1,
    "MICROCHIP_TRIANGLE": -1,

    # OXYGEN_SHAKE: GARLIC long vs EVENING_BREATH short
    "OXYGEN_SHAKE_GARLIC":         +1,
    "OXYGEN_SHAKE_EVENING_BREATH": -1,

    # GALAXY_SOUNDS: BLACK_HOLES long vs PLANETARY_RINGS short
    "GALAXY_SOUNDS_BLACK_HOLES":    +1,
    "GALAXY_SOUNDS_PLANETARY_RINGS": -1,

    # UV_VISOR: RED/MAGENTA long, AMBER/ORANGE short
    "UV_VISOR_RED":      +1,
    "UV_VISOR_MAGENTA":  +1,
    "UV_VISOR_AMBER":    -1,
    "UV_VISOR_ORANGE":   -1,

    # PANEL: 2X4 long, 4X4/1X4 short
    "PANEL_2X4":  +1,
    "PANEL_4X4":  -1,
    "PANEL_1X4":  -1,

    # ROBOT: MOPPING long, IRONING/VACUUMING short
    "ROBOT_MOPPING":   +1,
    "ROBOT_IRONING":   -1,
    "ROBOT_VACUUMING": -1,

    # TRANSLATOR: VOID_BLUE long, SPACE_GRAY/ASTRO_BLACK short
    "TRANSLATOR_VOID_BLUE":   +1,
    "TRANSLATOR_SPACE_GRAY":  -1,
    "TRANSLATOR_ASTRO_BLACK": -1,

    # SLEEP_POD: POLYESTER/SUEDE/COTTON long (correlated cluster)
    "SLEEP_POD_POLYESTER": +1,
    "SLEEP_POD_SUEDE":     +1,
    "SLEEP_POD_COTTON":    +1,

    # SNACKPACK: STRAWBERRY long, PISTACHIO short (CHOC/VAN handled separately)
    "SNACKPACK_STRAWBERRY":  +1,
    "SNACKPACK_PISTACHIO":   -1,
}

# CHOC/VAN pair-trade override
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

        # AGGRESSIVE DIRECTIONAL SIGNALS DISABLED (group-paired unconditional buy/sell removed)
        # Keeping only gated pair-trade signals below

        # SNACKPACK CHOC/VAN pair
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
                qty = min(need, -depth.sell_orders[best_ask])
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
            elif need < 0:
                qty = min(-need, depth.buy_orders[best_bid])
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
            if orders:
                result[product] = orders

        return result, 0, ""
