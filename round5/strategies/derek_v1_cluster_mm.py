"""
Round 5 Submission — v1: Cluster-informed market making
========================================================

Strategy summary
----------------
1. **Market making on all 50 products** with an order-book imbalance tilt.
   Fair value is estimated as an EMA of mid-price + half-spread × EMA-imbalance.
   Quotes are posted at fair ± half_spread.  Sizes scale down as we approach
   the ±10 position limit.

2. **Aggressive taking** when the smoothed imbalance is very strong (|imb| > 0.4).
   We cross the spread to capture the directional move.

3. **Inventory lean** — when position > LEAN_THRESHOLD, we widen the quote on
   the side that grows our position and tighten on the side that reduces it, so
   we drift flat over time.

4. **Within-cluster relative-value** for the ONLY clusters with meaningful
   correlation (|corr| > 0.1) found from Round 5 data:

   • SNACKPACK_PISTACHIO cluster  (PISTACHIO ↔ STRAWBERRY +0.913, RASPBERRY mirror −0.923/−0.831)
   • SNACKPACK_CHOCOLATE cluster  (CHOCOLATE ↔ VANILLA −0.915, anti-correlated pair)
   • PEBBLES_XL cluster           (XL ↔ L/M/S/XS −0.475 to −0.506, XL moves opposite)

   NOTE: 40 of the 50 products have NO meaningful correlation with anything else.
   The old clusters (GALAXY_SOUNDS, MICROCHIP, etc.) were AP artefacts based on
   noise (avg intra-cluster corr < 0.01) and have been removed.

Position limit: 10 for every product.
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POS_LIMIT      = 10
LEAN_THRESHOLD = 6      # when |pos| >= this, lean quotes to reduce inventory
CLIP           = 4      # max units per side per tick (keeps it conservative)

# EMA decay rates
ALPHA_MID = 0.10   # slow — tracks fair value
ALPHA_IMB = 0.25   # faster — captures short-term imbalance
ALPHA_CLU = 0.10   # cluster mean EMA

# Signal thresholds
IMB_TAKE  = 0.40   # cross the spread and take when |ema_imb| exceeds this
Z_THRESH  = 1.5    # relative-value z-score to lean quotes (cluster signal)

# ---------------------------------------------------------------------------
# All 50 products
# ---------------------------------------------------------------------------
def _ab(p: str) -> str:
    """4-char abbreviation: first letter of each underscore segment."""
    return "".join(s[0] for s in p.split("_"))[:4]


ALL_PRODUCTS: List[str] = [
    "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
    "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
    "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
    "ROBOT_LAUNDRY", "ROBOT_IRONING",
    "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
    "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
    "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
]

# ---------------------------------------------------------------------------
# Meaningful RV clusters — ONLY products with |corr| > 0.1 from Round 5 data.
# Key insight: 40 of 50 products are uncorrelated noise — do NOT trade RV on them.
#
# Within each cluster the z-score of a product's normalised mid vs the cluster
# mean drives a lean: expensive → lean sell, cheap → lean buy.
#
# Note on signs:
#   SNACKPACK_PISTACHIO cluster  — PISTACHIO & STRAWBERRY co-move (+0.913),
#                                   RASPBERRY is their MIRROR (−0.923/−0.831).
#                                   The z-score of RASPBERRY should be *inverted*
#                                   before use (handled in the RV block below).
#   SNACKPACK_CHOCOLATE cluster  — CHOCOLATE & VANILLA are anti-correlated (−0.915).
#   PEBBLES_XL cluster           — XL moves OPPOSITE to L/M/S/XS (−0.475 to −0.506).
# ---------------------------------------------------------------------------
RV_CLUSTERS: Dict[str, List[str]] = {
    "snackpack_pistrawb": [          # PISTACHIO & STRAWBERRY co-move
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
    ],
    "snackpack_rasp_mirror": [       # RASPBERRY is the mirror of PISTACHIO/STRAWBERRY
        "SNACKPACK_RASPBERRY",
    ],
    "snackpack_choc_van": [          # CHOCOLATE & VANILLA anti-correlated pair
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
    ],
    "pebbles_body": [                # L / M / S / XS move together
        "PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XS",
    ],
    "pebbles_xl_mirror": [           # XL moves opposite to the body
        "PEBBLES_XL",
    ],
}

# Mapping from mirror cluster → the cluster it tracks (inverted)
RV_MIRRORS: Dict[str, str] = {
    "snackpack_rasp_mirror": "snackpack_pistrawb",
    "pebbles_xl_mirror":     "pebbles_body",
}


# ---------------------------------------------------------------------------
# Helper: extract best bid / ask and volumes from an OrderDepth
# ---------------------------------------------------------------------------
def _book_stats(od: OrderDepth) -> Tuple[int, int, float, float, float]:
    """Returns (best_bid, best_ask, mid, spread, imbalance).  Raises if no book."""
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    mid      = (best_bid + best_ask) / 2.0
    spread   = float(best_ask - best_bid)
    bid_vol  = sum(v for v in od.buy_orders.values() if v > 0)
    ask_vol  = sum(abs(v) for v in od.sell_orders.values())
    denom    = bid_vol + ask_vol
    imb      = (bid_vol - ask_vol) / denom if denom > 0 else 0.0
    return best_bid, best_ask, mid, spread, imb


# ---------------------------------------------------------------------------
# Main Trader class
# ---------------------------------------------------------------------------
class Trader:

    def run(self, state: TradingState):
        # ---- Load persistent state ----------------------------------------
        try:
            data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        result: Dict[str, List[Order]] = {}
        n_orders = 0
        logs: List[str] = []   # notable events this tick
        # ---- Per-product pass -----------------------------------------------
        for product in ALL_PRODUCTS:
            od = state.order_depths.get(product)
            if od is None or not od.buy_orders or not od.sell_orders:
                continue

            try:
                best_bid, best_ask, mid, spread, imb = _book_stats(od)
            except Exception:
                continue

            pos = state.position.get(product, 0)

            # ---- EMA updates -----------------------------------------------
            ema_mid = data.get(f"em_{product}", mid)
            ema_imb = data.get(f"ei_{product}", 0.0)
            ema_mid = ALPHA_MID * mid      + (1 - ALPHA_MID) * ema_mid
            ema_imb = ALPHA_IMB * imb      + (1 - ALPHA_IMB) * ema_imb
            data[f"em_{product}"] = ema_mid
            data[f"ei_{product}"] = ema_imb

            # ---- Relative-value z-score (cluster signal) -------------------
            # For mirror clusters (RASPBERRY, PEBBLES_XL) we compare against
            # the *opposite* cluster mean and invert the sign.
            rv_z = 0.0
            for cluster_name, members in RV_CLUSTERS.items():
                if product not in members:
                    continue
                # Which cluster provides the reference mean?
                ref_cluster = RV_MIRRORS.get(cluster_name, cluster_name)
                ref_members = RV_CLUSTERS[ref_cluster]
                ckey = f"ec_{ref_cluster}"
                c_mean = data.get(ckey, ema_mid)
                n = max(len(ref_members), 1)
                c_mean = (ALPHA_CLU / n) * mid + (1 - ALPHA_CLU / n) * c_mean
                data[ckey] = c_mean
                scale = max(spread, 0.5)
                raw_z = (ema_mid - c_mean) / scale
                # Mirror clusters move OPPOSITE so flip the signal
                rv_z = -raw_z if cluster_name in RV_MIRRORS else raw_z
                break

            # ---- Fair value ------------------------------------------------
            # Tilt fair value by imbalance (positive imb → price likely higher)
            tilt  = ema_imb * spread * 0.5
            # Also tilt by relative-value z: if product is expensive vs cluster,
            # lower our fair value estimate (sell-lean)
            rv_tilt = -rv_z * spread * 0.25
            fair  = ema_mid + tilt + rv_tilt

            # ---- Position lean ---------------------------------------------
            # When inventory is large, lean quotes to mean-revert position.
            pos_ratio   = pos / POS_LIMIT          # in [-1, 1]
            pos_tilt    = -pos_ratio * spread * 0.3

            bid_price = round(fair + pos_tilt - spread * 0.5)
            ask_price = round(fair + pos_tilt + spread * 0.5)

            # Keep quotes inside the current spread (don't cross the market)
            bid_price = min(bid_price, best_bid)
            ask_price = max(ask_price, best_ask)

            orders: List[Order] = []

            # ---- Aggressive taking on extreme imbalance --------------------
            net_imb_signal = ema_imb - rv_z * 0.1   # blend imb + rv signal
            if net_imb_signal > IMB_TAKE and pos < POS_LIMIT:
                take_qty = min(POS_LIMIT - pos, CLIP)
                if take_qty > 0:
                    # Buy at best ask (lift the offer)
                    avail = sum(abs(v) for v in od.sell_orders.values())
                    take_qty = min(take_qty, avail)
                    if take_qty > 0:
                        orders.append(Order(product, best_ask, take_qty))
                        pos += take_qty
                        logs.append(f"BUY {_ab(product)} imb={ema_imb:+.2f}")

            elif net_imb_signal < -IMB_TAKE and pos > -POS_LIMIT:
                take_qty = min(POS_LIMIT + pos, CLIP)
                if take_qty > 0:
                    avail = sum(v for v in od.buy_orders.values() if v > 0)
                    take_qty = min(take_qty, avail)
                    if take_qty > 0:
                        orders.append(Order(product, best_bid, -take_qty))
                        pos -= take_qty
                        logs.append(f"SEL {_ab(product)} imb={ema_imb:+.2f}")

            # ---- Passive market-making quotes --------------------------------
            max_buy  = POS_LIMIT - pos
            max_sell = POS_LIMIT + pos

            # Scale down quote size as we approach limits
            if abs(pos) >= LEAN_THRESHOLD:
                # Only quote the side that reduces inventory
                if pos > 0:
                    max_buy  = max(0, POS_LIMIT - pos - (pos - LEAN_THRESHOLD))
                else:
                    max_sell = max(0, POS_LIMIT + pos - (-pos - LEAN_THRESHOLD))

            buy_qty  = max(0, min(max_buy,  CLIP))
            sell_qty = max(0, min(max_sell, CLIP))

            if buy_qty > 0:
                orders.append(Order(product, bid_price,  buy_qty))
            if sell_qty > 0:
                orders.append(Order(product, ask_price, -sell_qty))

            if orders:
                result[product] = orders
                n_orders += len(orders)
                if abs(pos) >= POS_LIMIT - 1:
                    logs.append(f"LIM {_ab(product)} {pos:+d}")

        # ---- Persist state --------------------------------------------------
        # Print compact tick summary + any notable events (stay < 100 chars/tick)
        summary = f"t={state.timestamp} o={n_orders}"
        if logs:
            summary += " | " + " ".join(logs[:3])   # cap at 3 events per tick
        print(summary[:98])

        trader_data = json.dumps(data)
        return result, 0, trader_data
