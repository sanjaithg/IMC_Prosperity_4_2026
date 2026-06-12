"""
submission35_top_of_book_mr.py
================================
sub34 + aggressive top-of-orderbook taking on high-conviction trades.

What changes from sub34
-----------------------
sub34 always quotes passively (bid+1 / ask-1). Fine for low-conviction
ticks, but on high-conviction events (3+ aligned votes, or 2+ with
amplifier) we want GUARANTEED fills — informed flow doesn't wait around.

sub35 adds a TOP-OF-BOOK taking layer:
  - On LOT_FAST or LOT_MAX cycles, lift the best ask (when buying) or
    hit the best bid (when selling) up to available liquidity.
  - Whatever quantity isn't filled by taking, post passively at bid+1 /
    ask-1 (sub34 default).

Cost: pay the full spread on aggressive fills (vs half-spread on passive
fills). For VELVET with 5-chip spread that's 2.5 chips of cost per
aggressive fill. Worth it when the signal predicts a 5+ chip move at
lag 100.

Trigger logic (per-tick, per-product)
-------------------------------------
  aligned_votes < 2     -> passive only (sub34 default)
  aligned_votes >= 2    -> hybrid: take top-of-book + post passive
                            (with the same lot speed sub34 used)

This keeps the strategy passive in low-information regimes (most ticks)
and switches to aggressive ONLY when the multi-signal conviction is high.
"""

from datamodel import TradingState, Order
from typing import Dict, List
import json


class Trader:
    POSITION_LIMIT = 200
    VOUCHER_LIMIT = 300
    SPREAD_GATE = 2

    PRICE_TIERS = [
        (5220, +1.00),
        (5230, +0.65),
        (5240, +0.40),
        (5248, +0.20),
        (5252,  0.00),
        (5260, -0.20),
        (5270, -0.40),
        (5280, -0.65),
    ]
    EXTREME_SHORT = -1.00

    SMA_WINDOW = 2000
    SMA_WARMUP = 100
    DIST_SCALE = 12.0

    LOT_BASE  = 10
    LOT_BOOST = 25
    LOT_FAST  = 40
    LOT_MAX   = 55

    # Aggressive-take threshold: only at LOT_MAX (3+ votes or 2+amp).
    # 2-vote events are NOT high-conviction enough to pay full spread.
    AGGRESSIVE_LOT = 55       # = LOT_MAX only

    MM_PRODUCTS = {
        "VELVETFRUIT_EXTRACT",
        "VEV_4000",
        "VEV_4500",
        "VEV_5000",
        "VEV_5100",
        "VEV_5200",
        "VEV_5300",
    }

    SPREAD_SIGNALS = [
        ("VEV_4500", 15, 17),
        ("VEV_5000",  5,  7),
        ("VEV_5100",  3,  5),
        ("VEV_5200",  2,  4),
    ]

    BF_5100_HI = -6.25
    BF_5100_LO = -10.75
    BF_5200_HI = -10.0
    BF_5200_LO = -13.0

    IMB_BUY  = +0.40
    IMB_SELL = -0.30

    LEAD_WINDOW = 50
    TIGHT_SURFACE_PRODUCTS = ["VEV_5200", "VEV_5300"]
    TIGHT_SPREAD_THRESHOLD = 2

    # ─── helpers ──────────────────────────────────────────────────
    def _position_limit(self, product: str) -> int:
        return self.VOUCHER_LIMIT if product.startswith("VEV_") else self.POSITION_LIMIT

    @staticmethod
    def _spread(od):
        if not od or not od.buy_orders or not od.sell_orders:
            return None
        return min(od.sell_orders) - max(od.buy_orders)

    @staticmethod
    def _mid(od):
        if not od or not od.buy_orders or not od.sell_orders:
            return None
        return 0.5 * (max(od.buy_orders) + min(od.sell_orders))

    @staticmethod
    def _imb(od):
        if not od or not od.buy_orders or not od.sell_orders:
            return None
        bv = od.buy_orders[max(od.buy_orders)]
        av = -od.sell_orders[min(od.sell_orders)]
        s = bv + av
        if s <= 0: return None
        return (bv - av) / s

    def _static_target(self, price: float) -> float:
        for upper, frac in self.PRICE_TIERS:
            if price < upper:
                return frac
        return self.EXTREME_SHORT

    def _continuous_target(self, price: float, sma):
        if sma is None:
            return 0.0
        z = (price - sma) / self.DIST_SCALE
        target = -z
        return max(-1.0, min(1.0, target))

    def _spread_votes(self, state):
        buy = sell = 0
        sp45_buy = sp5k_buy = False
        for prod, bv, sv in self.SPREAD_SIGNALS:
            spr = self._spread(state.order_depths.get(prod))
            if spr == bv:
                buy += 1
                if prod == "VEV_4500": sp45_buy = True
                elif prod == "VEV_5000": sp5k_buy = True
            elif spr == sv:
                sell += 1
        return buy, sell, sp45_buy and sp5k_buy

    def _butterfly_votes(self, state):
        buy = sell = 0
        m5000 = self._mid(state.order_depths.get("VEV_5000"))
        m5100 = self._mid(state.order_depths.get("VEV_5100"))
        m5200 = self._mid(state.order_depths.get("VEV_5200"))
        m5300 = self._mid(state.order_depths.get("VEV_5300"))
        if None not in (m5000, m5100, m5200):
            bf = m5100 - 0.5 * (m5000 + m5200)
            if bf > self.BF_5100_HI: sell += 1
            elif bf < self.BF_5100_LO: buy += 1
        if None not in (m5100, m5200, m5300):
            bf = m5200 - 0.5 * (m5100 + m5300)
            if bf > self.BF_5200_HI: sell += 1
            elif bf < self.BF_5200_LO: buy += 1
        return buy, sell

    def _imbalance_votes(self, state):
        imb = self._imb(state.order_depths.get("VELVETFRUIT_EXTRACT"))
        if imb is None: return 0, 0
        if imb > self.IMB_BUY: return 1, 0
        if imb < self.IMB_SELL: return 0, 1
        return 0, 0

    def _tight_surface(self, state):
        for p in self.TIGHT_SURFACE_PRODUCTS:
            spr = self._spread(state.order_depths.get(p))
            if spr is None or spr > self.TIGHT_SPREAD_THRESHOLD:
                return False
        return True

    # ─── Order placement: hybrid aggressive + passive ─────────────
    def _walk_one(self, product, od, pos, target_frac, lot, aggressive: bool):
        if not od or not od.buy_orders or not od.sell_orders:
            return []
        bid = max(od.buy_orders); ask = min(od.sell_orders)
        if ask - bid <= self.SPREAD_GATE:
            return []
        limit = self._position_limit(product)
        target_pos = int(round(target_frac * limit))
        diff = target_pos - pos
        if diff == 0:
            return []
        orders: List[Order] = []
        if diff > 0:
            qty_remaining = min(lot, diff, limit - pos)
            if aggressive:
                # 1. Lift the best ask up to available liquidity
                ask_avail = -od.sell_orders[ask]
                take_qty = min(qty_remaining, ask_avail)
                if take_qty > 0:
                    orders.append(Order(product, ask, take_qty))
                    qty_remaining -= take_qty
            # 2. Post passive at bid+1 for the rest
            if qty_remaining > 0:
                orders.append(Order(product, bid + 1, qty_remaining))
        else:
            qty_remaining = min(lot, -diff, limit + pos)
            if aggressive:
                # 1. Hit the best bid up to available liquidity
                bid_avail = od.buy_orders[bid]
                take_qty = min(qty_remaining, bid_avail)
                if take_qty > 0:
                    orders.append(Order(product, bid, -take_qty))
                    qty_remaining -= take_qty
            # 2. Post passive at ask-1 for the rest
            if qty_remaining > 0:
                orders.append(Order(product, ask - 1, -qty_remaining))
        return orders

    # ─── main ─────────────────────────────────────────────────────
    def run(self, state: TradingState):
        mem = self._load(state.traderData)
        result: Dict[str, List[Order]] = {}
        positions = state.position or {}

        und_mid = self._mid(state.order_depths.get("VELVETFRUIT_EXTRACT"))
        if und_mid is None:
            for p in state.order_depths:
                result[p] = []
            return result, 0, self._dump(mem)

        # SMA update
        sma_sum = float(mem.get("sma_sum", 0.0))
        sma_buf = list(mem.get("sma_buf", []))
        sma_buf.append(und_mid)
        sma_sum += und_mid
        if len(sma_buf) > self.SMA_WINDOW:
            sma_sum -= sma_buf.pop(0)
        mem["sma_sum"] = sma_sum
        mem["sma_buf"] = sma_buf
        sma = (sma_sum / len(sma_buf)) if len(sma_buf) >= self.SMA_WARMUP else None

        # Hybrid target
        static_t = self._static_target(und_mid)
        cont_t   = self._continuous_target(und_mid, sma)
        target_frac = cont_t if abs(cont_t) > abs(static_t) else static_t

        # Lead-lag
        m45 = self._mid(state.order_depths.get("VEV_4500"))
        m5k = self._mid(state.order_depths.get("VEV_5000"))
        h45 = list(mem.get("h45", []))
        h5k = list(mem.get("h5k", []))
        if m45 is not None:
            h45.append(m45)
            if len(h45) > self.LEAD_WINDOW + 1: h45.pop(0)
        if m5k is not None:
            h5k.append(m5k)
            if len(h5k) > self.LEAD_WINDOW + 1: h5k.pop(0)
        mem["h45"] = h45
        mem["h5k"] = h5k

        lead_dev = None
        if len(h45) > self.LEAD_WINDOW and len(h5k) > self.LEAD_WINDOW:
            lead_dev = (h45[-1] - h45[0]) - 2.0 * (h5k[-1] - h5k[0])

        # Vote counting
        sb, ss, both_buy_45_5k = self._spread_votes(state)
        bb, bs = self._butterfly_votes(state)
        ib, is_ = self._imbalance_votes(state)
        buy_votes = sb + bb + ib
        sell_votes = ss + bs + is_

        tight = self._tight_surface(state)
        lead_buy_amp = (
            both_buy_45_5k and lead_dev is not None and lead_dev < 0
        )

        for product in state.order_depths:
            pos = int(positions.get(product, 0))
            if product not in self.MM_PRODUCTS:
                result[product] = []
                continue
            limit = self._position_limit(product)
            target_pos = int(round(target_frac * limit))
            diff = target_pos - pos

            if diff > 0:
                aligned = buy_votes
                if (aligned >= 3
                        or (aligned >= 2 and lead_buy_amp)
                        or (aligned >= 2 and tight)):
                    lot = self.LOT_MAX
                elif aligned >= 2:
                    lot = self.LOT_FAST
                elif aligned == 1:
                    lot = self.LOT_BOOST
                else:
                    lot = self.LOT_BASE
            elif diff < 0:
                aligned = sell_votes
                if aligned >= 3:
                    lot = self.LOT_MAX
                elif aligned >= 2:
                    lot = self.LOT_FAST
                elif aligned == 1:
                    lot = self.LOT_BOOST
                else:
                    lot = self.LOT_BASE
            else:
                lot = self.LOT_BASE

            # Aggressive only when lot >= AGGRESSIVE_LOT (i.e., LOT_FAST+)
            aggressive = lot >= self.AGGRESSIVE_LOT

            result[product] = self._walk_one(
                product, state.order_depths.get(product), pos,
                target_frac, lot, aggressive,
            )

        return result, 0, self._dump(mem)

    @staticmethod
    def _load(td):
        if not td: return {}
        try:
            d = json.loads(td); return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _dump(d):
        try: return json.dumps(d, separators=(",", ":"))
        except Exception: return ""

    def bid(self):
        return 15
