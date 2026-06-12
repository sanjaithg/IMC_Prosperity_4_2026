"""
submission_final.py
=====================
Merged final submission combining:

  (A) sub34 — quad-voucher mean-reversion ladder (VELVETFRUIT + VEV_*)
      * Hybrid static-tier + rolling-SMA(2000) target
      * 4 voucher spread signals (4500/15-17, 5000/5-7, 5100/3-5, 5200/2-4)
      * 2 butterfly votes, VELVET book imbalance, lead-lag amp
      * Sonic's tight-surface buy gate
      * Walk speeds: 10 / 25 / 40 / 55

  (B) X2 — inventory-skewed MM with TAKE layer on VELVETFRUIT
      * Captures obvious book dislocations (ask < fair - 2)

  (C) X1 — HYDROGEL_PACK mean-reversion state machine
      * IDLE -> SHORT (on spread spike) -> WAIT (spread normalizes)
        -> BUY (when price falling) -> back to IDLE

Conflict resolution
-------------------
HYDROGEL_PACK -> X1 only (sub34 doesn't trade it)
VELVETFRUIT_EXTRACT -> sub34 + X2 TAKE layer prepended
VEV_* -> sub34 only

State (traderData)
------------------
Single JSON dict with separate keys for each strategy:
  sma_buf, sma_sum, h45, h5k -> sub34
  hyd_spreads, hyd_prices, hyd_mode, hyd_local_min -> X1
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json


class Trader:
    POSITION_LIMIT = 200
    VOUCHER_LIMIT = 300
    SPREAD_GATE = 2

    # ─── sub34: price tiers ───────────────────────────────────────
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

    # ─── X2: VELVET take-layer ────────────────────────────────────
    X2_INV_SKEW = 2.0          # tick shift in fair when |pos|=LIMIT
    X2_TAKE_EDGE = 2           # take asks <= fair - 2 / bids >= fair + 2

    # ─── X1: HYDROGEL_PACK state-machine ──────────────────────────
    HYD_PRODUCT = "HYDROGEL_PACK"
    HYD_LIMIT = 200
    HYD_HIST_WINDOW = 200
    HYD_WARMUP = 100
    HYD_BUY_LOT = 1

    # ──────────────────────────────────────────────────────────────
    # Helpers (sub34 core)
    # ──────────────────────────────────────────────────────────────
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
        return max(-1.0, min(1.0, -z))

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

    def _walk_one(self, product, od, pos, target_frac, lot):
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
        if diff > 0:
            qty = min(lot, diff, limit - pos)
            if qty > 0:
                return [Order(product, bid + 1, qty)]
        else:
            qty = min(lot, -diff, limit + pos)
            if qty > 0:
                return [Order(product, ask - 1, -qty)]
        return []

    # ──────────────────────────────────────────────────────────────
    # X2: take-layer for VELVETFRUIT  (run BEFORE the walk)
    # ──────────────────────────────────────────────────────────────
    def _x2_take_orders(self, depth: OrderDepth, pos: int, mid: float):
        """Take any obvious dislocations on VELVET vs inventory-skewed fair."""
        skew = -(pos / self.POSITION_LIMIT) * self.X2_INV_SKEW
        fair = mid + skew
        room_buy  = self.POSITION_LIMIT - pos
        room_sell = self.POSITION_LIMIT + pos
        orders: List[Order] = []
        # buy any asks <= fair - TAKE_EDGE
        if room_buy > 0:
            for px in sorted(depth.sell_orders.keys()):
                if px > fair - self.X2_TAKE_EDGE:
                    break
                avail = -depth.sell_orders[px]
                qty = min(room_buy, avail)
                if qty > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", px, qty))
                    room_buy -= qty
                if room_buy <= 0: break
        # sell any bids >= fair + TAKE_EDGE
        if room_sell > 0:
            for px in sorted(depth.buy_orders.keys(), reverse=True):
                if px < fair + self.X2_TAKE_EDGE:
                    break
                avail = depth.buy_orders[px]
                qty = min(room_sell, avail)
                if qty > 0:
                    orders.append(Order("VELVETFRUIT_EXTRACT", px, -qty))
                    room_sell -= qty
                if room_sell <= 0: break
        return orders

    # ──────────────────────────────────────────────────────────────
    # X1: HYDROGEL_PACK state-machine
    # ──────────────────────────────────────────────────────────────
    def _hyd_run(self, od, pos, mem):
        spreads = list(mem.get("hyd_spreads", []))
        prices  = list(mem.get("hyd_prices", []))
        mode    = mem.get("hyd_mode", "IDLE")
        local_min = mem.get("hyd_local_min", None)

        orders: List[Order] = []
        if not od or not od.buy_orders or not od.sell_orders:
            mem["hyd_spreads"] = spreads[-self.HYD_HIST_WINDOW:]
            mem["hyd_prices"]  = prices[-self.HYD_HIST_WINDOW:]
            mem["hyd_mode"]    = mode
            mem["hyd_local_min"] = local_min
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = 0.5 * (best_bid + best_ask)
        spread = best_ask - best_bid

        spreads.append(spread)
        prices.append(mid)
        spreads = spreads[-self.HYD_HIST_WINDOW:]
        prices  = prices[-self.HYD_HIST_WINDOW:]

        buy_room  = self.HYD_LIMIT - pos
        sell_room = self.HYD_LIMIT + pos

        # State correction
        if mode == "IDLE" and pos <= -150:
            mode = "SHORT"
        elif pos >= 0 and mode == "BUY":
            mode = "IDLE"

        if len(spreads) >= self.HYD_WARMUP:
            avg_spread = sum(spreads[-self.HYD_WARMUP:]) / self.HYD_WARMUP

            # Step 1: SHORT on spread spike
            if mode == "IDLE" and spread > avg_spread:
                if sell_room > 0:
                    qty = min(self.HYD_LIMIT, sell_room)
                    orders.append(Order(self.HYD_PRODUCT, best_bid, -qty))

            # Step 2: WAIT after short
            elif mode == "SHORT" and spread <= avg_spread:
                mode = "WAIT"
                local_min = mid

            # Trend logic for buy entry
            falling_trend = False
            if len(prices) >= 60:
                short_avg = sum(prices[-15:]) / 15
                long_avg = sum(prices[-60:-15]) / 45
                falling_trend = short_avg < long_avg

            # Step 3: WAIT -> BUY entry
            if mode == "WAIT":
                if local_min is None or mid < local_min:
                    local_min = mid
                if falling_trend:
                    mode = "BUY"

            # Step 4: BUY (DCA only)
            elif mode == "BUY":
                if falling_trend and buy_room > 0:
                    qty = min(self.HYD_BUY_LOT, buy_room)
                    orders.append(Order(self.HYD_PRODUCT, best_ask, qty))
                if not falling_trend and pos > 0:
                    mode = "IDLE"
                    local_min = None

        mem["hyd_spreads"]   = spreads
        mem["hyd_prices"]    = prices
        mem["hyd_mode"]      = mode
        mem["hyd_local_min"] = local_min
        return orders

    # ──────────────────────────────────────────────────────────────
    # MAIN
    # ──────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        mem = self._load(state.traderData)
        result: Dict[str, List[Order]] = {}
        positions = state.position or {}

        # ─── (C) HYDROGEL strategy (X1) ──────────────────────────
        hyd_depth = state.order_depths.get(self.HYD_PRODUCT)
        if hyd_depth is not None:
            hyd_pos = int(positions.get(self.HYD_PRODUCT, 0))
            hyd_orders = self._hyd_run(hyd_depth, hyd_pos, mem)
            if hyd_orders:
                result[self.HYD_PRODUCT] = hyd_orders

        # ─── (A) sub34 framework: VELVET + VEV_* ─────────────────
        und_mid = self._mid(state.order_depths.get("VELVETFRUIT_EXTRACT"))
        if und_mid is None:
            for p in state.order_depths:
                result.setdefault(p, [])
            return result, 0, self._dump(mem)

        # Update rolling SMA
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

        # ─── VELVET: X2 take layer FIRST, then walk ──────────────
        velv_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        velv_pos_start = int(positions.get("VELVETFRUIT_EXTRACT", 0))
        velv_orders: List[Order] = []
        if velv_depth is not None:
            x2 = self._x2_take_orders(velv_depth, velv_pos_start, und_mid)
            velv_orders.extend(x2)

        # ─── Walk for each MM product ────────────────────────────
        for product in state.order_depths:
            if product == self.HYD_PRODUCT:
                continue                          # handled above
            if product not in self.MM_PRODUCTS:
                result.setdefault(product, [])
                continue

            pos = int(positions.get(product, 0))
            # account for VELVET take layer fills already submitted
            if product == "VELVETFRUIT_EXTRACT":
                pos += sum(o.quantity for o in velv_orders)

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

            walk_orders = self._walk_one(
                product, state.order_depths.get(product), pos, target_frac, lot,
            )
            if product == "VELVETFRUIT_EXTRACT":
                velv_orders.extend(walk_orders)
            elif walk_orders:
                result[product] = walk_orders

        if velv_orders:
            result["VELVETFRUIT_EXTRACT"] = velv_orders

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
