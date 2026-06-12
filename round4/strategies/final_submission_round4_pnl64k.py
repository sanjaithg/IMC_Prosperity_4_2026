from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json


class Trader:
    POSITION_LIMIT = 200
    VOUCHER_LIMIT = 300
    SPREAD_GATE = 2

    # Deep OTM vouchers are quoted 0 x 1 in the local data.  We park a small
    # bid at 0; this costs nothing if filled and gives free upside if a tail
    # voucher ever starts trading above zero.  Local conservative matching
    # usually leaves these unfilled, so it should not disturb the core strategy.
    TAIL_LOTTO_PRODUCTS = ("VEV_6000", "VEV_6500")
    TAIL_LOTTO_TARGET = 300

    # ===================== HYDRO =====================
    HYD_WINDOW = 20

    # ===================== VELVET =====================
    PRICE_TIERS = [
        (5220, +1.00),
        (5230, +0.65),
        (5240, +0.40),
        (5248, +0.20),
        (5252, 0.00),
        (5260, -0.20),
        (5270, -0.40),
        (5280, -0.65),
    ]
    EXTREME_SHORT = -1.00

    LOT_BASE = 10
    LOT_BOOST = 25
    LOT_FAST = 40
    LOT_MAX = 55
    AGGRESSIVE_LOT = 55

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
        ("VEV_5000", 5, 7),
        ("VEV_5100", 3, 5),
        ("VEV_5200", 2, 4),
    ]

    BF_5100_HI = -6.25
    BF_5100_LO = -10.75
    BF_5200_HI = -10.0
    BF_5200_LO = -13.0

    IMB_BUY = +0.40
    IMB_SELL = -0.30

    LEAD_WINDOW = 50
    TIGHT_SURFACE_PRODUCTS = ["VEV_5200", "VEV_5300"]
    TIGHT_SPREAD_THRESHOLD = 2

    # ============================================================
    # ======================== MAIN ==============================
    # ============================================================

    def run(self, state: TradingState):
        mem = self._load(state.traderData)
        result: Dict[str, List[Order]] = {}
        positions = state.position or {}

        # ================= HYDRO MEMORY =================
        hyd_prices = mem.get("hyd_prices", [])

        # ================= VELVET CORE =================
        und_mid = self._mid(state.order_depths.get("VELVETFRUIT_EXTRACT"))
        if und_mid is not None:
            static_t = self._static_target(und_mid)
            target_frac = static_t

            # Lead-lag
            m45 = self._mid(state.order_depths.get("VEV_4500"))
            m5k = self._mid(state.order_depths.get("VEV_5000"))

            h45 = list(mem.get("h45", []))
            h5k = list(mem.get("h5k", []))

            if m45 is not None:
                h45.append(m45)
                if len(h45) > self.LEAD_WINDOW + 1:
                    h45.pop(0)

            if m5k is not None:
                h5k.append(m5k)
                if len(h5k) > self.LEAD_WINDOW + 1:
                    h5k.pop(0)

            mem["h45"] = h45
            mem["h5k"] = h5k

            lead_dev = None
            if len(h45) > self.LEAD_WINDOW and len(h5k) > self.LEAD_WINDOW:
                lead_dev = (h45[-1] - h45[0]) - 2.0 * (h5k[-1] - h5k[0])

            # Votes
            sb, ss, both_buy_45_5k = self._spread_votes(state)
            bb, bs = self._butterfly_votes(state)
            ib, is_ = self._imbalance_votes(state)

            buy_votes = sb + bb + ib
            sell_votes = ss + bs + is_

            tight = self._tight_surface(state)
            lead_buy_amp = both_buy_45_5k and lead_dev is not None and lead_dev < 0

            for product in self.MM_PRODUCTS:
                od = state.order_depths.get(product)
                if not od:
                    continue

                pos = int(positions.get(product, 0))
                limit = self._position_limit(product)
                target_pos = int(round(target_frac * limit))
                diff = target_pos - pos

                if diff > 0:
                    aligned = buy_votes
                    if aligned >= 3 or (aligned >= 2 and (lead_buy_amp or tight)):
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

                aggressive = lot >= self.AGGRESSIVE_LOT

                orders = self._walk_one(product, od, pos, target_frac, lot, aggressive)
                if orders:
                    result[product] = orders

        # ================= HYDRO TRADING =================
        if "HYDROGEL_PACK" in state.order_depths:
            od = state.order_depths["HYDROGEL_PACK"]
            pos = positions.get("HYDROGEL_PACK", 0)

            hyd_orders, hyd_prices = self.trade_hydro(od, pos, hyd_prices)
            if hyd_orders:
                result["HYDROGEL_PACK"] = hyd_orders

        # ================= FREE TAIL BIDS =================
        for product in self.TAIL_LOTTO_PRODUCTS:
            od = state.order_depths.get(product)
            pos = int(positions.get(product, 0))
            if od and pos < self.TAIL_LOTTO_TARGET:
                result.setdefault(product, []).append(
                    Order(product, 0, min(self.TAIL_LOTTO_TARGET - pos, self.VOUCHER_LIMIT - pos))
                )

        mem["hyd_prices"] = hyd_prices[-self.HYD_WINDOW :]

        return result, 0, self._dump(mem)

    # ============================================================
    # ================= HYDRO LOGIC ==============================
    # ============================================================

    def trade_hydro(self, od: OrderDepth, pos: int, prices: List[float]):
        orders: List[Order] = []

        if not od.buy_orders or not od.sell_orders:
            return orders, prices

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        mid = (best_bid + best_ask) / 2
        prices.append(mid)

        if len(prices) < self.HYD_WINDOW:
            return orders, prices

        fair = sum(prices) / len(prices)

        buy_room = self.POSITION_LIMIT - pos
        sell_room = self.POSITION_LIMIT + pos

        bid_price = best_bid + 1
        ask_price = best_ask - 1

        if mid < fair:
            if buy_room > 0:
                orders.append(Order("HYDROGEL_PACK", bid_price, buy_room))
            if pos > 0:
                orders.append(Order("HYDROGEL_PACK", int(fair + 1), -pos))

        elif mid > fair:
            if sell_room > 0:
                orders.append(Order("HYDROGEL_PACK", ask_price, -sell_room))
            if pos < 0:
                orders.append(Order("HYDROGEL_PACK", int(fair - 1), -pos))

        else:
            if buy_room > 0:
                orders.append(Order("HYDROGEL_PACK", bid_price, buy_room))
            if sell_room > 0:
                orders.append(Order("HYDROGEL_PACK", ask_price, -sell_room))

        return orders, prices

    # ============================================================
    # ================= VELVET HELPERS ===========================
    # ============================================================

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
        if s <= 0:
            return None
        return (bv - av) / s

    def _static_target(self, price: float) -> float:
        for upper, frac in self.PRICE_TIERS:
            if price < upper:
                return frac
        return self.EXTREME_SHORT

    def _spread_votes(self, state):
        buy = sell = 0
        sp45_buy = sp5k_buy = False
        for prod, bv, sv in self.SPREAD_SIGNALS:
            spr = self._spread(state.order_depths.get(prod))
            if spr == bv:
                buy += 1
                if prod == "VEV_4500":
                    sp45_buy = True
                elif prod == "VEV_5000":
                    sp5k_buy = True
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
            if bf > self.BF_5100_HI:
                sell += 1
            elif bf < self.BF_5100_LO:
                buy += 1

        if None not in (m5100, m5200, m5300):
            bf = m5200 - 0.5 * (m5100 + m5300)
            if bf > self.BF_5200_HI:
                sell += 1
            elif bf < self.BF_5200_LO:
                buy += 1

        return buy, sell

    def _imbalance_votes(self, state):
        imb = self._imb(state.order_depths.get("VELVETFRUIT_EXTRACT"))
        if imb is None:
            return 0, 0
        if imb > self.IMB_BUY:
            return 1, 0
        if imb < self.IMB_SELL:
            return 0, 1
        return 0, 0

    def _tight_surface(self, state):
        for p in self.TIGHT_SURFACE_PRODUCTS:
            spr = self._spread(state.order_depths.get(p))
            if spr is None or spr > self.TIGHT_SPREAD_THRESHOLD:
                return False
        return True

    def _walk_one(self, product, od, pos, target_frac, lot, aggressive: bool):
        if not od or not od.buy_orders or not od.sell_orders:
            return []

        bid = max(od.buy_orders)
        ask = min(od.sell_orders)

        if ask - bid <= self.SPREAD_GATE:
            return []

        limit = self._position_limit(product)
        target_pos = int(round(target_frac * limit))
        diff = target_pos - pos

        if diff == 0:
            return []

        orders: List[Order] = []

        if diff > 0:
            qty = min(lot, diff, limit - pos)
            if aggressive:
                take = min(qty, -od.sell_orders[ask])
                if take > 0:
                    orders.append(Order(product, ask, take))
                    qty -= take
            if qty > 0:
                orders.append(Order(product, bid + 1, qty))

        else:
            qty = min(lot, -diff, limit + pos)
            if aggressive:
                take = min(qty, od.buy_orders[bid])
                if take > 0:
                    orders.append(Order(product, bid, -take))
                    qty -= take
            if qty > 0:
                orders.append(Order(product, ask - 1, -qty))

        return orders

    @staticmethod
    def _load(td):
        if not td:
            return {}
        try:
            d = json.loads(td)
            return d if isinstance(d, dict) else {}
        except:
            return {}

    @staticmethod
    def _dump(d):
        try:
            return json.dumps(d, separators=(",", ":"))
        except:
            return ""