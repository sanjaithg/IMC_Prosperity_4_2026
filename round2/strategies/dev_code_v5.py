from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple


class Trader:

    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    # ── Osmium parameters ──
    FAIR_VALUE = 10000
    MIN_SPREAD = 2

    FALLBACK_BID = {"ASH_COATED_OSMIUM": 9910}
    FALLBACK_ASK = {"ASH_COATED_OSMIUM": 10090}

    SPIKE_THRESHOLD = 6
    MIN_LIQUIDITY = 5
    MM_SIZE = 30            # tuned

    SLOW_ALPHA = 0.002
    FADE_TRIGGER = 5
    FADE_GAIN = 6           # tuned

    # ── IPR parameters ──
    IPR_MM_SIZE = 5

    # ── Osmium state ──
    prev_best_bid = None
    prev_best_ask = None
    slow_fair = None

    # ── IPR state ──
    start_best_ask = None
    ipr_prev_best_ask = None

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            lim = self.POSITION_LIMITS.get(product, 0)

            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_ipr(od, pos, lim)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(od, pos, lim)
            else:
                result[product] = []
        return result, 0, ""

    # ─────────────────────────────────────────────
    # PEPPER ROOT — UNCHANGED (Original Logic)
    # ─────────────────────────────────────────────
    def trade_ipr(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        orders: List[Order] = []

        if not od.sell_orders:
            return orders

        best_ask = min(od.sell_orders.keys())

        if self.start_best_ask is None:
            self.start_best_ask = best_ask

        # ── Phase 1: Aggressive accumulation ──
        buy_remaining = lim - pos
        price_jumped = (
            self.ipr_prev_best_ask is not None
            and (best_ask - self.ipr_prev_best_ask) > 1
        )

        if buy_remaining > 0 and not price_jumped:
            sorted_asks = sorted(od.sell_orders.keys())
            prev_level = None
            for ask_price in sorted_asks:
                if prev_level is not None and (ask_price - prev_level) > 1:
                    break
                qty = min(-od.sell_orders[ask_price], buy_remaining)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, qty))
                    buy_remaining -= qty
                prev_level = ask_price
                if buy_remaining <= 0:
                    break

        self.ipr_prev_best_ask = best_ask

        # ── Phase 2: Passive MM bid at start_best_ask ──
        total_buys_so_far = sum(o.quantity for o in orders if o.quantity > 0)
        remaining_capacity = lim - pos - total_buys_so_far
        mm_qty = min(self.IPR_MM_SIZE, max(0, remaining_capacity))
        if mm_qty > 0:
            orders.append(
                Order("INTARIAN_PEPPER_ROOT", self.start_best_ask, mm_qty)
            )

        return orders

    # ─────────────────────────────────────────────
    # OSMIUM — SKEWED MARKET TAKING + MARKET MAKING
    # ─────────────────────────────────────────────
    def trade_osmium(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        orders: List[Order] = []

        if not od.buy_orders and not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # ── EMA FAIR VALUE CALCULATION (Kept for state tracking) ──
        mid = None
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
            if self.slow_fair is None:
                self.slow_fair = mid
            else:
                self.slow_fair = (
                    self.SLOW_ALPHA * mid + (1 - self.SLOW_ALPHA) * self.slow_fair
                )

        total_bid_vol = sum(od.buy_orders.values()) if od.buy_orders else 0
        total_ask_vol = sum(-v for v in od.sell_orders.values()) if od.sell_orders else 0
        has_bids = total_bid_vol >= self.MIN_LIQUIDITY
        has_asks = total_ask_vol >= self.MIN_LIQUIDITY

        # ── STATE TRACKING ──
        if best_bid is not None:
            self.prev_best_bid = best_bid
        if best_ask is not None:
            self.prev_best_ask = best_ask

        # =====================================================================
        # 1. CONTROLLED MARKET TAKING (Absolute Price Skewing)
        # Sweeps the book but strictly caps position based on price extremes.
        # Max taking is 60, leaving 20 slots reserved for passive MM.
        # =====================================================================

        # ── SELLING INTO SPIKES (Price >= 10005) ──
        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price >= 10005:
                    if bid_price >= 10010:
                        target_short = -60
                    else:
                        # Smooth linear interpolation between -10 and -60
                        target_short = -10 - ((bid_price - 10005) / 5.0) * 50
                    
                    target_short = max(-60, int(target_short))
                    
                    if pos > target_short:
                        need = pos - target_short
                        qty = min(od.buy_orders[bid_price], need)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            pos -= qty
                else:
                    break

        # ── BUYING INTO DIPS (Price <= 9995) ──
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price <= 9995:
                    if ask_price <= 9900:
                        target_long = 60
                    else:
                        # Smooth linear interpolation between +10 and +60
                        target_long = 10 + ((9995 - ask_price) / 95.0) * 50
                        
                    target_long = min(60, int(target_long))

                    if pos < target_long:
                        need = target_long - pos
                        qty = min(-od.sell_orders[ask_price], need)
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            pos += qty
                else:
                    break

        # =====================================================================
        # 2. LIQUIDITY PROVISION (Pure Market Making)
        # Uses the absolute limit (80) to soak up the final 20 slots.
        # =====================================================================
        if has_bids and has_asks and best_bid is not None and best_ask is not None:
            if best_ask - best_bid > self.MIN_SPREAD:
                buy_qty = min(self.MM_SIZE, lim - pos)
                if buy_qty > 0:
                    orders.append(Order(product, best_bid + 1, buy_qty)) 
                
                sell_qty = min(self.MM_SIZE, lim + pos)
                if sell_qty > 0:
                    orders.append(Order(product, best_ask - 1, -sell_qty)) 
        else:
            buy_qty = min(self.MM_SIZE, lim - pos)
            if buy_qty > 0:
                orders.append(Order(product, self.FALLBACK_BID[product], buy_qty))
            sell_qty = min(self.MM_SIZE, lim + pos)
            if sell_qty > 0:
                orders.append(Order(product, self.FALLBACK_ASK[product], -sell_qty))

        return orders