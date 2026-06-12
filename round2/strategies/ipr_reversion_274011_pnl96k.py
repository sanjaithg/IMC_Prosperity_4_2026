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

    # NEW: short-term reversion cycling around the long position
    IPR_CYCLE_WINDOW = 30          # lookback window (timestamps) for z-score
    IPR_CYCLE_Z_THRESH = 1.0       # z threshold to trigger a passive cycle quote
    IPR_CYCLE_SIZE = 6             # qty per cycle quote
    IPR_MAX_TRANSIENT_DEV = 20     # max units we'll go below target_pos transiently
    IPR_CYCLE_NEAR_TARGET = 10     # only cycle when (target - pos) ≤ this

    # ── Osmium state ──
    prev_best_bid = None
    prev_best_ask = None
    slow_fair = None

    # ── IPR state ──
    start_best_ask = None
    ipr_prev_best_ask = None
    ipr_mid_history: List[float] = []

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
    # PEPPER ROOT — accumulation unchanged, NEW reversion-cycle layer on top
    # ─────────────────────────────────────────────
    def trade_ipr(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        orders: List[Order] = []

        if not od.sell_orders:
            return orders

        best_ask = min(od.sell_orders.keys())
        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None

        # Track rolling mid history for the cycle signal
        if best_bid is not None:
            mid = (best_bid + best_ask) / 2.0
            self.ipr_mid_history.append(mid)
            if len(self.ipr_mid_history) > 200:
                self.ipr_mid_history.pop(0)

        if self.start_best_ask is None:
            self.start_best_ask = best_ask

        # ── Phase 1: Aggressive accumulation (unchanged) ──
        buy_remaining = lim - pos
        price_jumped = (
            self.ipr_prev_best_ask is not None
            and (best_ask - self.ipr_prev_best_ask) > 1
        )

        aggressive_buy_qty = 0
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
                    aggressive_buy_qty += qty
                prev_level = ask_price
                if buy_remaining <= 0:
                    break

        self.ipr_prev_best_ask = best_ask

        # ── NEW Phase 2: Short-term mean-reversion cycling ──
        # Kicks in once we're close to the long target, using a detrended z-score
        # on recent mids. Only posts PASSIVE quotes (ask-1 / bid+1) — never crosses.
        target_pos = lim
        near_target = (target_pos - pos) <= self.IPR_CYCLE_NEAR_TARGET
        have_history = len(self.ipr_mid_history) >= self.IPR_CYCLE_WINDOW

        if near_target and have_history and best_bid is not None:
            z = self._detrended_z(self.ipr_mid_history[-self.IPR_CYCLE_WINDOW:])

            # Price spiked ABOVE local trend → passive sell at ask-1
            if z > self.IPR_CYCLE_Z_THRESH:
                max_extra_short = max(
                    0, self.IPR_MAX_TRANSIENT_DEV - (target_pos - pos)
                )
                sell_cap = min(lim + pos, max_extra_short)
                sell_qty = min(self.IPR_CYCLE_SIZE, sell_cap)
                if sell_qty > 0:
                    orders.append(
                        Order("INTARIAN_PEPPER_ROOT", best_ask - 1, -sell_qty)
                    )

            # Price dipped BELOW local trend → passive buy at bid+1
            elif z < -self.IPR_CYCLE_Z_THRESH:
                buy_cap = min(lim - pos - aggressive_buy_qty, target_pos - pos)
                buy_qty = min(self.IPR_CYCLE_SIZE, max(0, buy_cap))
                if buy_qty > 0:
                    orders.append(
                        Order("INTARIAN_PEPPER_ROOT", best_bid + 1, buy_qty)
                    )

        # ── Phase 3: Passive MM bid at start_best_ask (unchanged) ──
        total_buys_so_far = sum(o.quantity for o in orders if o.quantity > 0)
        remaining_capacity = lim - pos - total_buys_so_far
        mm_qty = min(self.IPR_MM_SIZE, max(0, remaining_capacity))
        if mm_qty > 0:
            orders.append(
                Order("INTARIAN_PEPPER_ROOT", self.start_best_ask, mm_qty)
            )

        return orders

    @staticmethod
    def _detrended_z(window: List[float]) -> float:
        """Z-score of the last point's residual from a linear trend through window."""
        n = len(window)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(window) / n
        num = sum((xs[i] - mx) * (window[i] - my) for i in range(n))
        den = sum((xs[i] - mx) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        slope = num / den
        intercept = my - slope * mx
        residuals = [window[i] - (slope * xs[i] + intercept) for i in range(n)]
        var = sum(r * r for r in residuals) / n
        if var <= 0:
            return 0.0
        return residuals[-1] / (var ** 0.5)

    # ─────────────────────────────────────────────
    # OSMIUM (unchanged — the tuned version you already have)
    # ─────────────────────────────────────────────
    def trade_osmium(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        orders: List[Order] = []

        if not od.buy_orders and not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        mid = None
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
            if self.slow_fair is None:
                self.slow_fair = mid
            else:
                self.slow_fair = (
                    self.SLOW_ALPHA * mid + (1 - self.SLOW_ALPHA) * self.slow_fair
                )
        fair = self.slow_fair if self.slow_fair is not None else self.FAIR_VALUE

        total_bid_vol = sum(od.buy_orders.values()) if od.buy_orders else 0
        total_ask_vol = sum(-v for v in od.sell_orders.values()) if od.sell_orders else 0
        has_bids = total_bid_vol >= self.MIN_LIQUIDITY
        has_asks = total_ask_vol >= self.MIN_LIQUIDITY

        spike = False
        if self.prev_best_bid is not None and best_bid is not None:
            if best_bid - self.prev_best_bid > self.SPIKE_THRESHOLD:
                spike = True
        if self.prev_best_ask is not None and best_ask is not None:
            if self.prev_best_ask - best_ask > self.SPIKE_THRESHOLD:
                spike = True

        if best_bid is not None:
            self.prev_best_bid = best_bid
        if best_ask is not None:
            self.prev_best_ask = best_ask

        if spike:
            if od.sell_orders:
                for ask_price in sorted(od.sell_orders.keys()):
                    if ask_price <= fair:
                        qty = min(-od.sell_orders[ask_price], lim - pos)
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            pos += qty
            if od.buy_orders:
                for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_price >= fair:
                        qty = min(od.buy_orders[bid_price], lim + pos)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            pos -= qty
            return orders

        consumed_ask = {}
        consumed_bid = {}

        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair - 1:
                    available = -od.sell_orders[ask_price] - consumed_ask.get(ask_price, 0)
                    qty = min(available, lim - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        pos += qty
                        consumed_ask[ask_price] = consumed_ask.get(ask_price, 0) + qty

        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair + 1:
                    available = od.buy_orders[bid_price] - consumed_bid.get(bid_price, 0)
                    qty = min(available, lim + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        pos -= qty
                        consumed_bid[bid_price] = consumed_bid.get(bid_price, 0) + qty

        if mid is not None:
            dev = mid - fair
            if abs(dev) >= self.FADE_TRIGGER:
                target_pos = int(-dev * self.FADE_GAIN)
                target_pos = max(-lim, min(lim, target_pos))

                if dev >= self.FADE_TRIGGER and pos > target_pos:
                    need = pos - target_pos
                    for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                        if need <= 0:
                            break
                        if bid_price >= fair:
                            available = (
                                od.buy_orders[bid_price]
                                - consumed_bid.get(bid_price, 0)
                            )
                            qty = min(available, lim + pos, need)
                            if qty > 0:
                                orders.append(Order(product, bid_price, -qty))
                                pos -= qty
                                consumed_bid[bid_price] = (
                                    consumed_bid.get(bid_price, 0) + qty
                                )
                                need -= qty
                elif dev <= -self.FADE_TRIGGER and pos < target_pos:
                    need = target_pos - pos
                    for ask_price in sorted(od.sell_orders.keys()):
                        if need <= 0:
                            break
                        if ask_price <= fair:
                            available = (
                                -od.sell_orders[ask_price]
                                - consumed_ask.get(ask_price, 0)
                            )
                            qty = min(available, lim - pos, need)
                            if qty > 0:
                                orders.append(Order(product, ask_price, qty))
                                pos += qty
                                consumed_ask[ask_price] = (
                                    consumed_ask.get(ask_price, 0) + qty
                                )
                                need -= qty

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