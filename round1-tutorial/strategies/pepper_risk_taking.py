from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import jsonpickle

class Trader:

    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    # Osmium Parameters (unchanged)
    FAIR_VALUE = 10000
    TAKE_BUY_BELOW = 9999
    TAKE_SELL_ABOVE = 10001
    MIN_SPREAD = 2
    FALLBACK_BID = {"ASH_COATED_OSMIUM": 9910}
    FALLBACK_ASK = {"ASH_COATED_OSMIUM": 10090}
    SPIKE_THRESHOLD = 6
    MIN_LIQUIDITY = 5
    MM_SIZE = 20

    # Pepper Parameters
    PEPPER_BASE = 10990
    PEPPER_CORE = 65           # 🔒 SACRED - never touched
    PEPPER_FLOAT = 10          # Used for MM and spikes
    PEPPER_MM_SIZE = 5         # Size for market making
    PEPPER_SPIKE_SIZE = 5     # Size for spike taking

    prev_best_bid = None
    prev_best_ask = None

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        timestamp = state.timestamp
        
        if state.traderData:
            try:
                memory = jsonpickle.decode(state.traderData)
                self.prev_best_bid = memory.get("prev_bid")
                self.prev_best_ask = memory.get("prev_ask")
            except:
                pass

        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            lim = self.POSITION_LIMITS.get(product, 0)

            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_pepper_sacred_core(od, pos, lim, timestamp)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(od, pos, lim)
            else:
                result[product] = []

        memory = {"prev_bid": self.prev_best_bid, "prev_ask": self.prev_best_ask}
        return result, 0, jsonpickle.encode(memory)

    # ──────────────────────────────────────────────────────────
    # PEPPER ROOT — Sacred Core (65) + Trading Float (15)
    # ──────────────────────────────────────────────────────────
    def trade_pepper_sacred_core(self, od: OrderDepth, pos: int, lim: int, timestamp: int) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        orders: List[Order] = []
        
        if not od.buy_orders or not od.sell_orders:
            return orders
            
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        
        # Dynamic fair value
        fair_value = self.PEPPER_BASE + (timestamp / 10.0)
        
        # ──────────────────────────────────────────────────────
        # PHASE 1: Build Sacred Core (pos < 65)
        # ──────────────────────────────────────────────────────
        if pos < self.PEPPER_CORE:
            # Aggressively accumulate until core is full
            buy_qty = self.PEPPER_CORE - pos
            
            for ask_price in sorted(od.sell_orders.keys()):
                # Don't grossly overpay (> fair + 5)
                if ask_price > fair_value + 5:
                    break
                    
                qty = min(-od.sell_orders[ask_price], buy_qty)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_qty -= qty
                
                if buy_qty <= 0:
                    break
            
            # Passive bid for remaining
            if buy_qty > 0:
                my_bid = min(best_bid + 1, int(fair_value))
                if my_bid < best_ask:
                    orders.append(Order(product, my_bid, min(buy_qty, 10)))
            
            return orders
        
        # ──────────────────────────────────────────────────────
        # PHASE 2: Core is Full - Trade Only the Float
        # ──────────────────────────────────────────────────────
        
        # Available float capacity
        max_buy = self.PEPPER_CORE + self.PEPPER_FLOAT - pos   # Can buy up to 80
        max_sell = pos - (self.PEPPER_CORE - self.PEPPER_FLOAT)  # Can sell down to 50
        
        # Spike detection
        spike = False
        if not hasattr(self, 'prev_pepper_mid'):
            self.prev_pepper_mid = None
        if self.prev_pepper_mid is not None:
            if abs(mid - self.prev_pepper_mid) > 6:
                spike = True
        self.prev_pepper_mid = mid
        
        # ──────────────────────────────────────────────────────
        # SPIKE MODE: Aggressive taking with float only
        # ──────────────────────────────────────────────────────
        if spike:
            # BUY SPIKE: Ask dropped significantly (undervalued)
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < fair_value - 4:
                    qty = min(-od.sell_orders[ask_price], max_buy, self.PEPPER_SPIKE_SIZE)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        max_buy -= qty
                        # IMMEDIATELY prepare to sell to revert to core
                        # (Will happen naturally via market making or next spike)
            
            # SELL SPIKE: Bid spiked significantly (overvalued)
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > fair_value + 4:
                    qty = min(od.buy_orders[bid_price], max_sell, self.PEPPER_SPIKE_SIZE)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        max_sell -= qty
            
            # After spike, return to normal (don't market make this iteration)
            return orders
        
        # ──────────────────────────────────────────────────────
        # NORMAL MODE: Market Making with Float Only
        # ──────────────────────────────────────────────────────
        
        # Ideal quotes centered on fair value
        ideal_bid = int(fair_value - 2)
        ideal_ask = int(fair_value + 2)
        
        my_bid = min(ideal_bid, best_ask - 1)
        my_ask = max(ideal_ask, best_bid + 1)
        
        # Inventory skew: want to stay near core (65)
        # If pos > 65: bias toward selling (lower bid, lower ask)
        # If pos < 65: bias toward buying (raise bid, raise ask)
        if pos > self.PEPPER_CORE:
            skew = (pos - self.PEPPER_CORE) // 3
            my_bid = max(my_bid - skew, best_bid - 3)
            my_ask = max(my_ask - skew, best_bid + 1)
        elif pos < self.PEPPER_CORE:
            skew = (self.PEPPER_CORE - pos) // 3
            my_bid = min(my_bid + skew, best_ask - 1)
            my_ask = min(my_ask + skew, best_ask + 3)
        
        # Place market making orders (only within float limits)
        buy_qty = min(self.PEPPER_MM_SIZE, max_buy)
        if buy_qty > 0:
            orders.append(Order(product, my_bid, buy_qty))
        
        sell_qty = min(self.PEPPER_MM_SIZE, max_sell)
        if sell_qty > 0:
            orders.append(Order(product, my_ask, -sell_qty))
        
        # ──────────────────────────────────────────────────────
        # OPPORTUNISTIC TAKING: Small nibbles at good prices
        # ──────────────────────────────────────────────────────
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < fair_value - 3 and max_buy > 0:
                qty = min(-od.sell_orders[ask_price], max_buy, 3)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    break
                    
        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > fair_value + 3 and max_sell > 0:
                qty = min(od.buy_orders[bid_price], max_sell, 3)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    break
        
        return orders

    # ──────────────────────────────────────────────────────────
    # OSMIUM — Unchanged (your existing optimal strategy)
    # ──────────────────────────────────────────────────────────
    def trade_osmium(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        orders: List[Order] = []

        if not od.buy_orders and not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

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
                    if ask_price <= self.FAIR_VALUE:
                        qty = min(-od.sell_orders[ask_price], lim - pos)
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            pos += qty
            if od.buy_orders:
                for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_price >= self.FAIR_VALUE:
                        qty = min(od.buy_orders[bid_price], lim + pos)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            pos -= qty
            return orders

        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < self.TAKE_BUY_BELOW:
                    qty = min(-od.sell_orders[ask_price], lim - pos)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        pos += qty

        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > self.TAKE_SELL_ABOVE:
                    qty = min(od.buy_orders[bid_price], lim + pos)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        pos -= qty

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