from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import jsonpickle

class Trader:
    """
    PEPPER ROOT: Fair Value MM (Fair = Base + Timestamp/10)
    OSMIUM: Top-of-book MM + market taking around fair value 10,000
    """

    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    MIN_SPREAD = 2
    
    # Osmium Parameters
    OSMIUM_FAIR = 10000
    TAKE_BUY_BELOW = 9999
    TAKE_SELL_ABOVE = 10001
    FALLBACK_BID = {"ASH_COATED_OSMIUM": 9910}
    FALLBACK_ASK = {"ASH_COATED_OSMIUM": 10090}
    
    # Pepper Parameters
    PEPPER_BASE = 10990  # Fair = Base + timestamp/10
    
    # Pepper Inventory Management
    PEPPER_TARGET_POSITION = 40  # Aim for neutral-ish position
    PEPPER_MM_SPREAD = 4         # Quote width around fair value

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        timestamp = state.timestamp
        
        # Load memory for tracking Pepper fair value drift
        if state.traderData:
            try:
                memory = jsonpickle.decode(state.traderData)
            except:
                memory = {"pepper_trades": []}
        else:
            memory = {"pepper_trades": []}

        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            lim = self.POSITION_LIMITS.get(product, 0)

            if lim == 0:
                result[product] = []
                continue

            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_pepper_mm(od, pos, lim, timestamp)
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_osmium(od, pos, lim)
            else:
                result[product] = []

        return result, 0, jsonpickle.encode(memory)

    # ──────────────────────────────────────────────────────────
    # PEPPER ROOT — Fair Value Market Making
    # Fair Price = Base + Timestamp/10
    # ──────────────────────────────────────────────────────────
    def trade_pepper_mm(self, od: OrderDepth, pos: int, lim: int, timestamp: int) -> List[Order]:
        product = "INTARIAN_PEPPER_ROOT"
        orders: List[Order] = []
        
        if not od.buy_orders or not od.sell_orders:
            return orders
            
        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        
        # Calculate dynamic fair value based on timestamp drift
        fair_value = self.PEPPER_BASE + (timestamp / 10.0)
        
        # --- LAYER 1: Aggressive Taking (When market is mispriced) ---
        # Buy if ask is significantly below fair value (undervalued)
        for ask_price in sorted(od.sell_orders.keys()):
            if ask_price < fair_value - 5:  # At least 5 ticks undervalued
                ask_vol = -od.sell_orders[ask_price]
                can_buy = lim - pos
                qty = min(ask_vol, can_buy)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    pos += qty
                    
        # Sell if bid is significantly above fair value (overvalued)
        for bid_price in sorted(od.buy_orders.keys(), reverse=True):
            if bid_price > fair_value + 5:  # At least 5 ticks overvalued
                bid_vol = od.buy_orders[bid_price]
                can_sell = lim + pos
                qty = min(bid_vol, can_sell)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    pos -= qty

        # --- LAYER 2: Passive Market Making Around Fair Value ---
        # Calculate ideal bid/ask centered on fair value
        ideal_bid = int(fair_value - self.PEPPER_MM_SPREAD // 2)
        ideal_ask = int(fair_value + self.PEPPER_MM_SPREAD // 2)
        
        # Don't cross the market - be passive
        my_bid = min(ideal_bid, best_ask - 1)
        my_ask = max(ideal_ask, best_bid + 1)
        
        # Adjust quotes based on inventory skew
        # If long, lower bid and lower ask to encourage selling
        if pos > self.PEPPER_TARGET_POSITION:
            inventory_skew = (pos - self.PEPPER_TARGET_POSITION) // 5
            my_bid = max(my_bid - inventory_skew, best_bid - 5)
            my_ask = max(my_ask - inventory_skew, best_bid + 1)
        
        # If short, raise bid and raise ask to encourage buying
        elif pos < -self.PEPPER_TARGET_POSITION:
            inventory_skew = (-pos - self.PEPPER_TARGET_POSITION) // 5
            my_bid = min(my_bid + inventory_skew, best_ask - 1)
            my_ask = min(my_ask + inventory_skew, best_ask + 5)
        
        # Place orders if within position limits
        buy_qty = min(10, lim - pos)
        if buy_qty > 0 and my_bid > 0:
            orders.append(Order(product, my_bid, buy_qty))
            
        sell_qty = min(10, lim + pos)
        if sell_qty > 0:
            orders.append(Order(product, my_ask, -sell_qty))

        return orders

    # ──────────────────────────────────────────────────────────
    # OSMIUM — MM + Taking around fair value 10,000 (UNCHANGED)
    # ──────────────────────────────────────────────────────────
    def trade_osmium(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        product = "ASH_COATED_OSMIUM"
        orders: List[Order] = []

        has_bids = bool(od.buy_orders)
        has_asks = bool(od.sell_orders)

        # ── TAKE: Buy cheap asks, sell to expensive bids ──
        if has_asks:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < self.TAKE_BUY_BELOW:
                    ask_vol = -od.sell_orders[ask_price]
                    can_buy = lim - pos
                    qty = min(ask_vol, can_buy)
                    if qty > 0:
                        orders.append(Order(product, ask_price, qty))
                        pos += qty

        if has_bids:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > self.TAKE_SELL_ABOVE:
                    bid_vol = od.buy_orders[bid_price]
                    can_sell = lim + pos
                    qty = min(bid_vol, can_sell)
                    if qty > 0:
                        orders.append(Order(product, bid_price, -qty))
                        pos -= qty

        # ── MAKE: Top-of-book MM with greedy fallbacks ──
        if has_bids and has_asks:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())

            if best_ask - best_bid > self.MIN_SPREAD:
                buy_qty = lim - pos
                if buy_qty > 0:
                    orders.append(Order(product, best_bid + 1, buy_qty))

                sell_qty = lim + pos
                if sell_qty > 0:
                    orders.append(Order(product, best_ask - 1, -sell_qty))

        elif has_bids and not has_asks:
            buy_qty = lim - pos
            if buy_qty > 0:
                orders.append(Order(product, max(od.buy_orders.keys()) + 1, buy_qty))
            sell_qty = lim + pos
            if sell_qty > 0:
                orders.append(Order(product, self.FALLBACK_ASK[product], -sell_qty))

        elif has_asks and not has_bids:
            buy_qty = lim - pos
            if buy_qty > 0:
                orders.append(Order(product, self.FALLBACK_BID[product], buy_qty))
            sell_qty = lim + pos
            if sell_qty > 0:
                orders.append(Order(product, min(od.sell_orders.keys()) - 1, -sell_qty))

        else:
            buy_qty = lim - pos
            if buy_qty > 0:
                orders.append(Order(product, self.FALLBACK_BID[product], buy_qty))
            sell_qty = lim + pos
            if sell_qty > 0:
                orders.append(Order(product, self.FALLBACK_ASK[product], -sell_qty))

        return orders
