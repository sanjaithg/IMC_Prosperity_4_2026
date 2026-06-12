from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import jsonpickle

class Trader:
    """
    PEPPER ROOT: Buy & hold (unchanged)
    OSMIUM: SMA(20) Mean Reversion + Wall Mid MM
    """

    POSITION_LIMITS = {
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    MIN_SPREAD = 2
    FALLBACK_BID = {"ASH_COATED_OSMIUM": 9910}
    FALLBACK_ASK = {"ASH_COATED_OSMIUM": 10090}
    
    # SMA Parameters
    SMA_WINDOW = 20
    
    # Accumulation Zones (SMA thresholds)
    ZONE_AGGRESSIVE_BUY = 9990      # SMA < 9990 -> Buy 2x
    ZONE_SLOW_BUY = 9995            # SMA < 9995 -> Buy 1x
    ZONE_SLOW_SELL = 10005          # SMA > 10005 -> Sell 1x
    ZONE_AGGRESSIVE_SELL = 10010    # SMA > 10010 -> Sell 2x

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        
        # Load memory
        if state.traderData:
            try:
                memory = jsonpickle.decode(state.traderData)
            except:
                memory = {"osmium_sma_history": []}
        else:
            memory = {"osmium_sma_history": []}

        for product in state.order_depths:
            od = state.order_depths[product]
            pos = state.position.get(product, 0)
            lim = self.POSITION_LIMITS.get(product, 0)

            if lim == 0:
                result[product] = []
                continue

            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_ipr(od, pos, lim)
            elif product == "ASH_COATED_OSMIUM":
                result[product], memory = self.trade_osmium_sma(od, pos, lim, memory, product)
            else:
                result[product] = []

        return result, 0, jsonpickle.encode(memory)

    # ──────────────────────────────────────────────────────────
    # PEPPER ROOT — Buy & hold (UNCHANGED)
    # ──────────────────────────────────────────────────────────
    def trade_ipr(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        orders: List[Order] = []
        buy_qty = lim - pos

        if buy_qty <= 0:
            return orders

        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                can_buy = min(-od.sell_orders[ask_price], buy_qty)
                if can_buy > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ask_price, can_buy))
                    buy_qty -= can_buy
                if buy_qty <= 0:
                    break

        if buy_qty > 0 and od.buy_orders:
            best_bid = max(od.buy_orders.keys())
            orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid + 1, buy_qty))

        return orders

    # ──────────────────────────────────────────────────────────
    # OSMIUM — SMA(20) Mean Reversion Gradient
    # ──────────────────────────────────────────────────────────
    def trade_osmium_sma(self, od: OrderDepth, pos: int, lim: int, memory: Dict, product: str) -> Tuple[List[Order], Dict]:
        orders: List[Order] = []
        
        if not od.buy_orders or not od.sell_orders:
            # Empty book - use fallbacks
            buy_qty = lim - pos
            if buy_qty > 0:
                orders.append(Order(product, self.FALLBACK_BID[product], buy_qty))
            sell_qty = lim + pos
            if sell_qty > 0:
                orders.append(Order(product, self.FALLBACK_ASK[product], -sell_qty))
            return orders, memory

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2.0
        
        # --- Update SMA History (exclude NaN/None) ---
        history = memory.get("osmium_sma_history", [])
        if mid_price is not None and mid_price > 0:
            history.append(mid_price)
        if len(history) > self.SMA_WINDOW:
            history.pop(0)
        memory["osmium_sma_history"] = history
        
        # --- Calculate SMA ---
        if len(history) >= 5:  # Need minimum data for reliable SMA
            sma = sum(history) / len(history)
        else:
            sma = 10000.0  # Default until we have enough data
        
        # --- Determine Accumulation/De-accumulation Multiplier ---
        buy_multiplier = 0
        sell_multiplier = 0
        
        if sma < self.ZONE_AGGRESSIVE_BUY:
            buy_multiplier = 2   # Aggressive accumulation
        elif sma < self.ZONE_SLOW_BUY:
            buy_multiplier = 1   # Slow accumulation
        elif sma > self.ZONE_AGGRESSIVE_SELL:
            sell_multiplier = 2  # Aggressive de-accumulation
        elif sma > self.ZONE_SLOW_SELL:
            sell_multiplier = 1  # Slow de-accumulation
            
        # --- Layer 1: Directional Taking based on SMA Zones ---
        base_qty = 10  # Base order size
        
        # BUYING: Take asks when we want to accumulate
        if buy_multiplier > 0 and od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                ask_vol = -od.sell_orders[ask_price]
                can_buy = lim - pos
                # Apply multiplier to buying aggressiveness
                target_qty = min(base_qty * buy_multiplier, can_buy)
                qty = min(ask_vol, target_qty)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    pos += qty
                    break  # Only take one level per iteration to avoid over-aggression
                    
        # SELLING: Hit bids when we want to de-accumulate
        if sell_multiplier > 0 and od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                bid_vol = od.buy_orders[bid_price]
                can_sell = lim + pos
                target_qty = min(base_qty * sell_multiplier, can_sell)
                qty = min(bid_vol, target_qty)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    pos -= qty
                    break

        # --- Layer 2: Greedy Wall Mid Market Making (Always On) ---
        if best_ask - best_bid > self.MIN_SPREAD:
            buy_qty = lim - pos
            if buy_qty > 0:
                # If we're in accumulation zone, bid more aggressively
                bid_price = best_bid + 1
                if buy_multiplier > 0:
                    bid_price = best_bid + 2  # More aggressive bid when accumulating
                orders.append(Order(product, bid_price, min(buy_qty, 15)))

            sell_qty = lim + pos
            if sell_qty > 0:
                # If we're in de-accumulation zone, ask more aggressively
                ask_price = best_ask - 1
                if sell_multiplier > 0:
                    ask_price = best_ask - 2  # More aggressive ask when de-accumulating
                orders.append(Order(product, ask_price, -min(sell_qty, 15)))

        return orders, memory
