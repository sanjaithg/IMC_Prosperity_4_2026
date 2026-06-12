from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    """
    Strategy 1: Emerald Market Making
    EMERALDS are statistically mean-reverting around 10,000.
    This bot places orders at the edges of the spread to harvest profit.
    """

    def run(self, state: TradingState):
        result = {}
        
        # Emerald Position Limit is now 80 as per latest user info
        POSITION_LIMIT = 80
        FAIR_VALUE = 10000
        
        # Strat 1 Parameters
        BUY_PRICE = 9993
        SELL_PRICE = 10007

        for product in state.order_depths:
            if product == 'EMERALDS':
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []
                
                current_pos = state.position.get(product, 0)
                
                # --- MARKET TAKING (Aggressive) ---
                # Check if anyone is selling below our buy price
                if len(order_depth.sell_orders) != 0:
                    for price, quantity in sorted(order_depth.sell_orders.items()):
                        if price <= BUY_PRICE:
                            # We want to buy as much as possible up to limit
                            buy_qty = min(-quantity, POSITION_LIMIT - current_pos)
                            if buy_qty > 0:
                                orders.append(Order(product, price, buy_qty))
                                current_pos += buy_qty
                
                # Check if anyone is buying above our sell price
                if len(order_depth.buy_orders) != 0:
                    for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                        if price >= SELL_PRICE:
                            # We want to sell as much as possible up to limit
                            sell_qty = max(-quantity, -POSITION_LIMIT - current_pos)
                            if sell_qty < 0:
                                orders.append(Order(product, price, sell_qty))
                                current_pos += sell_qty

                # --- MARKET MAKING (Passive) ---
                # If we still have room in our position, place limit orders
                # Note: These might not fill in the current iteration
                
                # Passive Buy
                if current_pos < POSITION_LIMIT:
                    limit_buy_qty = POSITION_LIMIT - current_pos
                    orders.append(Order(product, BUY_PRICE, limit_buy_qty))
                    
                # Passive Sell
                if current_pos > -POSITION_LIMIT:
                    limit_sell_qty = -POSITION_LIMIT - current_pos
                    orders.append(Order(product, SELL_PRICE, limit_sell_qty))

                result[product] = orders
        
        traderData = ""
        conversions = 0
        return result, conversions, traderData
