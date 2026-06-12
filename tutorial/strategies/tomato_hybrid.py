from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import math

class Trader:
    """
    Strategy 4: Tomato Hybrid (Active Fair Value + Passive Penny Jumping)
    This bot uses EMA for baseline fair value and attempts to outbid the market 
    (penny jumping) to get filled on passive orders, while keeping an active
    'taking' layer for extreme prices.
    """

    def run(self, state: TradingState):
        result = {}
        
        # --- Parameters ---
        POSITION_LIMIT = 80
        EMA_SPAN = 20
        ACTIVE_THRESHOLD = 8.0  # Deviation to trigger market takes
        PASSIVE_OFFSET = 1.0    # Offset from mid to ensure profit on limit orders
        
        # --- Persistent State (traderData) ---
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except:
            data = {}
            
        last_ema = data.get("TOMATOES_EMA", None)

        for product in state.order_depths:
            if product == 'TOMATOES':
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []
                
                # 1. Update Fair Value (EMA)
                if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    mid_price = (best_bid + best_ask) / 2
                else:
                    continue

                alpha = 2 / (EMA_SPAN + 1)
                if last_ema is None:
                    fair_value = mid_price
                else:
                    fair_value = (mid_price * alpha) + (last_ema * (1 - alpha))
                
                last_ema = fair_value
                current_pos = state.position.get(product, 0)
                
                # --- 2. ACTIVE LAYER (Market Taking) ---
                
                # Aggressive Buy: Hit sells far below fair value
                buy_take_threshold = fair_value - ACTIVE_THRESHOLD
                for price, quantity in sorted(order_depth.sell_orders.items()):
                    if price <= buy_take_threshold:
                        buy_qty = min(-quantity, POSITION_LIMIT - current_pos)
                        if buy_qty > 0:
                            orders.append(Order(product, price, buy_qty))
                            current_pos += buy_qty
                
                # Aggressive Sell: Hit bids far above fair value
                sell_take_threshold = fair_value + ACTIVE_THRESHOLD
                for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                    if price >= sell_take_threshold:
                        sell_qty = max(-quantity, -POSITION_LIMIT - current_pos)
                        if sell_qty < 0:
                            orders.append(Order(product, price, sell_qty))
                            current_pos += sell_qty

                # --- 3. PASSIVE LAYER (Penny Jumping) ---
                
                # We want to outbid the best bid, but stay below fair_value
                if current_pos < POSITION_LIMIT:
                    # Target = BestBid + 1, but capped at (FairValue - Offset)
                    jump_bid = best_bid + 1
                    max_allowed_bid = int(math.floor(fair_value - PASSIVE_OFFSET))
                    
                    target_bid = min(jump_bid, max_allowed_bid)
                    
                    # Ensure we aren't crossing our own logic (bid must be reasonable)
                    if target_bid >= best_ask: # Safety
                        target_bid = best_ask - 1
                        
                    limit_buy_qty = POSITION_LIMIT - current_pos
                    if limit_buy_qty > 0:
                        orders.append(Order(product, target_bid, limit_buy_qty))
                    
                # We want to undercut the best ask, but stay above fair_value
                if current_pos > -POSITION_LIMIT:
                    # Target = BestAsk - 1, but floored at (FairValue + Offset)
                    jump_ask = best_ask - 1
                    min_allowed_ask = int(math.ceil(fair_value + PASSIVE_OFFSET))
                    
                    target_ask = max(jump_ask, min_allowed_ask)
                    
                    if target_ask <= best_bid: # Safety
                        target_ask = best_bid + 1
                        
                    limit_sell_qty = -POSITION_LIMIT - current_pos
                    if limit_sell_qty < 0:
                        orders.append(Order(product, target_ask, limit_sell_qty))

                result[product] = orders
        
        # Serialize state
        new_data = {"TOMATOES_EMA": last_ema}
        traderData = json.dumps(new_data)
        
        conversions = 0
        return result, conversions, traderData
