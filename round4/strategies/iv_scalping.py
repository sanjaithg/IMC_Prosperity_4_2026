"""
IV Scalping Algorithm for Prosperity 4 Round 4

Leverages counterparty information to identify and execute delta-hedged vega scalps.
Strategy:
- Monitor Mark 01/22 counterparty activity for market timing signals
- Identify scalping opportunities where IV spread > price spread
- Execute delta-hedged vega scalps in voucher options
- Use mean reversion patterns for exits
"""

from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import List, Dict, Tuple, Optional
import json
from math import exp, sqrt, log, pi

class Trader:
    """
    IV Scalping Strategy for Prosperity 4 Round 4.
    """
    
    # Position limits per product (from Round 4 spec)
    POSITION_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300, "VEV_5100": 300,
        "VEV_5200": 300, "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }
    
    # Voucher structure: (strike, option_type)
    VOUCHERS = {
        "VEV_4000": (4000, "call"),
        "VEV_4500": (4500, "call"),
        "VEV_5000": (5000, "call"),
        "VEV_5100": (5100, "call"),
        "VEV_5200": (5200, "call"),
        "VEV_5300": (5300, "call"),
        "VEV_5400": (5400, "call"),
        "VEV_5500": (5500, "call"),
        "VEV_6000": (6000, "call"),
        "VEV_6500": (6500, "call"),
    }
    
    # Time to expiry schedule (days)
    TTE_SCHEDULE = {1: 4, 2: 3, 3: 2}
    
    # Mark counterparty IDs
    MARK_IDS = {"Mark 01", "Mark 22"}
    
    # Base spread thresholds for scalping
    TIGHT_SPREAD_THRESHOLD = 2  # Price spread <= 2 is considered tight
    MIN_IV_SPREAD = 0.01  # Minimum IV spread (1%) to consider scalp
    
    def __init__(self):
        self.mark_activity_history = []
        self.position_history = {}
        
    def bid(self):
        """Required for Round 2 compatibility (ignored in Round 4)."""
        return 15
    
    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Simple approximation of standard normal CDF."""
        return (1 + 0.196854 * abs(x) + 0.115194 * abs(x)**2 + 
                0.000344 * abs(x)**3 + 0.019527 * abs(x)**4) ** -4 if x >= 0 else \
               1 - (1 + 0.196854 * abs(x) + 0.115194 * abs(x)**2 + 
                    0.000344 * abs(x)**3 + 0.019527 * abs(x)**4) ** -4
    
    @staticmethod
    def _norm_pdf(x: float) -> float:
        """Standard normal PDF."""
        return (1 / sqrt(2 * pi)) * exp(-0.5 * x * x)
    
    def _black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes call price."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        try:
            d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            call = S * self._norm_cdf(d1) - K * exp(-r * T) * self._norm_cdf(d2)
            return max(call, 0)
        except:
            return max(S - K, 0)
    
    def _black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Black-Scholes put price."""
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        try:
            d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
            put = K * exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)
            return max(put, 0)
        except:
            return max(K - S, 0)
    
    def _implied_vol_newton(self, market_price: float, S: float, K: float, T: float, 
                            option_type: str = "call", r: float = 0.0) -> Optional[float]:
        """Estimate implied volatility using Newton-Raphson method."""
        if T <= 0 or market_price <= 0:
            return 0.25
        
        # Intrinsic value check
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        if market_price < intrinsic * 0.99:
            return None
        
        # Newton-Raphson iteration
        sigma = 0.5
        for iteration in range(15):
            if option_type == "call":
                price = self._black_scholes_call(S, K, T, r, sigma)
            else:
                price = self._black_scholes_put(S, K, T, r, sigma)
            
            diff = price - market_price
            if abs(diff) < 1e-6:
                return max(sigma, 0.001)
            
            # Vega for derivative
            try:
                d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
                vega = S * self._norm_pdf(d1) * sqrt(T)
                
                if abs(vega) < 1e-8:
                    return max(sigma, 0.001)
                
                sigma = max(0.0001, sigma - diff / vega)
            except:
                return max(sigma, 0.001)
        
        return max(sigma, 0.001)
    
    def _get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculate mid price from order book."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2.0
    
    def _get_spread(self, order_depth: OrderDepth) -> Tuple[int, int, int]:
        """Get spread and best bid/ask volumes."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0, 0, 0
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = -order_depth.sell_orders[best_ask]
        
        return best_ask - best_bid, bid_vol, ask_vol
    
    def _detect_mark_activity(self, market_trades: Dict[str, List[Trade]]) -> Dict[str, int]:
        """Detect Mark counterparty activity."""
        mark_activity = {}
        
        for product, trades in market_trades.items():
            mark_volume = 0
            
            for trade in trades:
                if trade.buyer in self.MARK_IDS or trade.seller in self.MARK_IDS:
                    mark_volume += trade.quantity
            
            if mark_volume > 0:
                mark_activity[product] = mark_volume
        
        return mark_activity
    
    def _calculate_available_position(self, product: str, current_position: int, side: str) -> int:
        """Calculate how much more we can trade on a given side."""
        limit = self.POSITION_LIMITS.get(product, 100)
        
        if side == "buy":
            # Can go up to +limit
            available = limit - current_position
        else:  # sell
            # Can go down to -limit
            available = limit + current_position
        
        return max(0, available)
    
    def run(self, state: TradingState):
        """
        Main IV scalping trading logic.
        
        Returns:
            result: Dict[product] -> List[Orders]
            conversions: 0
            traderData: Serialized state
        """
        
        result = {}
        
        # Detect Mark activity
        mark_activity = self._detect_mark_activity(state.market_trades)
        
        # Infer day from timestamp
        day = 1
        if state.timestamp > 500000:
            day = 2
        if state.timestamp > 1000000:
            day = 3
        
        # Get TTE for this day
        tte_days = self.TTE_SCHEDULE.get(day, 2)
        tte_years = tte_days / 365.0
        r = 0.0
        
        # Process each product
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []
            
            current_position = state.position.get(product, 0)
            spread, bid_vol, ask_vol = self._get_spread(order_depth)
            
            # Base products: simple mean reversion on tight spreads
            if product in ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]:
                mid_price = self._get_mid_price(order_depth)
                
                if mid_price and spread > 0 and spread <= self.TIGHT_SPREAD_THRESHOLD:
                    # Buy on weak asks
                    if len(order_depth.sell_orders) > 0:
                        best_ask = min(order_depth.sell_orders.keys())
                        ask_qty = -order_depth.sell_orders[best_ask]
                        available = self._calculate_available_position(product, current_position, "buy")
                        
                        buy_qty = min(ask_qty, available)
                        if buy_qty > 0:
                            orders.append(Order(product, best_ask, buy_qty))
                    
                    # Sell on strong bids
                    if len(order_depth.buy_orders) > 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        bid_qty = order_depth.buy_orders[best_bid]
                        available = self._calculate_available_position(product, current_position, "sell")
                        
                        sell_qty = min(bid_qty, available)
                        if sell_qty > 0:
                            orders.append(Order(product, best_bid, -sell_qty))
            
            # Vouchers: IV scalping strategy
            elif product in self.VOUCHERS:
                strike, option_type = self.VOUCHERS[product]
                
                # Get spot from underlying
                underlying_od = state.order_depths.get("VELVETFRUIT_EXTRACT", OrderDepth())
                spot = self._get_mid_price(underlying_od)
                
                if spot and len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    bid_qty = order_depth.buy_orders[best_bid]
                    ask_qty = -order_depth.sell_orders[best_ask]
                    mid_price = (best_bid + best_ask) / 2.0
                    price_spread = best_ask - best_bid
                    
                    # Compute IVs
                    iv_bid = self._implied_vol_newton(best_bid, spot, strike, tte_years, option_type)
                    iv_ask = self._implied_vol_newton(best_ask, spot, strike, tte_years, option_type)
                    
                    if iv_bid and iv_ask:
                        iv_spread = iv_ask - iv_bid
                        
                        # IV Scalp Signal
                        if price_spread > 0 and iv_spread > self.MIN_IV_SPREAD:
                            
                            # Mark activity amplifies signal
                            mark_multiplier = 1.5 if product in mark_activity else 1.0
                            
                            # Buy (go long vol)
                            available_buy = self._calculate_available_position(product, current_position, "buy")
                            if available_buy > 0:
                                buy_qty = int(min(ask_qty, available_buy) * mark_multiplier * 0.5)
                                if buy_qty > 0:
                                    orders.append(Order(product, best_ask, buy_qty))
                            
                            # Sell at better price (establish short position)
                            available_sell = self._calculate_available_position(product, current_position, "sell")
                            if available_sell > 0:
                                sell_price = int(mid_price) + 1
                                if sell_price <= best_ask:
                                    sell_qty = int(min(bid_qty // 2, available_sell) * 0.5)
                                    if sell_qty > 0:
                                        orders.append(Order(product, sell_price, -sell_qty))
                        
                        # Exit on IV compression
                        elif price_spread <= 1 or iv_spread < 0.001:
                            if current_position > 0:
                                available_sell = self._calculate_available_position(product, current_position, "sell")
                                exit_qty = min(current_position, available_sell)
                                if exit_qty > 0:
                                    orders.append(Order(product, best_bid, -exit_qty))
                            
                            elif current_position < 0:
                                available_buy = self._calculate_available_position(product, current_position, "buy")
                                exit_qty = min(abs(current_position), available_buy)
                                if exit_qty > 0:
                                    orders.append(Order(product, best_ask, exit_qty))
            
            result[product] = orders
        
        # Serialize state
        trader_state = {
            "timestamp": state.timestamp,
            "day": day,
            "mark_activity_count": len(mark_activity),
        }
        traderData = json.dumps(trader_state)
        
        return result, 0, traderData
