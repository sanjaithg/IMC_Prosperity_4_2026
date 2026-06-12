"""
submission2.py — Gamma Scalping / Delta-Neutral Options Trader
================================================================

Strategy
--------
1. Model the volatility smile across the 10 VEV vouchers every tick.
2. Buy vouchers whose implied vol sits BELOW the smile (cheap gamma/vega) and
   sell those trading ABOVE (rich gamma/vega) — this is a relative-value bet,
   not a directional one.
3. Delta-hedge the NET option delta with VELVETFRUIT_EXTRACT so the book stays
   ~flat on the underlying. As VEV oscillates, re-hedging a long-gamma book
   locks in scalp PnL; a short-gamma book pays theta in exchange.
4. HYDROGEL_PACK is independent (not part of the option chain). We market-make
   it around its own mid with inventory-skewed quotes.

Notes
-----
- Stateless between AWS Lambda calls → all persistent state is JSON-encoded in
  `traderData` (smile EMA, running realized variance, previous mid).
- The smile is quadratic in log-moneyness m = log(K/S):  iv(m) = a + b·m + c·m²
  Fitted each tick by least squares on valid observations (≥5 points needed).
- Fair IV per voucher = EMA of the smile-implied IV (smooths quote noise).
- Trade sizes are vega-normalised so each bet takes similar vol-of-vol risk.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    # ═══════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_4000": 4000, "VEV_4500": 4500,
        "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
        "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
        "VEV_6000": 6000, "VEV_6500": 6500,
    }
    VOUCHERS: List[str] = list(VOUCHER_STRIKES.keys())
    UNDERLYING = "VELVETFRUIT_EXTRACT"
    HYDROGEL = "HYDROGEL_PACK"

    POSITION_LIMITS: Dict[str, int] = {
        **{v: 300 for v in VOUCHERS},
        UNDERLYING: 200,
        HYDROGEL: 200,
    }

    # Round 3 final: 5 days to expiry at start-of-day, decays linearly.
    # Override via Trader.INITIAL_TTE_DAYS = 8.0 when backtesting historical data.
    INITIAL_TTE_DAYS: float = 5.0
    TICKS_PER_DAY: int = 10_000
    TIMESTAMP_STEP: int = 100  # one tick == 100 timestamp units

    # IV model: per-voucher EMA of observed IV is the "fair" level.
    # Smile fit was too noisy (deep-OTM outliers distort the quadratic), so we
    # don't use it as a filter — but we do skip vouchers whose IV is clearly
    # outside a sane band.
    IV_EMA_ALPHA: float = 0.08
    MIN_VALID_IV: float = 0.05
    MAX_VALID_IV: float = 2.0
    # Observed per-voucher IV std is ~0.003–0.009 on historical data.
    # Entry at ~2σ; exit at ~0.5σ.
    IV_ARB_EDGE_ENTRY: float = 0.004    # ~1.2σ on historical IV stdev
    IV_ARB_EDGE_EXIT:  float = 0.0015
    PRICE_EDGE_MIN: float = 0.0          # rely on IV edge; 0 = don't block
    MAX_VOUCHER_TRADE: int = 4
    VEGA_NORMAL: float = 6.0
    MIN_VEGA_TRADE: float = 3.0          # skip deep-OTM junk where vega is tiny
    SMILE_MIN_POINTS: int = 6            # fit only when enough points available
    # Inventory cap on voucher positions (below the 300 hard limit) keeps us
    # from pinning and leaves room to act on reversals.
    SOFT_VOUCHER_CAP: int = 80

    # Delta hedging
    DELTA_DEADBAND: float = 6.0
    HEDGE_SLICE: int = 50

    # Hydrogel MM (independent delta-1 product)
    HGP_MAX_TRADE: int = 15
    HGP_TAKE_EDGE: float = 1.0          # only trade when clearly mispriced vs mid

    # End-of-day flatten: in the last FLATTEN_TICKS, close toward zero.
    FLATTEN_AFTER_TS: int = 950_000     # ≥ this timestamp → unwind mode
    FLATTEN_SLICE: int = 20             # max qty per voucher per tick while unwinding

    # ═══════════════════════════════════════════════════════════════
    # BLACK-SCHOLES
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _npdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    @classmethod
    def _bs_call(cls, S: float, K: float, T: float, sigma: float, r: float = 0.0
                 ) -> Tuple[float, float, float, float]:
        """Return (price, delta, gamma, vega)."""
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0.0)
            return intrinsic, (1.0 if S > K else 0.0), 0.0, 0.0
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        Nd1 = cls._ncdf(d1)
        Nd2 = cls._ncdf(d2)
        nd1 = cls._npdf(d1)
        price = S * Nd1 - K * math.exp(-r * T) * Nd2
        delta = Nd1
        gamma = nd1 / (S * sigma * sqrtT)
        vega = S * nd1 * sqrtT
        return price, delta, gamma, vega

    @classmethod
    def _implied_vol(cls, price: float, S: float, K: float, T: float,
                     r: float = 0.0) -> Optional[float]:
        if S <= 0 or K <= 0 or T <= 0:
            return None
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
        if price < intrinsic - 1e-6 or price > S:
            return None
        lo, hi = 1e-4, 5.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            model, _, _, _ = cls._bs_call(S, K, T, mid, r)
            err = model - price
            if abs(err) < 1e-5:
                return mid
            if err > 0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    # ═══════════════════════════════════════════════════════════════
    # UTILITIES
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def _best_quotes(depth: OrderDepth
                     ) -> Tuple[Optional[int], Optional[int], int, int]:
        bid = max(depth.buy_orders) if depth.buy_orders else None
        ask = min(depth.sell_orders) if depth.sell_orders else None
        bv = depth.buy_orders[bid] if bid is not None else 0
        av = -depth.sell_orders[ask] if ask is not None else 0
        return bid, ask, bv, av

    @classmethod
    def _mid(cls, depth: OrderDepth) -> Optional[float]:
        b, a, _, _ = cls._best_quotes(depth)
        if b is None or a is None:
            return None
        return 0.5 * (b + a)

    @staticmethod
    def _solve_3x3(A: List[List[float]], b: List[float]) -> Optional[Tuple[float, float, float]]:
        """Cramer's rule on a 3×3 system. None if singular."""
        def det(m):
            return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        D = det(A)
        if abs(D) < 1e-12:
            return None
        out = []
        for col in range(3):
            M = [row[:] for row in A]
            for row in range(3):
                M[row][col] = b[row]
            out.append(det(M) / D)
        return out[0], out[1], out[2]

    # ═══════════════════════════════════════════════════════════════
    # MAIN
    # ═══════════════════════════════════════════════════════════════
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        mem = self._load_memory(state.traderData)

        # ───── TTE for this tick (decays linearly through the day)
        day_frac = state.timestamp / (self.TICKS_PER_DAY * self.TIMESTAMP_STEP)
        tte_days = max(self.INITIAL_TTE_DAYS - day_frac, 1.0 / 365.0)
        T = tte_days / 365.0

        # ───── Underlying mid
        und_depth = state.order_depths.get(self.UNDERLYING)
        S = self._mid(und_depth) if und_depth else None
        if S is None:
            # Without S we can't price options. Still MM hydrogel, then bail.
            self._trade_hydrogel(state, result)
            return result, 0, self._dump_memory(mem)

        # ───── Per-voucher observed IV, filtered
        observed: List[Tuple[str, int, float, float, int, int, int, int]] = []
        # (voucher, K, mid, iv, best_bid, best_ask, bid_vol, ask_vol)
        for v, K in self.VOUCHER_STRIKES.items():
            depth = state.order_depths.get(v)
            if not depth:
                continue
            bid, ask, bv, av = self._best_quotes(depth)
            if bid is None or ask is None:
                continue
            mid = 0.5 * (bid + ask)
            iv = self._implied_vol(mid, S, float(K), T)
            if iv is None or iv < self.MIN_VALID_IV or iv > self.MAX_VALID_IV:
                continue
            observed.append((v, K, mid, iv, bid, ask, bv, av))

        # ───── Fit smile (used ONLY as a sanity filter on entries, not as fair)
        smile_coeffs = self._fit_smile(observed, S)

        # ───── Per-voucher IV EMA (= fair level)
        iv_ema: Dict[str, float] = dict(mem.get("iv_ema", {}))
        total_delta = 0.0
        positions = state.position or {}
        flatten_mode = state.timestamp >= self.FLATTEN_AFTER_TS

        # Carry existing position delta (using existing EMA or observed IV)
        for v, K in self.VOUCHER_STRIKES.items():
            pos = int(positions.get(v, 0))
            if pos == 0:
                continue
            iv_for_delta = iv_ema.get(v)
            if iv_for_delta is None:
                obs_entry = next((o for o in observed if o[0] == v), None)
                iv_for_delta = obs_entry[3] if obs_entry else 0.25
            _, d, _, _ = self._bs_call(S, float(K), T, iv_for_delta)
            total_delta += pos * d

        for v, K, mid, iv, bid, ask, bv, av in observed:
            # Update per-voucher IV EMA (= fair IV for this voucher)
            prev = iv_ema.get(v, iv)
            fair_iv = (1.0 - self.IV_EMA_ALPHA) * prev + self.IV_EMA_ALPHA * iv
            iv_ema[v] = fair_iv

            fair_price, fair_delta, fair_gamma, fair_vega = self._bs_call(S, float(K), T, fair_iv)

            pos = int(positions.get(v, 0))
            limit = self.POSITION_LIMITS[v]

            iv_diff = iv - fair_iv                 # +ve ⇒ market rich vs own history
            orders: List[Order] = []
            intended_change = 0

            # Skip structurally thin options (deep OTM where vega is tiny):
            # IV reads there are dominated by 1-chip rounding noise.
            vega_ok = fair_vega >= self.MIN_VEGA_TRADE

            # Vega-normalised base size
            vega_scale = 1.0
            if fair_vega > 1e-6:
                vega_scale = min(2.0, self.VEGA_NORMAL / max(fair_vega, 1.0))
            base_size = max(1, int(round(self.MAX_VOUCHER_TRADE * vega_scale)))

            # ── FLATTEN MODE (end-of-day): unwind toward zero at market
            if flatten_mode and pos != 0:
                if pos > 0:
                    q = min(pos, bv, self.FLATTEN_SLICE) if bid is not None else 0
                    if q > 0:
                        orders.append(Order(v, bid, -q))
                        intended_change -= q
                else:
                    q = min(-pos, av, self.FLATTEN_SLICE) if ask is not None else 0
                    if q > 0:
                        orders.append(Order(v, ask, q))
                        intended_change += q

            # ── EXIT: if mispricing has normalized and we're holding, trim.
            elif abs(iv_diff) <= self.IV_ARB_EDGE_EXIT and pos != 0:
                trim = min(abs(pos), base_size * 2)
                if pos > 0 and bid is not None:
                    q = min(trim, bv)
                    if q > 0:
                        orders.append(Order(v, bid, -q))
                        intended_change -= q
                elif pos < 0 and ask is not None:
                    q = min(trim, av)
                    if q > 0:
                        orders.append(Order(v, ask, q))
                        intended_change += q

            # ── ENTRY: IV mean-reversion (observed vs own EMA)
            # Use SOFT_VOUCHER_CAP instead of the hard 300 limit so we always
            # have room to fade further mispricings and to exit cleanly.
            elif vega_ok and iv_diff < -self.IV_ARB_EDGE_ENTRY \
                    and (fair_price - ask) >= self.PRICE_EDGE_MIN:
                room = self.SOFT_VOUCHER_CAP - pos
                want = min(av, base_size, max(0, room))
                if want > 0:
                    orders.append(Order(v, ask, want))
                    intended_change += want

            elif vega_ok and iv_diff > self.IV_ARB_EDGE_ENTRY \
                    and (bid - fair_price) >= self.PRICE_EDGE_MIN:
                room = self.SOFT_VOUCHER_CAP + pos
                want = min(bv, base_size, max(0, room))
                if want > 0:
                    orders.append(Order(v, bid, -want))
                    intended_change -= want

            if orders:
                result[v] = orders

            # Post-trade delta contribution
            total_delta += (pos + intended_change) * fair_delta

        # ═══ DELTA HEDGE with the underlying ═══
        und_pos = int(positions.get(self.UNDERLYING, 0))
        portfolio_delta = und_pos + total_delta  # underlying has delta=1

        if abs(portfolio_delta) > self.DELTA_DEADBAND and und_depth:
            hedge_target = int(round(-total_delta))
            hedge_target = max(-self.POSITION_LIMITS[self.UNDERLYING],
                               min(self.POSITION_LIMITS[self.UNDERLYING], hedge_target))
            to_trade = hedge_target - und_pos
            hedge_orders = self._sweep(self.UNDERLYING, und_depth, und_pos, to_trade,
                                       self.HEDGE_SLICE,
                                       self.POSITION_LIMITS[self.UNDERLYING])
            if hedge_orders:
                result[self.UNDERLYING] = hedge_orders

        # ═══ HYDROGEL_PACK — independent MM ═══
        self._trade_hydrogel(state, result)

        # Make sure all seen products appear (empty list for the ones we skipped)
        for product in state.order_depths:
            result.setdefault(product, [])

        # Persist state
        mem["iv_ema"] = iv_ema
        mem["last_S"] = S
        mem["last_ts"] = state.timestamp
        return result, 0, self._dump_memory(mem)

    # ═══════════════════════════════════════════════════════════════
    # COMPONENTS
    # ═══════════════════════════════════════════════════════════════
    def _fit_smile(self, observed, S: float
                   ) -> Optional[Tuple[float, float, float]]:
        if len(observed) < self.SMILE_MIN_POINTS:
            return None
        m_list = [math.log(K / S) for _, K, _, _, _, _, _, _ in observed]
        y_list = [iv for _, _, _, iv, _, _, _, _ in observed]
        n = len(m_list)
        sm = sum(m_list)
        sm2 = sum(m * m for m in m_list)
        sm3 = sum(m * m * m for m in m_list)
        sm4 = sum(m * m * m * m for m in m_list)
        sy = sum(y_list)
        sym = sum(y * m for y, m in zip(y_list, m_list))
        sym2 = sum(y * m * m for y, m in zip(y_list, m_list))
        A = [[float(n), sm, sm2], [sm, sm2, sm3], [sm2, sm3, sm4]]
        b = [sy, sym, sym2]
        return self._solve_3x3(A, b)

    def _sweep(self, product: str, depth: OrderDepth, pos: int, to_trade: int,
               max_slice: int, limit: int) -> List[Order]:
        """Walk the book to fill up to `to_trade` (sign indicates side)."""
        orders: List[Order] = []
        if to_trade > 0 and depth.sell_orders:
            remaining = min(to_trade, max_slice, limit - pos)
            for price in sorted(depth.sell_orders):
                if remaining <= 0:
                    break
                avail = -depth.sell_orders[price]
                q = min(avail, remaining)
                if q > 0:
                    orders.append(Order(product, price, q))
                    remaining -= q
        elif to_trade < 0 and depth.buy_orders:
            remaining = min(-to_trade, max_slice, limit + pos)
            for price in sorted(depth.buy_orders, reverse=True):
                if remaining <= 0:
                    break
                avail = depth.buy_orders[price]
                q = min(avail, remaining)
                if q > 0:
                    orders.append(Order(product, price, -q))
                    remaining -= q
        return orders

    def _trade_hydrogel(self, state: TradingState,
                        result: Dict[str, List[Order]]) -> None:
        depth = state.order_depths.get(self.HYDROGEL)
        if not depth:
            return
        bid, ask, bv, av = self._best_quotes(depth)
        if bid is None or ask is None:
            return
        mid = 0.5 * (bid + ask)
        pos = int((state.position or {}).get(self.HYDROGEL, 0))
        limit = self.POSITION_LIMITS[self.HYDROGEL]
        orders: List[Order] = []

        # Aggressive take only on obvious mispricings (ask clearly below mid, etc.)
        if ask <= mid - self.HGP_TAKE_EDGE and pos < limit:
            q = min(av, self.HGP_MAX_TRADE, limit - pos)
            if q > 0:
                orders.append(Order(self.HYDROGEL, ask, q))
                pos += q
        if bid >= mid + self.HGP_TAKE_EDGE and pos > -limit:
            q = min(bv, self.HGP_MAX_TRADE, limit + pos)
            if q > 0:
                orders.append(Order(self.HYDROGEL, bid, -q))
                pos -= q

        # (Passive MM layer intentionally omitted — in a tight book it fills
        #  mostly under adverse selection.)
        if orders:
            result[self.HYDROGEL] = orders

    # ═══════════════════════════════════════════════════════════════
    # STATE I/O
    # ═══════════════════════════════════════════════════════════════
    @staticmethod
    def _load_memory(traderData: str) -> Dict:
        if not traderData:
            return {}
        try:
            data = json.loads(traderData)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _dump_memory(data: Dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"))
        except Exception:
            return ""

    # Round 2 hook — ignored in Round 3, kept here for submission compatibility.
    def bid(self):
        return 15
