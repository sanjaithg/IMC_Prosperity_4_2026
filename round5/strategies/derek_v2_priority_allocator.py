"""
Round 5 priority allocator strategy.

This is the same signal family as v2_priority_combined, but it allocates scarce
per-product capacity rule-by-rule instead of summing all targets and clipping.
That matters most for PEBBLES_XL, where multiple profitable structures want the
same side of the 10-lot limit.
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState


CONFIG = {
    "candidate_id": "EXP_priority_allocator",
    "pos_limit": 10,
    "clip": 1,
    "cooldown_ticks": 20,
    "max_orders_per_tick": 10,
    "pair_rules": [
        {
            "enabled": True,
            "name": "peb_xs_xl_spread_high",
            "a": "PEBBLES_XS",
            "b": "PEBBLES_XL",
            "basis": "spread",
            "window": 100,
            "entry_z": 2.5,
            "exit_z": 0.0,
            "target": 10,
            "side": "high",
            "priority": 4.0,
        },
        {
            "enabled": True,
            "name": "peb_s_xl_spread_high",
            "a": "PEBBLES_S",
            "b": "PEBBLES_XL",
            "basis": "spread",
            "window": 100,
            "entry_z": 3.0,
            "exit_z": 0.0,
            "target": 10,
            "side": "high",
            "priority": 3.0,
        },
        {
            "enabled": True,
            "name": "peb_s_xl_sum_low",
            "a": "PEBBLES_S",
            "b": "PEBBLES_XL",
            "basis": "sum",
            "window": 100,
            "entry_z": 2.5,
            "exit_z": 0.0,
            "target": 10,
            "side": "low",
            "priority": 2.0,
        },
        {
            "enabled": True,
            "name": "snack_pist_rasp_spread_high",
            "a": "SNACKPACK_PISTACHIO",
            "b": "SNACKPACK_RASPBERRY",
            "basis": "spread",
            "window": 200,
            "entry_z": 3.0,
            "exit_z": 0.0,
            "target": 10,
            "side": "high",
            "priority": 2.5,
        },
        {
            "enabled": False,
            "name": "panel_sleep_spread_high",
            "a": "PANEL_4X4",
            "b": "SLEEP_POD_COTTON",
            "basis": "spread",
            "window": 100,
            "entry_z": 2.5,
            "exit_z": 0.0,
            "target": 10,
            "side": "high",
            "priority": 1.5,
        },
        {
            "enabled": False,
            "name": "snack_choc_van_spread_high",
            "a": "SNACKPACK_CHOCOLATE",
            "b": "SNACKPACK_VANILLA",
            "basis": "spread",
            "window": 200,
            "entry_z": 3.0,
            "exit_z": 0.0,
            "target": 10,
            "side": "high",
            "priority": 1.0,
        },
    ],
    "pressure_rules": [
        {
            "enabled": True,
            "name": "snack_rasp_edge_pressure",
            "product": "SNACKPACK_RASPBERRY",
            "feature": "microprice_edge_rel",
            "threshold": 0.30,
            "target": 5,
            "priority": 0.5,
        }
    ],
}


Stats = Tuple[int, int, float, int, int, float, float, float]


def _book_stats(od: OrderDepth) -> Stats:
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    mid = (best_bid + best_ask) / 2.0

    bid_vol_1 = od.buy_orders.get(best_bid, 0)
    ask_vol_1 = abs(od.sell_orders.get(best_ask, 0))
    denom_l1 = max(bid_vol_1 + ask_vol_1, 1)
    imbalance_l1 = (bid_vol_1 - ask_vol_1) / denom_l1

    bid_levels = sorted(od.buy_orders.items(), reverse=True)[:3]
    ask_levels = sorted(od.sell_orders.items())[:3]
    bid_vol = sum(v for _price, v in bid_levels if v > 0)
    ask_vol = sum(abs(v) for _price, v in ask_levels)
    weighted_bid = sum(max(v, 0) * (3 - i) for i, (_price, v) in enumerate(bid_levels))
    weighted_ask = sum(abs(v) * (3 - i) for i, (_price, v) in enumerate(ask_levels))
    weighted_imbalance = (weighted_bid - weighted_ask) / max(weighted_bid + weighted_ask, 1)

    spread = max(float(best_ask - best_bid), 1.0)
    microprice = (best_ask * bid_vol_1 + best_bid * ask_vol_1) / denom_l1
    microprice_edge_rel = (microprice - mid) / spread
    return best_bid, best_ask, mid, bid_vol, ask_vol, weighted_imbalance, imbalance_l1, microprice_edge_rel


def _push_and_z(data: dict, key: str, value: float, window: int) -> float:
    hist = list(data.get(key, []))
    if len(hist) < max(20, window // 5):
        z = 0.0
    else:
        mean = sum(hist) / len(hist)
        var = sum((x - mean) * (x - mean) for x in hist) / len(hist)
        z = 0.0 if var <= 1e-12 else (value - mean) / (var ** 0.5)
    hist.append(value)
    data[key] = hist[-window:]
    return z


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


class Trader:
    def __init__(self):
        self.cfg = dict(CONFIG)

    def _pair_action(self, rule: dict, stats: Dict[str, Stats], data: dict) -> dict | None:
        a = rule["a"]
        b = rule["b"]
        if not rule.get("enabled", True) or a not in stats or b not in stats:
            return None

        basis_kind = str(rule.get("basis", "spread"))
        basis = stats[a][2] + stats[b][2] if basis_kind == "sum" else stats[a][2] - stats[b][2]
        name = str(rule.get("name", a + "_" + b))
        z = _push_and_z(data, f"hist_{name}", basis, int(rule.get("window", 100)))

        regime_key = f"regime_{name}"
        regime = int(data.get(regime_key, 0) or 0)
        entry = float(rule.get("entry_z", 3.0))
        exit_z = float(rule.get("exit_z", 0.0))
        side = str(rule.get("side", "both"))
        if regime == 0:
            if side in {"both", "high", "short_signal"} and z > entry:
                regime = -1
            elif side in {"both", "low", "long_signal"} and z < -entry:
                regime = 1
        elif abs(z) <= exit_z:
            regime = 0
        data[regime_key] = regime
        if regime == 0:
            return None

        target = int(rule.get("target", self.cfg.get("pos_limit", 10)))
        if basis_kind == "sum":
            deltas = {a: regime * target, b: regime * target}
        else:
            deltas = {a: regime * target, b: -regime * target}
        score = float(rule.get("priority", 1.0)) * max(abs(z) - entry, 0.0)
        return {"name": name, "deltas": deltas, "score": score}

    def _pressure_action(self, rule: dict, stats: Dict[str, Stats]) -> dict | None:
        product = rule["product"]
        if not rule.get("enabled", True) or product not in stats:
            return None
        feature = str(rule.get("feature", "microprice_edge_rel"))
        if feature == "imbalance_l1":
            signal = stats[product][6]
        elif feature == "weighted_imbalance":
            signal = stats[product][5]
        elif feature == "combo":
            signal = 0.50 * stats[product][5] + 0.30 * stats[product][6] + 0.20 * stats[product][7]
        else:
            signal = stats[product][7]
        threshold = float(rule.get("threshold", 0.30))
        if abs(signal) < threshold:
            return None
        raw = 1 if signal > 0 else -1
        target = int(rule.get("target", 5))
        score = float(rule.get("priority", 1.0)) * (abs(signal) - threshold)
        return {"name": str(rule.get("name", product)), "deltas": {product: raw * target}, "score": score}

    def _fits(self, targets: Dict[str, int], deltas: Dict[str, int], limit: int) -> bool:
        for product, delta in deltas.items():
            new_target = targets.get(product, 0) + delta
            if abs(new_target) > limit:
                return False
        return True

    def run(self, state: TradingState):
        try:
            data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        cfg = self.cfg
        products = set()
        for rule in cfg.get("pair_rules", []):
            products.add(rule["a"])
            products.add(rule["b"])
        for rule in cfg.get("pressure_rules", []):
            products.add(rule["product"])

        stats: Dict[str, Stats] = {}
        for product in sorted(products):
            od = state.order_depths.get(product)
            if od is not None and od.buy_orders and od.sell_orders:
                stats[product] = _book_stats(od)

        actions = []
        for rule in cfg.get("pair_rules", []):
            action = self._pair_action(rule, stats, data)
            if action:
                actions.append(action)
        for rule in cfg.get("pressure_rules", []):
            action = self._pressure_action(rule, stats)
            if action:
                actions.append(action)

        limit = int(cfg.get("pos_limit", 10))
        targets: Dict[str, int] = {product: 0 for product in products}
        for action in sorted(actions, key=lambda a: float(a["score"]), reverse=True):
            deltas = action["deltas"]
            if self._fits(targets, deltas, limit):
                for product, delta in deltas.items():
                    targets[product] = targets.get(product, 0) + delta
                continue
            # Allow one-product overlays such as pressure if they do not fight
            # the existing target direction.
            if len(deltas) == 1:
                product, delta = next(iter(deltas.items()))
                if _sign(delta) == _sign(targets.get(product, 0)):
                    targets[product] = max(-limit, min(limit, targets.get(product, 0) + delta))

        result: Dict[str, List[Order]] = {}
        n_orders = 0
        for product in sorted(targets):
            if product not in stats:
                continue
            cd_key = f"cd_{product}"
            cooldown = int(data.get(cd_key, 0) or 0)
            if cooldown > 0:
                data[cd_key] = cooldown - 1
                continue

            best_bid, best_ask, _mid, bid_vol, ask_vol, _weighted, _imb, _edge = stats[product]
            pos = int(state.position.get(product, 0))
            target = max(-limit, min(limit, int(targets.get(product, 0))))
            delta = target - pos
            if delta == 0:
                continue

            clip = int(cfg.get("clip", 1))
            orders: List[Order] = []
            if delta > 0:
                qty = min(delta, clip, ask_vol, limit - pos)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
            else:
                qty = min(-delta, clip, bid_vol, limit + pos)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
            if orders:
                result[product] = orders[: int(cfg.get("max_orders_per_tick", 10))]
                n_orders += len(result[product])
                data[cd_key] = int(cfg.get("cooldown_ticks", 20))

        if n_orders:
            print(f"t={state.timestamp} o={n_orders}"[:98])
        return result, 0, json.dumps(data)

