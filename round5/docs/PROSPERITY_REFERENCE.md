# Prosperity 4 — Algorithmic Trading Reference

## What You're Building

A Python `Trader` class with a `run()` method. Every simulation tick, `run()` receives a `TradingState` and returns orders. Goal: maximize XIRECs (the exchange currency).

- **Test simulation**: 1,000 iterations on historical sample data
- **Final simulation**: 10,000 iterations — determines your PnL for the round
- **Round 2 only**: also define a `bid()` method (safe to include in all rounds, ignored elsewhere)

---

## The `run()` Method — What Goes In, What Comes Out

```python
def run(self, state: TradingState):
    result = {}                    # product → List[Order]
    conversions = 0                # int, optional conversion request
    traderData = ""                # string, persists to next tick via state.traderData
    return result, conversions, traderData
```

**Response time limit: 900ms per call** (average should be ≤100ms).

---

## `TradingState` — The Market Snapshot

| Property | Type | What it contains |
|---|---|---|
| `timestamp` | `int` | Current simulation tick |
| `traderData` | `str` | String you returned last tick (your persistent state) |
| `order_depths` | `Dict[Symbol, OrderDepth]` | All live bot quotes per product |
| `own_trades` | `Dict[Symbol, List[Trade]]` | Your trades since last tick |
| `market_trades` | `Dict[Symbol, List[Trade]]` | Other participants' trades since last tick |
| `position` | `Dict[Product, int]` | Your current position per product (signed int) |
| `observations` | `Observation` | Market observations (see below) |

---

## `OrderDepth` — The Order Book

```python
order_depth.buy_orders   # Dict[price, quantity]  — positive quantities
order_depth.sell_orders  # Dict[price, quantity]  — NEGATIVE quantities
```

**Example:**
```python
buy_orders  = {9: 5, 10: 4}    # bots willing to buy: 5 units at 9, 4 units at 10
sell_orders = {11: -4, 12: -8} # bots willing to sell: 4 units at 11, 8 units at 12
```

- Best bid = highest key in `buy_orders`
- Best ask = lowest key in `sell_orders` (value will be negative)
- All buy prices are always strictly below all sell prices

---

## Sending Orders — The `Order` Class

```python
Order(symbol, price, quantity)
# quantity > 0  →  BUY  order
# quantity < 0  →  SELL order
```

**How matching works:**
1. If your order price crosses an existing bot quote → immediate execution
2. Remaining unfilled quantity stays visible to bots for the rest of the tick
3. If no bot trades against it → automatically cancelled at tick end
4. Bots can trade with each other after your orders are cancelled, before next tick

**Order execution is instantaneous** — no latency, no bot can front-run you.

---

## Position Limits

- Enforced per product. Absolute limit — can't exceed in either direction.
- If your aggregated buy (sell) orders **would** breach the limit if all fully filled → **all orders for that product are rejected**
- Formula: `max_buy_qty = position_limit - current_position`

**Example:** limit=30, position=-5 → max buy quantity = 30−(−5) = **35**

---

## Persistent State — `traderData`

AWS Lambda is stateless — class/global variables are NOT guaranteed to persist between ticks. Use `traderData` to carry state:

```python
import jsonpickle

# Save state
traderData = jsonpickle.encode({"my_var": 42, "prices": [100, 101]})

# Restore state
data = jsonpickle.decode(state.traderData) if state.traderData else {}
```

**Hard limit: 50,000 characters.** Exceeding this truncates the string and may corrupt deserialization.

---

## `Trade` Object

Fields: `symbol`, `price`, `quantity`, `buyer`, `seller`, `timestamp`

- `buyer`/`seller` are empty strings unless **you** are the counterparty
- Your trades show `buyer="SUBMISSION"` or `seller="SUBMISSION"`

---

## Observations

Two types inside `state.observations`:

| Type | Access | Use |
|---|---|---|
| Simple values | `state.observations.plainValueObservations[product]` | Plain int/float market signals |
| Conversion data | `state.observations.conversionObservations[product]` | For conversion requests |

`ConversionObservation` fields: `bidPrice`, `askPrice`, `transportFees`, `exportTariff`, `importTariff`, `sunlight`, `humidity`

**Conversion rules:**
- Must already hold a position to convert
- Request cannot exceed your current position size
- Costs: transport fees + import/export tariff apply
- Return `conversions = 0` or `None` if not using

---

## Supported Libraries (Python 3.12)

`pandas`, `numpy`, `statistics`, `math`, `typing`, `jsonpickle` + all Python 3.12 standard library. **No other external libraries.**

---

## Minimal Working Template

```python
from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle

class Trader:

    def bid(self):
        return 15  # Required for Round 2, ignored elsewhere

    def run(self, state: TradingState):
        # Restore persistent state
        data = jsonpickle.decode(state.traderData) if state.traderData else {}

        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            position = state.position.get(product, 0)

            # --- Your logic here ---
            fair_value = 10000  # compute this properly per product

            # Take cheap asks
            for ask, ask_qty in sorted(order_depth.sell_orders.items()):
                if ask < fair_value:
                    buy_qty = min(-ask_qty, 20 - position)  # respect position limit
                    if buy_qty > 0:
                        orders.append(Order(product, ask, buy_qty))
                        position += buy_qty

            # Take expensive bids
            for bid, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid > fair_value:
                    sell_qty = min(bid_qty, 20 + position)  # respect position limit
                    if sell_qty > 0:
                        orders.append(Order(product, bid, -sell_qty))
                        position -= sell_qty

            result[product] = orders

        traderData = jsonpickle.encode(data)
        conversions = 0
        return result, conversions, traderData
```

---

## Debugging Tips

- Use `print()` freely inside `run()` — output appears in the log file after upload
- Log file also shows your submission UUID and runID — include these when asking for help
- Test locally using `datamodel.py` (Appendix B) to construct `TradingState` objects manually
- `own_trades` tells you what actually filled last tick — use it to track slippage
- Watch `market_trades` to infer what other bots value a product at
