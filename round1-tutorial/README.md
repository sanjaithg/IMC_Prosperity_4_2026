# Round 1 / Tutorial

The opening round — single-product market making and mean reversion on the introductory
instruments (this team's themed names: **EMERALD**, **TOMATO**, **OSMIUM**, **PEPPER**).
The goal was to learn the platform: fair-value estimation, passive market making, inventory
control, and simple directional taking.

## Kept strategies (`strategies/`)

| File | Product | Idea |
| :--- | :--- | :--- |
| `pepper_risk_taking.py` | PEPPER | **Best PEPPER strategy.** "Sacred-core" inventory: hold a protected ~65-unit base plus a 15-unit trading float, with spike detection and an opportunistic taking layer. |
| `pepper_osmium_mm.py` | PEPPER + OSMIUM | Fair-value market making for PEPPER (dynamic fair = base + drift) coordinated with top-of-book MM on OSMIUM. |
| `osmium_sma_reversion.py` | OSMIUM | SMA(20) mean reversion with accumulation / de-accumulation zones and gradient size multipliers. |
| `tomato_hybrid.py` | TOMATO | EMA fair value with penny-jumping passive quotes plus an active market-taking layer and inventory skew. |
| `emerald_mm.py` | EMERALD | Simple, robust mean-reversion MM quoting around a stable fair value (~10,000). |

## Run a backtest

```bash
cd round1-tutorial
python backtester.py --submission strategies/pepper_risk_taking.py --data-dir data
```

`data/` holds the tutorial (`round_0`) and Round 1 (`round_1`) price/trade CSVs.

## Notes

These are the most developed versions of each idea, pulled together from across the
team's tutorial workspaces. They are intentionally simple — the heavier signal research starts
in Round 3.
