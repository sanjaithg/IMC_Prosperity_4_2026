# Tutorial Round

The practice round used to learn the platform — fair-value estimation, passive market making,
inventory control, and simple directional taking on the introductory instruments
**TOMATO** and **EMERALD**.

## Kept strategies (`strategies/`)

| File | Product | Idea |
| :--- | :--- | :--- |
| `tomato_hybrid.py` | TOMATO | EMA fair value with penny-jumping passive quotes plus an active market-taking layer and inventory skew. |
| `emerald_mm.py` | EMERALD | Simple, robust mean-reversion market making quoting around a stable fair value (~10,000). |

## Run a backtest

```bash
cd tutorial
python backtester.py strategies/tomato_hybrid.py --data data
```

`data/` holds the tutorial (`round_0`) price/trade CSVs. (Backtested PnL ~ tomato +17.8K,
emerald +16.6K across the available days.)
