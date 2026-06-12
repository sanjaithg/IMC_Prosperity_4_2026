# Round 1

The first scored round — single-product market making and mean reversion on **PEPPER** and
**OSMIUM**, with a focus on inventory control and disciplined directional taking.

## Kept strategies (`strategies/`)

| File | Product | Idea |
| :--- | :--- | :--- |
| `pepper_risk_taking.py` | PEPPER | **Best PEPPER strategy.** "Sacred-core" inventory: hold a protected ~65-unit base plus a 15-unit trading float, with spike detection and an opportunistic taking layer. |
| `pepper_osmium_mm.py` | PEPPER + OSMIUM | Fair-value market making for PEPPER (dynamic fair = base + drift) coordinated with top-of-book MM on OSMIUM. |
| `osmium_sma_reversion.py` | OSMIUM | SMA(20) mean reversion with accumulation / de-accumulation zones and gradient size multipliers. |

## Run a backtest

```bash
cd round1
python backtester.py strategies/pepper_risk_taking.py --data data
```

`data/` holds the Round 1 (`round_1`) price/trade CSVs.

> Note: `pepper_risk_taking.py` backtests to ~ **+274K** across the available days. The Round 1
> data files were normalized to a single (`;`) delimiter — the original source mixed `,` and `;`.
