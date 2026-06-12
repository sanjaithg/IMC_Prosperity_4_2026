# Round 2

Round 2 built on the tutorial instruments (notably **PEPPER** and **OSMIUM**) and introduced
basket / spread dynamics. The big jump in PnL here came from combining inventory-aware
accumulation on PEPPER with EMA fade-trading and spike handling on OSMIUM.

## Kept strategies (`strategies/`)

Ranked by backtested PnL (final submission snapshots — folder IDs are the platform submission IDs):

| File | Backtest PnL | Idea |
| :--- | -----------: | :--- |
| `final_362616_pnl98k.py` | **+98,131** | **Best.** Jump detection + gap avoidance for PEPPER accumulation; slow-EMA fade trading on OSMIUM with independent spike handling. Final tuned params (`mm_size=30`, `fade_gain=6`). |
| `ipr_reversion_274011_pnl96k.py` | **+96,627** | IPR reversion-cycle layer with a z-score signal over rolling mid history; PEPPER "sacred-core" target of 65 with bounded transient deviation. |
| `simplified_358105.py` | +9,405 | Stripped-down intermediate: gap-avoidance on PEPPER buys + simple OSMIUM MM. Kept as a readable baseline. |
| `dev_code_v5.py` | dev | Latest development iteration (`Code copy 5`) — the working strategy the submissions were cut from. |

## Run a backtest

```bash
cd round2
python backtester.py --submission strategies/final_362616_pnl98k.py --data-dir data
```

`data/` holds the Round 2 price/trade CSVs (`round_2`, days -1 / 0 / 1).

## Notes

The two top strategies are within ~1.5% of each other on PnL but take different routes:
`362616` leans on OSMIUM fade trading, `274011` on a cyclical mean-reversion signal. Both are
worth studying.
