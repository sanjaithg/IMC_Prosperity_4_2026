# Round 3

Round 3 introduced an **options sub-market**: a spot underlying plus a chain of European-style
call vouchers.

- `HYDROGEL_PACK` (limit ±200) — independent spot product, market-made on its own.
- `VELVETFRUIT_EXTRACT` (limit ±200) — the underlying for the voucher chain.
- `VEV_4000 … VEV_6500` (limit ±300 each) — 10 call vouchers across strikes.

The winning approach combined a **rolling-SMA mean-reversion ladder** on VELVET + vouchers with a
separate **HYDROGEL state machine**, layered with voucher spread signals, a butterfly-vote, book
imbalance, and lead-lag amplification.

## Kept strategies (`strategies/`)

| File | Backtest PnL | Idea |
| :--- | -----------: | :--- |
| `final_submission_pnl28k.py` | **+28,156** | **Best.** Merges the top-of-book mean-reversion ladder (VELVET + VEV) with the HYDROGEL state machine; 4 voucher spread signals, butterfly votes, book imbalance, lead-lag, tight-surface gate, aggressive top-of-book taking at lot max. |
| `submission_final.py` | top-tier | Sibling final: sub34 ladder + inventory-skewed MM with a TAKE layer + HYDROGEL state machine, with refined conflict resolution. |
| `submission35_top_of_book_mr.py` | strong | Adds an aggressive top-of-book taking layer (fires on 2+ aligned votes / high conviction) on top of the SMA(2000) reversion ladder; passive by default. |
| `submission34_mean_reversion_ladder.py` | foundation | The continuous rolling-SMA reversion target (`SMA_WINDOW=2000`, validated by 3-day Spearman analysis) that later submissions build on. |
| `submission02_gamma_scalp.py` | research | Distinct delta-neutral approach: fits a quadratic IV smile across vouchers, trades relative-value mispricings, delta-hedges with VELVET, market-makes HYDROGEL independently. |

## Run a backtest

```bash
cd round3
python backtester.py --submission strategies/final_submission_pnl28k.py --data-dir data
# strategy-specific charts:
python visualize_strategy.py
```

`data/` holds Round 3 prices/trades for days 0 / 1 / 2.

## Notes

`final_submission_pnl28k.py` and `submission_final.py` are two closely related "final" lines —
the former is the headline performer; the latter is kept because it resolves signal conflicts
differently and is a useful comparison. `submission02_gamma_scalp.py` is kept as the cleanest
example of the pure options-pricing route even though it underperformed the ladder.
