# Round 4

Round 4 traded the **same 12 instruments as Round 3** (`HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`,
and the `VEV_4000…VEV_6500` voucher chain) but added the round's defining twist: **trades now
carry counterparty IDs** (`Mark 01`, `Mark 14`, … `Mark 67`). One of them — **Mark 67
("Olivia")** — is a buy-only insider on VELVET with ~96% one-tick directional accuracy, making
counterparty-conditioned signals the key alpha of the round.

See **`HANDOVER.md`** for the full counterparty profiles, the verified "dealer-book" hypothesis,
and the market-data timing semantics — it is the single best reference for this round.

## Kept strategies (`strategies/`)

| File | Backtest PnL | Idea |
| :--- | -----------: | :--- |
| `final_submission_round4_pnl64k.py` | **+64,576** | **Best — the actual Round 4 submission.** Round 3 ladder + voucher spread signals + HYDROGEL MA reversion, plus `TAIL_LOTTO` free bids on far-OTM `VEV_6000/6500` and a simplified `HYD_WINDOW=20`. |
| `final_submission_round3_baseline.py` | baseline | The carried-over Round 3 final (SMA(2000) target, voucher spread signals, butterfly vote, lead-lag, tight-surface gate). The reference the Round 4 submission was cut from. |
| `iv_scalping.py` | research | Pure IV-scalping analyzer on the vouchers with Black–Scholes Greeks and Newton–Raphson IV; detects Mark 01/22 activity and amplifies signals. Analytical/dev — not the final submission. |

## Round-specific visualizers (`visualizers/`)

| File | Purpose |
| :--- | :--- |
| `bot_performance_visualizer.py` | Insider / "Mark" profiler — reproduces the Mark 67 scoreboard and per-counterparty PnL. |
| `signal_dashboard.py` | 16-card signal dashboard (`datavisualiser_featured`) for the Round 4 instruments. |
| `round4_log_visualizer.py` | Static per-product HTML emitter from logs. |

## Analysis notebooks (`analysis/`)

- `spread_analysis.ipynb` — HYDROGEL spread-signal discovery (forward returns conditioned on spread mode).
- `IV_scalping_mark_analysis.ipynb` — deep dive on Mark counterparty flow and IV opportunities.
- `Bot1_Trading_Analysis.ipynb` — historical bot trading-pattern analysis.

## Run a backtest

```bash
cd round4
python backtester.py --submission strategies/final_submission_round4_pnl64k.py --data-dir data
python visualizers/bot_performance_visualizer.py --data data --no-browser
```

`data/` holds Round 4 prices/trades for days 1 / 2 / 3 (trades include the counterparty IDs).

## Notes

The headline lesson of Round 4 (per `HANDOVER.md`) is that a pure spread-taker cannot make
+27k on VELVET unless it is informed — so following Mark 67 is real edge. The +64k submission
captures the directional/ladder edge but leaves a dedicated Mark-67-follower module on the table.
