# Round 5 — The Final Stretch

The final round replaced every previous product with **50 new instruments across 10 groups of 5**
(Galaxy Sounds, Sleep Pods, Microchips, Pebbles, Robots, UV-Visors, Translators, Panels, Oxygen
Shakes, Snack Packs). Every product has a position limit of **10**. The challenge — *"cherry-pick
winners"* — is that only some groups contain tradeable inefficiencies, so the round is about
**finding which products carry directional alpha or pair structure** and concentrating on them.

The standout result: a **full directional strategy across the top-10 alpha products** clears
**+687,942** over the 3 backtest days — far above naive buy-and-hold (+348,711).

## Kept strategies (`strategies/`)

Ranked by 3-day backtest PnL (full table in `backtest_report.md`):

| File | Backtest PnL | Idea |
| :--- | -----------: | :--- |
| `submission_10_full_directional_pnl688k.py` | **+687,942** | **Best.** Full directional exposure across the 50 products (top-50 directional sizing). |
| `submission_14_groupwise_paired_pnl611k.py` | +611,004 | Group-wise paired trading — exploits within-group co-movement between the 5 products of a group. |
| `submission_11_combined_top_alphas_pnl551k.py` | +550,996 | Combines the highest-scoring individual alpha signals into one allocator. |
| `submission_04_top10_directional_pnl515k.py` | +514,716 | Directional exposure restricted to the 10 strongest-signal products. |
| `submission_01_naive_buyhold_pnl349k.py` | +348,711 | Naive buy-and-hold baseline — the bar every alpha strategy must clear. |
| `derek_v2_priority_allocator.py` | research | Priority-based position allocation framework (alternative research line). |
| `derek_v1_cluster_mm.py` | research | Cluster-aware market making. |

> Catastrophic negative submissions (naive market making and unfiltered crossover variants that
> lost millions in backtest) were intentionally dropped. The full ranking, including those, is
> preserved in `backtest_report.md` for the record.

## Research tooling (`research/`)

- `overnight_alpha_loop.py` — the main signal-mining / backtest loop framework (the most developed research file).
- `signal_mining/` — feature engineering, clustering, scoring, and reporting package.
- `signal_visualizer.py`, `feature_scan.py`, `cluster.py`, `extended_signal_research.py` — signal exploration utilities.
- `spread_visualizer.py` — pair/spread analysis. `alpha_validator.py` — quick alpha-signal validation.

## Docs (`docs/`)

- `ROUND5_GROUND_TRUTH_PLAYBOOK.md` — best practices & ground truth.
- `ROUND5_ADVANCED_STRATEGY_BLUEPRINT.md` — strategy design reference.
- `ROUND5_SIGNAL_RESEARCH_SESSION_REPORT.md` — comprehensive signal-research findings.
- `PROSPERITY_REFERENCE.md`, `github_research_options_visualizers.md` — platform & external references.

## Run a backtest

```bash
cd round5
python backtester.py --submission strategies/submission_10_full_directional_pnl688k.py --data-dir data
# reproduce the whole ranking:
python run_all_backtests.py
```

`data/` holds Round 5 prices/trades for days 2 / 3 / 4 (the largest dataset — ~110 MB).

## Manual challenge (notes)

Round 5's manual challenge was a one-day Ignith-exchange portfolio with a convex trading fee
(`fee = (vol/100)^2 ^ budget`, budget = 1,000,000); it is not part of this algorithmic codebase.
See `docs/` for context.
