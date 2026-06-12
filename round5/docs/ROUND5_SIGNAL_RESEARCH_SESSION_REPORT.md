# Round 5 Signal Research Session Report

Generated: 2026-04-30

This document summarizes the signal-mining and correlation research from this session. It is intentionally focused on the analysis layer: feature families, correlation structure, residual lead-lag, clusters, verified constraints, and what translated into robust strategy ideas. Backtest PnL is included only where it helps judge whether a discovered relationship survived contact with execution.

## Executive Readout

The strongest research conclusion is that Round 5 contains several real local structures, but most raw correlation edges are either common-factor contaminated or too expensive to trade naively. The useful structures came from three places:

- `PEBBLES` constraint reversion, especially `PEBBLES_XS - PEBBLES_XL` when the spread is high.
- `SNACKPACK` local spread reversion, especially `SNACKPACK_PISTACHIO - SNACKPACK_RASPBERRY` and later `SNACKPACK_CHOCOLATE - SNACKPACK_VANILLA`.
- A handful of independent spread clusters, especially `PANEL_4X4 / SLEEP_POD_COTTON`.

The most important research correction was moving from broad raw correlations to refined/residualized analysis. The initial mining pass found many signals, but several pair edges looked suspiciously duplicated across products. The refined pass removed common market factors by demeaning features and targets cross-sectionally by day/timestamp. That produced cleaner lead-lag signals, smaller cluster graphs, and a better separation between real local relationships and market-wide noise.

The second important correction was one-sided testing. The event studies often reported a specific profitable side, such as high spread mean-reverting. When translated into two-sided reversion, several candidates lost badly because the unverified opposite side was harmful. Restricting to the verified side turned the best `PEBBLES` structures from large losers into large winners in backtest.

## Scope And Artifacts

Primary analysis artifacts:

- `artifacts/signal_mining_refined/final_report.md`
- `artifacts/signal_mining_refined/feature_scores.csv`
- `artifacts/signal_mining_refined/pair_scores.csv`
- `artifacts/signal_mining_refined/spread_scores.csv`
- `artifacts/signal_mining_refined/mirror_scores.csv`
- `artifacts/signal_mining_refined/cluster_scores.csv`
- `artifacts/signal_mining_refined/signal_catalog.csv`
- `artifacts/signal_mining_refined/priority_checks/priority_translation_report.md`
- `artifacts/signal_mining_refined/priority_checks/snackpack_pressure_thresholds.csv`
- `artifacts/signal_mining_refined/priority_checks/snackpack_constraint_reversion.csv`
- `artifacts/signal_mining_refined/priority_checks/pebbles_pair_reversion.csv`
- `artifacts/signal_mining_refined/priority_checks/microchip_oval_spread_volatility.csv`

Strategy and execution-check artifacts:

- `artifacts/priority_structure_tests/priority_backtest_one_sided_summary.csv`
- `artifacts/priority_structure_tests/combined_plus_summary.csv`
- `artifacts/priority_structure_tests/stable_timestamp_current_pnl.csv`
- `artifacts/secondary_signal_tests/secondary_summary.csv`

Key code created or modified for research:

- `analysis/signal_mining/features.py`
- `analysis/signal_mining/scoring.py`
- `analysis/signal_mining/clustering.py`
- `analysis/signal_mining/run_signal_miner.py`
- `analysis/signal_mining/verify_priority_structures.py`
- `Submissions_Derek/v2_priority_pair_reversion.py`
- `Submissions_Derek/v2_priority_combined.py`
- `Submissions_Derek/v2_secondary_signal_tests.py`
- `Submissions_Derek/v2_priority_allocator.py`
- `Submissions_Derek/v2_stable_timestamp.py`

## Methodology

The research moved through five stages.

Stage 1: broad signal mining. The initial signal miner generated a large catalog of single-product features, pair relationships, spread scores, mirror scores, interaction scores, and clusters. It searched days 2, 3, and 4.

Stage 2: refined signal mining. The refined pass narrowed the feature set to normalized and locally interpretable features, added residualized pair scoring, and tightened graph clustering. This produced `12950` feature scores, `171500` pair scores, `500` cataloged signals, and `13` clusters.

Stage 3: priority-claim verification. User-provided claims and observed structures were tested directly. This included snackpack pressure, snackpack constraints, pebbles constraints, and `MICROCHIP_OVAL` volatility behavior.

Stage 4: translation to sparse candidates. The strongest relationships were translated into simple market-taking strategies to see if the signal survived spread, fill behavior, position caps, and target conflicts.

Stage 5: robustness-oriented cleanup. Aggressive combinations were separated from cleaner submit-style strategies. This exposed the tradeoff between high in-sample PnL and turnover/overfit risk.

## Feature Families Studied

The refined feature set centered on interpretable microstructure and normalization rather than arbitrary model features.

Volume and depth features:

- `imbalance_l1`
- `imbalance_all`
- `weighted_imbalance`
- `depth_z_*`
- `total_depth_delta_*`
- `imbalance_l1_delta_*`
- `weighted_imbalance_delta_*`

Microprice features:

- `microprice_edge_rel`
- `microprice_edge_z_20`
- `microprice_edge_z_50`
- `microprice_edge_z_100`
- `microprice_edge_demeaned_20`
- `microprice_edge_demeaned_50`
- `microprice_edge_demeaned_100`

Spread and local regime features:

- `spread_z_20`
- `spread_z_50`
- `ret_mad_z_*`
- `volatility_50`
- `volatility_100`
- `autocorr_proxy_20`
- `oscillation_proxy`
- `trade_intensity_20`

The most consistently useful feature family was short-horizon book pressure in `SNACKPACK` products. The most useful structural family was pair/constraint reversion in `PEBBLES` and selected spread clusters.

## Top Single-Product Feature Findings

The refined report ranked `SNACKPACK` book-pressure and microprice signals highest among confirmed single-product signals.

Strong confirmed examples:

- `SNACKPACK_PISTACHIO` `weighted_imbalance`, horizon `1`, confirmed, score about `0.1476`.
- `SNACKPACK_PISTACHIO` `microprice_edge_rel`, horizon `1`, confirmed, score about `0.1465`.
- `SNACKPACK_PISTACHIO` `imbalance_l1`, horizon `1`, confirmed, score about `0.1465`.
- `SNACKPACK_PISTACHIO` `imbalance_all`, horizon `1`, confirmed, score about `0.1451`.
- `SNACKPACK_VANILLA` `weighted_imbalance`, horizon `1`, confirmed, score about `0.1363`.
- `SNACKPACK_VANILLA` `imbalance_l1`, horizon `1`, confirmed, score about `0.1348`.
- `SNACKPACK_CHOCOLATE` `microprice_edge_rel`, horizon `1`, confirmed, score about `0.1277`.

Interpretation:

The strongest single-name signals were not long-horizon trend signals. They were short-horizon order-book pressure signals. That makes them useful as overlays or execution filters, but dangerous as standalone large-position strategies because they can generate many fills and pay spread frequently.

Practical conclusion:

Use `SNACKPACK` pressure sparingly. It is real, but it is not automatically high-PnL after costs. The raspberry pressure overlay helped some combined tests, but broad pressure bundles and lead-lag takers did not generalize well in execution tests.

## Residual Lead-Lag Findings

The refined pair scorer added common-factor residualization:

- Leader features were demeaned by day/timestamp cross-sectional mean.
- Follower future returns were also demeaned by day/timestamp cross-sectional mean.
- This reduced broad market-move contamination.

Top residual lead-lag findings:

- `SNACKPACK_CHOCOLATE -> SNACKPACK_PISTACHIO`, feature `weighted_imbalance`, horizon `1`, confirmed, score about `0.04694`.
- `SNACKPACK_RASPBERRY -> SNACKPACK_PISTACHIO`, feature `weighted_imbalance`, horizon `1`, confirmed, score about `0.04694`.
- `SNACKPACK_STRAWBERRY -> SNACKPACK_PISTACHIO`, feature `weighted_imbalance`, horizon `1`, confirmed, score about `0.04694`.
- `SNACKPACK_VANILLA -> SNACKPACK_PISTACHIO`, feature `weighted_imbalance`, horizon `1`, confirmed, score about `0.04694`.
- `SNACKPACK_* -> SNACKPACK_VANILLA`, feature `weighted_imbalance`, horizon `1`, confirmed, scores around `0.04618`.
- `MICROCHIP_SQUARE -> MICROCHIP_CIRCLE`, feature `weighted_imbalance`, horizon `1`, confirmed, score about `0.04572`.

Interpretation:

The residual lead-lag structure looked statistically real, especially inside `SNACKPACK`, but the edge was small and very short horizon. When translated into simple market-taking strategies, the broad snackpack lead-lag tests lost money. That does not invalidate the signal; it means the signal likely needs better execution, passive quoting, lower spread exposure, or use as a direction filter rather than a direct taker.

Execution check:

- `SNACK_leadlag_basket_to_PISTACHIO_*` variants were negative on all three days in the simple taker tests.
- `SNACK_leadlag_basket_to_VANILLA_*` variants were also negative.
- `MICRO_SQUARE_to_CIRCLE_weighted_imbalance_follow` was only about `+172` total and not meaningful.

Practical conclusion:

Lead-lag is a research signal, not yet a submit-ready direct strategy. Treat it as a filter or a passive quoting lean, not a market-taking alpha.

## Cluster Findings

The refined cluster graph was intentionally stricter than the first broad pass. It limited edges per node and required stronger stability scores. This avoided the earlier problem where clusters became too large and meaningless.

Top refined spread mean-reversion clusters:

- `GALAXY_SOUNDS_DARK_MATTER`, `ROBOT_LAUNDRY`, `UV_VISOR_AMBER`, `UV_VISOR_ORANGE`, cluster score about `0.3847`.
- `GALAXY_SOUNDS_SOLAR_FLAMES`, `MICROCHIP_TRIANGLE`, `PANEL_1X2`, `PEBBLES_XL`, `TRANSLATOR_ASTRO_BLACK`, score about `0.3247`.
- `OXYGEN_SHAKE_MINT`, `PANEL_2X2`, `SNACKPACK_PISTACHIO`, `SNACKPACK_STRAWBERRY`, `UV_VISOR_MAGENTA`, score about `0.3145`.
- `MICROCHIP_OVAL`, `PANEL_1X4`, `SNACKPACK_VANILLA`, `TRANSLATOR_GRAPHITE_MIST`, score about `0.2633`.
- `ROBOT_MOPPING`, `ROBOT_VACUUMING`, `SNACKPACK_RASPBERRY`, score about `0.1811`.
- `MICROCHIP_CIRCLE`, `PEBBLES_S`, score about `0.0886`.
- `PANEL_4X4`, `SLEEP_POD_COTTON`, score about `0.0865`.
- `PEBBLES_XS`, `UV_VISOR_YELLOW`, score about `0.0810`.

Interpretation:

Cluster membership alone is not enough. Some high-scoring clusters had unintuitive mixed products and did not all translate cleanly into stable PnL. Smaller pair clusters were easier to translate because they avoided multi-product cap conflicts.

Execution check from secondary tests:

- `PEBBLES_XS / UV_VISOR_YELLOW` high spread: about `+40,697`, but day 4 negative.
- `ROBOT_MOPPING / ROBOT_VACUUMING` low spread: about `+30,438`, but day 2 negative.
- `MICROCHIP_CIRCLE / PEBBLES_S` low spread: about `+26,143`, but day 3 slightly negative.
- `PANEL_4X4 / SLEEP_POD_COTTON` high spread: about `+23,623`, positive all three days.
- `SNACKPACK_CHOCOLATE / SNACKPACK_VANILLA` high spread: about `+14,559`, positive all three days.

Practical conclusion:

The independent pairs `PANEL_4X4 / SLEEP_POD_COTTON` and `SNACKPACK_CHOCOLATE / SNACKPACK_VANILLA` were the most useful secondary cluster relationships because they were positive on all days and did not compete with the core `PEBBLES_XL` capacity.

## Priority Structure Verification

The priority verification script translated specific claims into threshold studies. This was the strongest bridge between raw signal mining and actionable strategy construction.

### Snackpack Pressure

Best verified rows:

- `SNACKPACK_RASPBERRY` `microprice_edge_rel`, horizon `10`, threshold `0.3`, side `short_signal`, `n=477`, mean signed move about `6.64`, hit rate about `0.616`.
- `SNACKPACK_RASPBERRY` `imbalance_l1`, horizon `10`, threshold `0.05` through `0.3`, side `short_signal`, `n=504`, mean signed move about `6.40`, hit rate about `0.611`.

Interpretation:

Raspberry book pressure is real in event-study form. But as a standalone market-taking system, it is modest and can be fragile. It works better as an overlay inside broader `SNACKPACK` structures than as the main engine.

### Snackpack Constraint Reversion

Best verified rows:

- `SNACKPACK_PISTACHIO / SNACKPACK_RASPBERRY`, spread basis, window `200`, horizon `50`, threshold `3.0`, side `short_signal`, `n=209`, mean signed move about `33.995`, hit rate about `0.684`.
- `SNACKPACK_PISTACHIO / SNACKPACK_RASPBERRY`, spread basis, window `100`, horizon `50`, threshold `3.0`, `n=223`, mean about `24.879`, hit rate about `0.637`.
- `SNACKPACK_PISTACHIO / SNACKPACK_RASPBERRY`, spread basis, window `200`, horizon `50`, threshold `2.5`, `n=722`, mean about `21.576`, hit rate about `0.596`.
- `SNACKPACK_CHOCOLATE / SNACKPACK_VANILLA`, spread basis, window `200`, horizon `50`, threshold `3.0`, side `short_signal`, `n=145`, mean about `12.786`, hit rate about `0.490`.

Interpretation:

`PISTACHIO/RASPBERRY` is the cleaner snackpack relationship. It is smaller than `PEBBLES`, but more stable in day-level PnL tests. `CHOCOLATE/VANILLA` has weaker hit rate in the threshold report, but still worked as an independent positive secondary pair when tested one-sided.

### Pebbles Constraint Reversion

Best verified rows:

- `PEBBLES_XS / PEBBLES_XL`, spread basis, window `100`, horizon `50`, threshold `3.0`, side `short_signal`, `n=194`, mean about `138.242`, hit rate about `0.691`.
- `PEBBLES_S / PEBBLES_XL`, sum basis, window `100`, horizon `50`, threshold `3.0`, side `long_signal`, `n=160`, mean about `103.072`, hit rate about `0.662`.
- `PEBBLES_XS / PEBBLES_XL`, spread basis, window `200`, horizon `50`, threshold `3.0`, `n=222`, mean about `92.396`, hit rate about `0.599`.
- `PEBBLES_XS / PEBBLES_XL`, spread basis, window `100`, horizon `50`, threshold `2.5`, `n=723`, mean about `89.537`, hit rate about `0.629`.
- `PEBBLES_S / PEBBLES_XL`, spread basis, window `100`, horizon `50`, threshold `3.0`, side `short_signal`, `n=213`, mean about `84.268`, hit rate about `0.624`.

Interpretation:

`PEBBLES` relationships are the highest magnitude structures discovered. They are not all safe to trade together because multiple rules compete for `PEBBLES_XL` inventory. The cleanest single relationship is `PEBBLES_XS - PEBBLES_XL` high-spread reversion.

Critical correction:

Two-sided reversion lost badly for several pebbles relationships. One-sided direction matching was required. For example, high `PEBBLES_XS - PEBBLES_XL` means short `PEBBLES_XS` and long `PEBBLES_XL`; trading the opposite tail as if symmetric was harmful.

### Microchip Oval Volatility

Verified rows:

- `MICROCHIP_OVAL`, smooth `100`, horizon `200`, correlation with realized volatility about `0.962`, correlation with absolute move about `0.207`.
- Similar values held for smooth `50`, `100`, and `200` at horizon `200`.
- At horizon `100`, realized-vol correlations were still around `0.927` to `0.932`.

Interpretation:

`MICROCHIP_OVAL` is a volatility/regime signal more than a direct directional alpha. It predicts realized volatility strongly, but its relationship to signed movement is weak. It should be used to scale aggressiveness, avoid high-risk windows, or decide when passive versus aggressive execution is appropriate.

## PnL Translation Results

These results are included only to judge which research signals survived simple execution assumptions.

### One-Sided Priority Tests

Best one-sided tests:

- `PEB_XS_XL_spread_w100_z25_high`: total about `+105,321`, day 2 `+61,976`, day 3 `-4,254`, day 4 `+47,599`.
- `PEB_XS_XL_spread_w100_z3_high`: total about `+104,965`, day 2 `+62,509`, day 3 `-5,233`, day 4 `+47,689`.
- `PEB_S_XL_spread_w100_z3_high`: total about `+85,233`, day 2 `+52,584`, day 3 `-11,680`, day 4 `+44,329`.
- `SNACK_PIST_RASP_spread_w200_z3_high`: total about `+15,614`, positive all three days.
- `SNACK_PIST_RASP_spread_w100_z25_high`: total about `+11,954`, positive all three days.

Takeaway:

The highest-PnL structure is pebbles, but it has a day 3 weakness. The cleaner all-day structure is snackpack pair reversion, but at much lower magnitude.

### Combined Priority Strategy

The initial combined strategy joined:

- `PEBBLES_XS / PEBBLES_XL` high-spread reversion.
- `PEBBLES_S / PEBBLES_XL` high-spread reversion.
- `PEBBLES_S / PEBBLES_XL` low-sum reversion.
- `SNACKPACK_PISTACHIO / SNACKPACK_RASPBERRY` high-spread reversion.
- `SNACKPACK_RASPBERRY` pressure overlay.

Result:

- `EXP_priority_combined`: about `+123,049`, day 2 `+68,516`, day 3 `+2,967`, day 4 `+51,566`.

Interpretation:

The combined system improved total PnL and fixed day 3 relative to pure `PEBBLES_XS/XL`, but it increased turnover and introduced rule interactions. The main structural issue is `PEBBLES_XL` capacity. Multiple pebbles rules want the same product and same side, so standalone PnLs cannot simply be added.

### Combined Plus Independent Pairs

Adding independent secondary pairs improved the combined strategy:

- `COMBINED_PLUS_panel_sleep`: about `+146,672`, positive all days.
- `COMBINED_PLUS_choc_van`: about `+137,608`, positive all days.
- `COMBINED_PLUS_panel_sleep_choc_van`: about `+161,231`, day 2 `+88,346`, day 3 `+9,967`, day 4 `+62,918`, positive all days.

Interpretation:

The best additions were independent from the core pebbles cap conflict:

- `PANEL_4X4 / SLEEP_POD_COTTON` high-spread reversion.
- `SNACKPACK_CHOCOLATE / SNACKPACK_VANILLA` high-spread reversion.

These are more believable add-ons than additional `PEBBLES_XL`-dependent rules because they consume separate position limits.

### Stable Submit-Style Strategy

The current corrected `v2_stable_timestamp.py` no longer contains the mistaken `10_000` trading cadence. It uses rolling history warmup from feature windows and trades when thresholds fire.

Current corrected backtest:

- Total about `+17,924`.
- Day 2 about `+310`.
- Day 3 about `+6,970`.
- Day 4 about `+10,644`.
- Fills about `3,592`.

Interpretation:

After removing the accidental timestamp cadence, the strategy traded too frequently and lost most of the prior conservative edge. It is not currently the preferred submission candidate. If a stable submission is desired, it should use explicit lower turnover controls such as higher thresholds, lower targets, larger cooldowns, or persistent-horizon exits rather than a misunderstood timestamp cadence.

## False Positives And Lessons

### Raw Correlation Can Overstate Edges

The first broad pair scan produced many edges that looked too uniform across leaders and followers. This suggested common-factor contamination. The refined residualized scorer was necessary.

Lesson:

Prefer residual pair scores over raw pair scores when evaluating lead-lag. Raw pair correlation can simply be market mode.

### Two-Sided Mean Reversion Was Often Wrong

Several event studies validated only one tail. When converted into two-sided reversion, the opposite tail damaged PnL severely.

Lesson:

Every spread rule must preserve the verified side. Do not assume symmetry.

### Short-Horizon Pressure Does Not Automatically Equal PnL

`SNACKPACK` book pressure was one of the strongest confirmed feature families, but broad market-taking pressure systems were not great.

Lesson:

Pressure signals are better as overlays, execution leans, or passive quoting signals than as high-turnover takers.

### Lead-Lag Needs Execution Care

Residual lead-lag was statistically visible, especially in `SNACKPACK`, but simple follower takers lost money.

Lesson:

Lead-lag signals may still be valuable, but not in the naive translation. They need passive execution, larger thresholds, or use only when aligned with spread/constraint signals.

### Position Cap Interactions Matter

The pebbles signals looked additive in standalone backtests but were not truly additive because they competed for the same `PEBBLES_XL` inventory.

Lesson:

Portfolio construction matters. Independent pairs are cleaner additions than more rules sharing the same binding product cap.

## Current Strategy File Map

`Submissions_Derek/v2_priority_pair_reversion.py`:

- Generic one-sided/two-sided pair and basket reversion tester.
- Good for isolated candidate tests.
- Default is `PEBBLES_XS / PEBBLES_XL` high-spread reversion.

`Submissions_Derek/v2_priority_combined.py`:

- Aggressive combined strategy.
- Currently includes core pebbles, snackpack, `PANEL_4X4 / SLEEP_POD_COTTON`, and `SNACKPACK_CHOCOLATE / VANILLA`.
- Best tested variant in this session was effectively `COMBINED_PLUS_panel_sleep_choc_van`, about `+161,231` on days 2/3/4.
- Higher turnover and likely more overfit risk than isolated rules.

`Submissions_Derek/v2_secondary_signal_tests.py`:

- Generic research tester for pressure, lead-lag, and cluster-pair candidates.
- Useful for future grids.
- Not intended as a final submission.

`Submissions_Derek/v2_priority_allocator.py`:

- Attempted to allocate scarce product capacity rule-by-rule.
- Did not beat naive combined in testing.
- Result was about `+112,139`, below `+123,049` initial combined and below the later `+161,231` combined-plus variant.

`Submissions_Derek/v2_stable_timestamp.py`:

- Conservative rolling-window strategy file.
- Corrected to remove the mistaken `10_000` cadence assumption.
- Current corrected result is much lower, about `+17,924`.
- Needs more turnover control before being considered a stable candidate.

## Recommended Trust Ranking

Highest confidence relationships:

1. `PEBBLES_XS / PEBBLES_XL` high-spread reversion. Strong magnitude, strong event study, strong one-sided PnL, but day 3 risk exists.
2. `SNACKPACK_PISTACHIO / SNACKPACK_RASPBERRY` high-spread reversion. Lower magnitude, but positive all three days in one-sided tests.
3. `PANEL_4X4 / SLEEP_POD_COTTON` high-spread reversion. Independent secondary pair, positive all three days in secondary tests.
4. `SNACKPACK_CHOCOLATE / SNACKPACK_VANILLA` high-spread reversion. Weaker threshold hit rate, but positive all three days in secondary tests and useful as an independent add-on.

Medium confidence:

- `PEBBLES_S / PEBBLES_XL` high-spread and low-sum rules. Strong in isolation, but they interact with `PEBBLES_XL` and can conflict with other pebbles rules.
- `PEBBLES_XS / UV_VISOR_YELLOW` high-spread. Good total PnL but day 4 loss.
- `ROBOT_MOPPING / ROBOT_VACUUMING` low-spread. Good total PnL but day 2 loss.
- `MICROCHIP_CIRCLE / PEBBLES_S` low-spread. Good total PnL but not clean all-day.

Research-only for now:

- Broad `SNACKPACK` residual lead-lag.
- `MICROCHIP_SQUARE -> MICROCHIP_CIRCLE` lead-lag.
- `MICROCHIP_OVAL` directional use.

## Recommended Next Research Steps

1. Rebuild a stable candidate around the four highest-confidence relationships only:
  - `PEBBLES_XS / PEBBLES_XL`
  - `SNACKPACK_PISTACHIO / RASPBERRY`
  - `PANEL_4X4 / SLEEP_POD_COTTON`
  - `SNACKPACK_CHOCOLATE / VANILLA`
2. Add explicit turnover controls:
  - Longer cooldowns.
  - Larger entry thresholds.
  - Lower targets.
  - Exit only near z-score neutral or after fixed holding horizon.
3. Avoid adding more `PEBBLES_XL`-dependent rules unless an allocator is improved. `PEBBLES_XL` is already the binding shared capacity.
4. Use `MICROCHIP_OVAL` as a regime filter:
  - Increase thresholds during high predicted volatility.
  - Reduce targets or skip entries during extreme volatility.
  - Do not treat it as a direct long/short signal without more evidence.
5. Re-test single-day sensitivity:
  - Day 3 is the stress day for pebbles.
  - A robust candidate should not rely entirely on day 2 and day 4.
6. Revisit lead-lag only with passive or hybrid execution:
  - The signal may be real, but taker tests are poor.
  - A passive quoting lean may capture the information without paying spread every time.

## Bottom Line

The session found real structure. The most important research signal is not broad correlation; it is verified one-sided constraint reversion. `PEBBLES` provides the largest edge, `SNACKPACK` provides cleaner smaller edges, and selected independent cluster pairs improve diversification. The biggest overfit risks are two-sided assumptions, common-factor pair edges, and piling multiple rules into the same product cap.

If the goal is maximum tested PnL, the aggressive combined-plus family is best from this session. If the goal is a more stable submit-and-see file, the next step should be a cleaner low-turnover version of the four highest-confidence independent relationships, not the current corrected `v2_stable_timestamp.py` as-is.