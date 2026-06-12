# Round 5 Ground Truth and Alpha Plan

This document is the current source of truth for Round 5 research and execution.

Execution stance:

- This is a **compute-first research program**, not a visualization-first workflow.
- We explicitly allow **overnight / multi-hour runs**.
- Runtime is secondary to robustness and leakage-safe validation.

Priority order:

1. correctness and data hygiene
2. out-of-sample robustness
3. breadth of candidate exploration
4. runtime speed

## 1) Ground Truth (What Data Actually Says)

Based on current analysis over days `2,3,4` (`prices_round_5_day_*.csv`):

- Only a small subset of products show meaningful pairwise structure.
- Strong relationships:
  - `SNACKPACK_RASPBERRY` vs `SNACKPACK_STRAWBERRY`: `-0.923`
  - `SNACKPACK_CHOCOLATE` vs `SNACKPACK_VANILLA`: `-0.915`
  - `SNACKPACK_PISTACHIO` vs `SNACKPACK_STRAWBERRY`: `+0.913`
  - `SNACKPACK_PISTACHIO` vs `SNACKPACK_RASPBERRY`: `-0.831`
  - `PEBBLES_XL` vs `PEBBLES_{L,M,S,XS}`: roughly `-0.48` to `-0.51`
- At threshold `|corr| > 0.10`:
  - meaningful pairs: `8`
  - effectively uncorrelated products: `40/50`

### Meaningful clusters in production now

- `SNACKPACK_PISTACHIO` cluster:
  - `SNACKPACK_PISTACHIO`, `SNACKPACK_STRAWBERRY`, `SNACKPACK_RASPBERRY`
- `SNACKPACK_CHOCOLATE` cluster:
  - `SNACKPACK_CHOCOLATE`, `SNACKPACK_VANILLA`
- `PEBBLES_XL` cluster:
  - `PEBBLES_XL`, `PEBBLES_L`, `PEBBLES_M`, `PEBBLES_S`, `PEBBLES_XS`

## 2) What We Should Stop Doing

- Stop trusting unsupervised cluster assignments with tiny average intra-cluster correlation.
- Stop deploying one "global alpha" uniformly to all 50 products.
- Stop evaluating only raw correlation magnitudes without out-of-sample checks.

## 3) Feature Evidence Snapshot (from `analysis/feature_scan.py`)

Across all products, average predictive relationships are weak and noisy:

- `imbalance` strongest at short horizons (lag 1): mean corr around `+0.045`
- `rel_depth` rises at long horizons (lag 100): mean corr around `+0.043`
- `rel_spread` and `volatility` are weak on average
- Directional accuracy is mostly near random (`~48%` to `51%`)

Interpretation:

- There is no universal alpha that is strong everywhere.
- We need cluster-specific and product-specific alpha selection.

## 4) Strategy Architecture Going Forward

### Layer A: Base MM engine (all products)

- Keep robust market-making baseline:
  - fair value from EMA mid + imbalance tilt
  - inventory-aware quote skew
  - strict position-limit protection

### Layer B: Cluster alpha overlays (only meaningful clusters)

- For each meaningful cluster, build dedicated alpha models:
  - spread/z-score mean reversion (cluster-relative)
  - mirror-leg inversion handling (e.g., raspberry, pebbles_xl)
  - rolling beta/hedge-ratio style signals

### Layer C: Alpha gating

- Enable alpha only when quality filters pass:
  - `|signal|` threshold
  - rolling stability threshold
  - minimum recent hit-rate / Sharpe-like filter

## 5) Workstreams (How We Tackle the Large Scope)

## W1. Data and labels

- Build a unified per-tick research table with:
  - book state (spread, depth, imbalance, pressure)
  - trade prints (signed volume proxy, aggressor approximation)
  - trend features (SMA/EMA slopes, momentum, regime flags)
  - target labels for multiple horizons (`1,3,5,10,25,50,100`)

Deliverable:

- `analysis/alpha_dataset.py` + parquet/csv outputs by day and product.

## W2. Cluster-specific alpha library

- For each of the 3 clusters, evaluate:
  - mean-reversion z-score variants
  - divergence/convergence speed signals
  - imbalance + trend interactions
  - volatility-conditioned signals

Deliverable:

- `analysis/alpha_library.py` producing ranked alpha candidates per cluster.

## W3. Validation protocol

- Train/test split by day (time-safe), not random row split.
- Use purged/embargoed splits where horizon overlap can leak.
- Report:
  - corr(IC), hit-rate, turnover, PnL contribution in backtest
  - stability across days
  - sensitivity to thresholds

Deliverable:

- `analysis/alpha_validation_report.md` (auto-generated summary).

## W4. Trading integration

- Integrate only top validated alpha(s) into `Submissions_Derek/v1_cluster_mm.py`.
- Preserve existing safe MM behavior when alpha confidence is low.
- Add compact diagnostic logging for alpha decisions.

Deliverable:

- `Submissions_Derek/v2_cluster_mm.py` with feature flags and fallback path.

## 6) Execution Cadence (Practical)

- Step 1: lock ground-truth clusters (done)
- Step 2: generate alpha dataset tables
- Step 3: rank and prune candidate alphas per cluster
- Step 4: backtest shortlist with ablations
- Step 5: integrate top alpha(s), re-backtest, finalize submission

Overnight operation mode:

- Run broad candidate sweeps with checkpoint/resume.
- Persist partial leaderboards continuously.
- Store run manifests (`dataset hash`, `config hash`, `candidate counts`, `elapsed time`) for reproducibility.

Rule:

- No alpha enters production unless it beats baseline in out-of-sample backtest.

## 7) Current Repository Status

- `analysis/cluster_visualizer.py` now focuses on the 3 meaningful clusters and hides weak/noisy correlations.
- `Submissions_Derek/v1_cluster_mm.py` now uses meaningful cluster structure with mirror handling.
- This playbook becomes the checklist for all next research tasks.

## 8) Open Gap: "Previous year strategies"

This repository currently does not contain historical Round 1-4 submission strategy files to mine directly.
If you want that study included, we should add those strategy files (or links) and run a dedicated extraction pass:

- common alpha motifs
- risk control patterns that survived
- failures to avoid under Round 5 microstructure

