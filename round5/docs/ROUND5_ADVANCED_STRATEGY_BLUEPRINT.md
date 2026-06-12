# Round 5 Advanced Strategy Blueprint

This document is a clear-cut, extensive blueprint for a **headless alpha calculator** in Round 5.

It is designed for handoff into a new chat/workstream and focuses on:

- **cluster-first modeling**
- **high-value correlators/features**
- **advanced model families (neural, spectral, state-space)**
- **robust validation and deployment constraints**
- **massive brute-force search over feature/model/hyperparameter combinations**

---

## 1) Problem Framing

Round 5 has 50 products, but current evidence indicates only a small subset exhibits strong exploitable structure.  
Therefore:

1. Treat market as **heterogeneous**, not one monolithic process.
2. Build alpha around **meaningful clusters first**.
3. Keep the rest as **execution/MM-only** unless new evidence appears.

Primary objective:

- Maximize risk-adjusted PnL under realistic fill assumptions and strict position limits.

Secondary objective:

- Minimize model fragility and overfit risk.

Operating mindset:

- This is **not** a visualizer-first project.
- This is a **compute-first brute-force engine** that systematically tries huge alpha combinations, scores them, and keeps only robust winners.
- **Time is not a hard constraint**: we optimize for search depth, robustness, and evidence quality over runtime.
- Overnight and multi-hour runs are expected and encouraged.

Optimization priority:

1. correctness and leakage safety
2. robustness across splits/regimes
3. breadth of candidate exploration
4. runtime speed (last priority)

---

## 2) Ground-Truth Cluster Focus

Current meaningful product structure (from in-repo analysis):

- Snackpack structure:
  - `SNACKPACK_PISTACHIO` ↔ `SNACKPACK_STRAWBERRY`: strong positive
  - `SNACKPACK_RASPBERRY`: mirror/opposite leg
  - `SNACKPACK_CHOCOLATE` ↔ `SNACKPACK_VANILLA`: strong anti-correlation
- Pebbles structure:
  - `PEBBLES_XL` opposite to `PEBBLES_{L,M,S,XS}`

Working rule:

- Do **not** apply complex alpha search uniformly across all 50 products.
- Prioritize the above clusters for advanced fitting.

---

## 3) Correlator Catalog (Important Features to Include)

The list below is intentionally broad. Build feature families, then prune with ablation.

## A) Price/Return Correlators

- **Instant returns**: log/mid returns at lags `1,3,5,10,25,50,100`
- **Cumulative returns**: rolling cumulative return windows
- **Residual returns**: product return minus cluster mean return
- **Lead-lag returns**: `r_i[t]` vs `r_j[t+k]` for signed lags
- **Autocorrelation profiles**: rolling ACF/PACF summaries
- **Pair spread**:
  - raw spread `P_a - beta*P_b`
  - normalized spread z-score

## B) Order Book Correlators

- **Level-1 imbalance**: `(bid_vol1 - ask_vol1)/(bid_vol1 + ask_vol1)`
- **Multi-level imbalance**: weighted L1/L2/L3 imbalance
- **Depth slope/shape**: near vs far level depth ratios
- **Micro-price** and **micro-price delta**
- **Spread state**:
  - absolute spread
  - relative spread (`spread / mid`)
  - spread regime transitions (narrow→wide, wide→narrow)
- **Queue pressure proxies**:
  - bid queue growth rate
  - ask queue growth rate
  - cancellation pressure proxies from volume changes

## C) Trade Flow Correlators

- **Signed volume proxy** (aggressor approximation)
- **Trade imbalance**: buy-initiated vs sell-initiated ratio
- **Trade intensity**: trades per interval
- **Average trade size / burstiness**
- **Trade-to-book pressure**:
  - traded volume / available top-book depth
- **Short-term impact proxy**:
  - post-trade drift conditioned on print direction

## D) Trend/Mean-Reversion Correlators

- **SMA/EMA families** (short, medium, long)
- **EMA slope and curvature**
- **MACD-like components** (fast EMA - slow EMA, signal line)
- **RSI-like bounded oscillator**
- **Bollinger-style band position**
- **Distance to rolling VWAP-like proxy** (if volume proxy available)
- **Half-life proxy** for spread mean reversion

## E) Volatility/Regime Correlators

- **Realized volatility** (multiple windows)
- **Vol-of-vol**
- **Jump/outlier flags** (tail event indicators)
- **Regime indicators**:
  - low/high vol
  - trend/mean-revert classification
  - spread-compression vs expansion

## F) Cluster-Relative Correlators (Highest Priority)

- **Cluster mean residual**: `x_i - mean(cluster_ex_i)`
- **Mirror-leg residual**: sign-corrected residual for anti-correlated legs
- **Rolling beta residual**:
  - estimate beta to cluster factor
  - use residual as alpha driver
- **Intra-cluster rank**:
  - rank by residual, spread, pressure
- **Cross-cluster spread** (when comparing clusters in research mode)

## G) Spectral/Transform Correlators

- **FFT band energy** over rolling windows
- **STFT energy drift** (time-local frequency content)
- **Wavelet coefficients** (multi-scale decomposition)
- **Hilbert amplitude/phase proxies** (careful with instability)
- **Spectral entropy** for regime quality gating

## H) Nonlinear Interaction Correlators

- imbalance × spread
- imbalance × volatility
- residual × regime
- depth-skew × trend slope
- signed-flow × spread-state transition

---

## 4) Model Families to Evaluate

Use a staged approach (simple → complex) with strict acceptance criteria.

## Stage 0 (reference baselines)

- Linear z-score mean reversion
- Ridge/Lasso with engineered features
- Tree model baseline (LightGBM/XGBoost style, if allowed in research env)

## Stage 1 (state-space/classical advanced)

- Kalman filter for dynamic hedge ratio and spread mean
- HMM / Markov switching for regime gating
- Cointegration tests + rolling hedge updates

## Stage 2 (sequence neural models)

- TCN / 1D-CNN (preferred first neural baseline)
- GRU/LSTM (compare vs TCN)
- Small Transformer encoder (only if data supports it)

## Stage 3 (spectral + hybrid)

- Spectral features into TCN/MLP head
- Regime-conditioned ensemble:
  - model A for mean-revert regime
  - model B for momentum regime

---

## 5) Label Design and Targets

Define targets at multiple horizons:

- `y_h = mid[t+h] - mid[t]` (raw delta)
- `sign(y_h)` (direction)
- residual target against cluster factor

For execution-aware modeling:

- target net edge after estimated crossing/spread costs
- binary label: `edge > threshold`

---

## 6) Validation Protocol (Non-Negotiable)

- **Purged walk-forward** CV (avoid overlap leakage between feature and target windows)
- **Day-based holdout** (time-safe splits)
- **Ablation tests** by feature family
- **Stability checks** by:
  - day
  - regime
  - product within cluster
- **Capacity checks**:
  - turnover
  - max inventory swings
  - sensitivity to fill assumptions

Required reporting per candidate:

- IC / rank-IC
- hit rate
- mean edge
- PnL contribution in backtest
- drawdown / tail behavior
- turnover-adjusted efficiency

---

## 7) Risk and Execution Constraints

Any advanced alpha must pass:

- strict position limit compliance
- bounded order-rate behavior
- fallback to safe MM when confidence low
- no dependence on unavailable features at inference time
- inference-time budget compatible with live loop

Practical safeguards:

- confidence gating
- regime gating
- kill-switch on unstable residuals

---

## 8) Research Pipeline (Recommended Implementation Order)

1. Build unified research dataset (`product`, `timestamp`, all correlators, multi-horizon labels).
2. Run correlation + MI + interaction screening.
3. Train Stage 0/1 baselines on cluster-only data.
4. Add spectral feature block and compare uplift.
5. Train compact sequence model (TCN first).
6. Add meta-label gate (trade/no-trade filter).
7. Integrate best model into strategy with conservative sizing.
8. Backtest with ablations and stress checks.

---

## 8.1) Massive Brute-Forcer Spec (Headless)

This section defines the "massive calculator" mode.

### A) Search dimensions

Generate candidates from cartesian products of:

- product scope:
  - cluster-level
  - pair-level
  - mirror-adjusted pair-level
- feature sets:
  - single family
  - pairwise family combos
  - selected nonlinear interaction bundles
- transforms:
  - raw
  - z-scored
  - winsorized
  - differenced
  - spectral-augmented
- model families:
  - linear
  - tree
  - state-space
  - sequence neural
- hyperparameters:
  - horizon
  - window lengths
  - regularization
  - thresholding rules
  - gating rules

### B) Compute strategy

- Use batched jobs keyed by `candidate_id`.
- Cache intermediate tensors/tables aggressively.
- Early-stop low-potential candidates using cheap proxy metrics.
- Run heavy models only on survivors from cheap stages.
- Prefer long overnight sweeps with checkpointing/resume support.
- Log progress and partial leaderboards continuously so long runs are inspectable.

### C) Candidate lifecycle

1. **Generate**
2. **Quick screen** (fast IC/hit-rate proxies)
3. **Robust validation** (purged walk-forward)
4. **Backtest integration score**
5. **Rank and archive**

### D) Score function (single scalar)

Use weighted objective, e.g.:

`score = w1*oos_pnl + w2*ic - w3*drawdown - w4*turnover_penalty - w5*instability`

Hard constraints before score:

- minimum OOS stability
- maximum drawdown cap
- minimum trade count
- capacity sanity

### E) Required outputs per run

- `leaderboard.csv` (top candidates with full metrics)
- `rejections.csv` (failed candidates + reason)
- `artifacts/<candidate_id>/`:
  - config
  - metrics by split
  - feature importances/diagnostics
  - reproducibility hash
- `run_manifest.json`:
  - start/end time
  - dataset/version hashes
  - total candidates generated/screened/validated
  - checkpoint references for resume

### F) Anti-overfit controls

- strict temporal splits
- purging/embargo
- no test leakage in feature normalization
- re-check top candidates on untouched final slice

### G) Promotion rule to live strategy

Promote only candidates that beat baseline on:

- OOS PnL
- drawdown-adjusted return
- robustness across splits
- execution realism sensitivity

---

## 9) “Did We Miss Anything?” Checklist

Before moving to production, verify:

- [ ] Mirror-leg sign handling is correct.
- [ ] Feature leakage is eliminated.
- [ ] Model outperforms baseline on unseen days.
- [ ] Performance not concentrated in one day only.
- [ ] Alpha survives tighter execution assumptions.
- [ ] Feature importance is interpretable enough for debugging.
- [ ] Failure mode documented (when model should NOT trade).
- [ ] Fallback MM behavior works when model confidence is low.

---

## 10) Suggested Success Criteria

A model/alpha is accepted only if:

1. Out-of-sample PnL > current baseline,
2. Drawdown does not worsen materially,
3. Turnover-adjusted efficiency improves,
4. Edge remains positive under stricter slippage/fill assumptions,
5. Results are stable across multiple day splits.

---

## 11) Final Principle

For Round 5, edge is likely **cluster-local and regime-dependent**, not universal.

The winning approach is:

- strong baseline execution/MM engine
- small set of validated cluster alphas
- robust gating/risk controls
- disciplined validation over flashy complexity

