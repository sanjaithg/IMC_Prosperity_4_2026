# Round 4 — Handover

Status as of **2026-04-27**. This document is the successor's quickstart: what the round is, what we discovered, what tooling exists, and what still needs to be built into the final submission.

---

## 1. Round at a glance

Two parallel tasks:

1. **Algorithmic trading** — same 12 instruments as Round 3:
   - `HYDROGEL_PACK` (limit ±200)
   - `VELVETFRUIT_EXTRACT` (limit ±200) — the underlying for the voucher chain
   - 10 European-style call vouchers `VEV_4000` … `VEV_6500` (limit ±300 each)
     - TTE on Round 4: **4d / 3d / 2d** for days 1 / 2 / 3
   - **New twist**: `Trade.buyer` and `Trade.seller` are no longer `None` — they carry counterparty IDs (`Mark 01`, `Mark 14`, `Mark 22`, `Mark 38`, `Mark 49`, `Mark 55`, `Mark 67`, plus `SUBMISSION` for our own).
2. **Manual trading** — Aether Crystal + a portfolio of option contracts. One-shot submission, completely separate from the algo container. **Not addressed in this codebase.**

Position limits are unchanged from Round 3.

## 2. Market data semantics — IMPORTANT

This was confirmed against the brief and matters for strategy design:

- `state.market_trades[product]` and `state.own_trades[product]` contain trades that happened **since the last `TradingState` came in** (i.e., during the *previous* iteration, not the current one).
- Counterparty IDs (`Mark XX`) appear **only on completed trades**, never on resting `order_depth` quotes.
- `order_depth` shows **aggregated price levels**; you cannot identify which bot is quoting where.
- Order matching is instantaneous — no other bot is faster than the player's order during the same iteration.

**Consequence:** any Mark-conditioned signal has at minimum a one-tick reaction latency. When we see a Mark 67 buy, that trade happened at *t−1*; the earliest we can act on it is *t*, and the meaningful holding window is roughly k=2 to k=20.

## 3. Counterparty profiles (the seven Marks)

Verified across all 3 days and 12 products. There are exactly 7 disclosed counterparties.

| Mark | Type | Direction | Aggression | Products | TOTAL PnL |
|---|---|---|---|---|---:|
| **Mark 14** | Spot market maker | balanced | passive (0% cross) | HYDROGEL, VELVET (68.7% of all spot fills) | **+42,205** |
| **Mark 67** | **Insider ("Olivia")** | buy-only | aggressive taker | VELVET only | **+27,261** |
| Mark 01 | Passive accumulator | buyer | passive | VELVET + far-OTM vouchers | +10,100 |
| Mark 55 | Spread-payer | balanced | aggressive (100% cross) | VELVET only | −13,204 |
| Mark 49 | Block seller | seller | passive | VELVET only | −15,346 |
| Mark 22 | OTM-call seller | seller | aggressive (91% cross) | OTM vouchers (5300/5400/5500/6000/6500) | −17,395 |
| Mark 38 | Spread-payer | balanced | aggressive (100% cross) | HYDROGEL only | −33,622 |

### The dealer-book hypothesis (verified)

Mark 22 + Mark 01 + Mark 67 + Mark 49 are not independent traders. Strike-by-strike volume shows:

```
VEV_5500: 1042 of 1069 sold by Mark 22 went directly to Mark 01 (97%)
VEV_6000: 1105 of 1105 (100%)
VEV_6500: 1105 of 1105 (100%)
```

This is **one dealer's two legs**:
- Mark 22 = short-call book
- Mark 01 = long-call book
- Mark 67 = the dynamic delta-hedger that buys VELVET as the dealer's short-call deltas grow
- Mark 49 = the periodic spot inventory unwinder

This explains the otherwise puzzling fact that Mark 67's buys *and* Mark 49's sells both precede +2 spot moves: they're hedging the same book in opposite legs, and the book is the source of the price action.

## 4. Mark 67 = "Olivia" — the insider

**The headline finding.** Mark 67's trade-by-trade scoreboard:

```
n_trades         165
qty             1510  (100% buy-only)
TOTAL PnL    +27,261  (~$165 per trade)
cross_rate      99.4% (pays the spread on every fill)
good_rate_k1    95.8% ← informational, ~12σ above random for n=165
good_rate_k5    83.0%
good_rate_k20   64.8%
good_rate_k50   58.8%  (decays exactly like an information signal)
```

**The clinching evidence:**
- A pure spread-taker (cross_rate 99.4%) cannot make $27k unless they have predictive info — they're paying half the L1 spread on every fill.
- 95.8% directional accuracy at k=1 over 165 trades is statistically impossible by chance.
- The good-rate decays smoothly with horizon — exactly what an information signal looks like (vs. a delta-hedger, whose forward returns would be noisy).
- They trade **only VELVETFRUIT_EXTRACT** and **only buy** — directional bullish info on one specific instrument.

This is the Round 4 analog of Olivia from Prosperity 3.

## 5. Spread distribution signal (HYDROGEL_PACK biggest)

Found via [spread_analysis.ipynb](spread_analysis.ipynb). For every product the L1 spread is heavily concentrated on a single mode value. Forward returns are conditional on the spread being at non-mode values — a clean tradeable signal.

**HYDROGEL_PACK** (mode spread = 16):

| spread | mean fwd k=1 | mean fwd k=50 | n |
|---|---:|---:|---:|
| 15 (mode−1) | +0.184 | **+3.134** | 767 |
| 16 (mode) | −0.009 | +0.035 | 27,751 |
| 17 (mode+1) | +0.006 | **−3.582** | 494 |

→ HYDROGEL spread = 15 ⇒ **lean LONG**, expect +3 over next 50 ticks.
→ HYDROGEL spread = 17 ⇒ **lean SHORT**, expect −3.6 over next 50 ticks.

Same shape (smaller magnitude) on `VEV_4500` and `VEV_5000`. `VELVETFRUIT_EXTRACT` shows a different bearish-on-tight pattern.

The full conditional table is in the executed notebook ([spread_analysis_executed.ipynb](spread_analysis_executed.ipynb)).

## 6. Tooling built

Three independent Dash dashboards plus a notebook. They all share `data_visualizer.py` as the single source of truth for `load_data`, BS helpers, and counterparty constants — none of the new files mutate it.

| Port | File | Purpose |
|---:|---|---|
| 8050 | `data_visualizer.py` | Base market explorer (price, depth, smile, spread filter). Original Round-3 file extended to surface Round 4 counterparty info. |
| 8060 | `datavisualiser_featured.py` | 16 signal cards: counterparty flow, dealer imbalance, IV smile/skew/curvature, spread distribution, per-strike IV, Greeks. |
| 8090 | `bot_performance_visualizer.py` | Per-Mark PnL, trade table, position curves, insider scoreboard with `good %k` columns. **Use this to confirm Mark 67.** |
| —    | `spread_analysis.ipynb` / `spread_analysis_executed.ipynb` | The HYDROGEL spread signal and per-product forward-return tables. |

Launch any of them from `Round 4/`:

```bash
./venv/bin/python data_visualizer.py            --data Dataset/ROUND_4 --port 8050
./venv/bin/python datavisualiser_featured.py    --data Dataset/ROUND_4 --port 8060
./venv/bin/python bot_performance_visualizer.py --data Dataset/ROUND_4 --port 8090
./venv/bin/jupyter notebook spread_analysis_executed.ipynb
```

A second helper file `round4_log_visualizer.py` exists from an earlier iteration and emits static HTML per product into `viz_out/`. Lower priority; mostly superseded by 8050.

## 7. Files in this directory

```
Round 4/
├── HANDOVER.md                       this file
├── data_visualizer.py                base explorer (port 8050)
├── datavisualiser_featured.py        signal dashboard (port 8060)
├── bot_performance_visualizer.py     bot/Olivia scanner (port 8090)
├── round4_log_visualizer.py          static HTML emitter (older)
├── spread_analysis.ipynb             spread signal hunt (source)
├── spread_analysis_executed.ipynb    same, with all outputs embedded (3.5 MB)
├── backtester.py                     Round 3 backtester, NOT yet rewired for Round 4
├── backtest_log_visualizer.py        plots backtester output csvs
├── viz_out/                          static HTMLs from round4_log_visualizer.py
├── Dataset/ROUND_4/                  prices_round_4_day_{1,2,3}.csv + trades_*.csv
└── venv/                             Python 3.12 env: pandas, numpy, plotly, dash, jupyter
```

## 8. What is NOT yet done

These are the open items. They are listed in priority order — the first two are where the realized edge actually lives.

### a. VELVETFRUIT_EXTRACT module in the final submission (NOT in repo)

There is **no VELVET strategy in our final algo**. This is the largest gap. Three signals to combine:

1. **Olivia follower (Mark 67)**. When `Mark 67` appears as buyer in `state.market_trades['VELVETFRUIT_EXTRACT']`, immediately lift offers up to position limit and hold ~10–20 ticks. Their good-rate at k=20 is still 64.8% which is alpha after spread cost.
2. **Market-make against Mark 55**. They cross 100% of the time on VELVET, balanced both ways. Quote inside Mark 14's L1 when flat; step back when loaded. Pure spread-collect EV.
3. **Skew on Mark 22 / Mark 49 flow.** When Mark 22 just dumped vouchers (their dealer's short-call book just grew), Mark 67 is about to buy spot — pre-position by leaning offer up.

### b. HYDROGEL spread signal in the algo

The spread = 15 / spread = 17 directional bias is a clean, isolated signal that doesn't conflict with anything else in the book. Implementation is ~30 lines: read `order_depth.bid_orders` and `order_depth.sell_orders`, compute current spread, lean long if 15 / short if 17, target a small position (~30 contracts), exit at k=20.

### c. Backtester for Round 4

`backtester.py` was copied from Round 3 and still has Round 3 defaults. Needs:
- `--data-dir Dataset/ROUND_4` default
- `--days 1 2 3` default
- `DAY_TO_TTE = {1: 4, 2: 3, 3: 2}` (currently `{0: 8, 1: 7, 2: 6}`)
- `datamodel.py` not present in `Round 4/` — copy from `/home/hillman/IMC_India1/datamodel.py` or symlink
- Counterparty-aware fill simulation (currently doesn't synthesize buyer/seller for the trader's fills — needs at minimum `SUBMISSION` on the side of own trades)

Until this is done, we can't measure the VELVET / HYDROGEL signals' realized PnL on historical data.

### d. Manual trading (Aether Crystal options)

Completely outside this directory. The user mentioned exotic option contracts on the Aether Crystal — needs separate analysis once the manual brief is published.

## 9. Recommended next moves

1. **Fix the backtester** (item 8c) — small mechanical work.
2. **Drop the HYDROGEL spread module in** (item 8b) — smallest, cleanest, easiest to validate.
3. **Add the Mark-67-follower VELVET module** — biggest expected edge. Sized small at first because 165 trades over 3 days means it fires only ~55 times per day on average; carry can be a problem.
4. **Add VELVET market-making against Mark 55** — independent revenue stream from spread capture.
5. Re-run the backtester after each addition; check own-fill PnL by counterparty (already supported by `bot_performance_visualizer.py` if backtester emits `fills.csv`).

## 10. Things to watch out for

- The dealer book bots (Mark 01 / 22 / 49 / 67) move together. If you fade Mark 67 you are also implicitly fading the dealer's hedge of Mark 22's short-call sales. Don't double-count the same trade.
- Mark 22 trades vouchers at posted prices like `0.0` and `1.0` for far-OTM strikes. Those are internal book transfers, NOT real market prices. Do not treat them as fair value when computing voucher IV — restrict the IV fit to strikes where Mark 22↔Mark 01 trades happen with size and the prices are non-trivial (5300 / 5400 / 5500 are the safe ones; 5200 if you're greedy).
- Voucher TTE is short (4 → 2 days). Theta decay is steep. If you go long any voucher, plan on flattening before EOD on day 3 or accept the assignment economics at expiry.
- The good-rate decay we measured for Mark 67 is on **historical** data; in live trading the latency is one tick higher than the measurement, so the realistic edge is at k=2 onwards (still 80%+ at k=5).

## 11. Quick re-verification commands

```bash
cd "Round 4"

# Sanity-check the dataset is intact
./venv/bin/python -c "
from data_visualizer import load_data
p, t = load_data('Dataset/ROUND_4')
print('prices', len(p), 'trades', len(t))
print('marks:', sorted(set(t['buyer']).union(t['seller']) - {'UNKNOWN'}))
"

# Re-confirm Mark 67's insider scoreboard from raw data
./venv/bin/python -c "
import importlib.util, sys; sys.path.insert(0, '.')
spec = importlib.util.spec_from_file_location('bp', 'bot_performance_visualizer.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
p, t = m.load_data('Dataset/ROUND_4')
e = m._enrich_trades(p, t)
s, _ = m.compute_pnl_per_mark(p, e)
print(s.to_string(index=False))
"

# Re-execute the spread notebook
./venv/bin/jupyter nbconvert --to notebook --execute spread_analysis.ipynb --output spread_analysis_executed.ipynb --ExecutePreprocessor.timeout=300
```

If any of those produce different numbers than what's documented above, **the dataset has changed** and every conclusion in this doc needs to be re-run before trusting it.

---

*End of handover.*
