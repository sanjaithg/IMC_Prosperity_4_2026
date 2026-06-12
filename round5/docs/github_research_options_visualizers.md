# GitHub Research: Prosperity Options, Vouchers, and Visualizers

This note is a GitHub-first research summary for Round 3 preparation.

The goal is not to copy anyone's strategy. The goal is to extract:

- what prior Prosperity rounds did with option-like products
- what public teams found useful when trading vouchers / coupons
- what open-source tooling patterns are worth borrowing for our own dashboard

This document is based primarily on public GitHub repositories and README writeups, not on the official IMC wiki.

## Scope

The most relevant public precedents are:

1. **Prosperity 2 (2024), Round 4**
   `COCONUT_COUPON` as an option-like product on `COCONUT`
2. **Prosperity 3 (2025), Round 3**
   `VOLCANIC_ROCK_VOUCHER_*` products as voucher / call-option analogs
3. **Prosperity 2 / 3 / 4 tooling**
   visualizers, backtesters, dashboards, and logging workflows built by public teams

## Highest-signal repos

### Option / voucher analogs

- `ericcccsliu/imc-prosperity-2`
  Strong Prosperity 2 writeup with a clean summary of `COCONUT_COUPON`
- `jmerle/imc-prosperity-2`
  Strong tooling repo plus a useful round summary from a top solo finisher
- `TimoDiehm/imc-prosperity-3`
  Very strong Prosperity 3 writeup, including dashboard philosophy and options round details
- `chrispyroberts/imc-prosperity-3`
  Strong writeup on volatility smile modeling and hedging trade-offs
- `angus4718/imc-prosperity-3-public`
  Concise but useful options-trading summary with explicit smile-fit coefficients
- `CarterT27/imc-prosperity-3`
  Strong high-level summary of how a top team approached voucher pricing

### Tooling / dashboard repos

- `jmerle/imc-prosperity-2-backtester`
- `jmerle/imc-prosperity-3-backtester`
- `nabayansaha/imc-prosperity-4-backtester`
- `chrispyroberts/imc-prosperity-4`

## What prior years say about option-like products

## Prosperity 2 Round 4: Coconut Coupon

This is the clearest public "options analog" from Prosperity 2.

From the public writeups:

- `COCONUT_COUPON` was treated as a **10,000 strike call option** on `COCONUT`
- it had a **long time to expiry** relative to the Prosperity 3 vouchers
- teams used **Black-Scholes**
- teams explicitly looked at **implied volatility**
- teams thought in terms of **delta**, **vega**, and whether to hedge

### Key lessons from public teams

From `ericcccsliu/imc-prosperity-2`:

- the coupon was near-the-money because `COCONUT` traded around `10,000`
- their implied vol estimate oscillated around roughly `16%`
- they computed delta and tried to hedge with the underlying
- because the coupon / underlying limits were asymmetric, they could not fully hedge the max coupon position
- they still chose to run some residual delta risk because they believed the vol exposure remained positive EV

From `jmerle/imc-prosperity-2`:

- Round 4 clearly pushed many players toward **Black-Scholes pricing**
- he explicitly says the product was "option-like"
- he tuned Black-Scholes parameters based on backtests
- he made money in coupons while losing money in the underlying directional leg, which is a useful warning against mixing "option pricing edge" with "underlying directional conviction"

### Why this matters for us

Prosperity 2 suggests that once IMC introduces an option-like product, public teams usually move quickly toward:

- fair-value modeling in option space
- implied vol estimation
- cross-checking market price versus model price
- deciding whether hedging is worth the cost

That is directly relevant to `VEV_*`.

## Prosperity 3 Round 3: Volcanic Rock Vouchers

This is the closest public analog to our current Round 3.

Across multiple public repos, the setup is described as:

- one underlying: `VOLCANIC_ROCK`
- multiple vouchers with different strikes
- vouchers behaving like **call options**
- time to expiry decreasing as rounds progressed

Publicly cited strikes included:

- `9500`
- `9750`
- `10000`
- `10250`
- `10500`

Public descriptions also say:

- vouchers started with **7 trading days to expiry**
- by the final round they had **2 days left**

This is structurally very similar to our Round 3:

- one underlying
- many strike-specific vouchers
- shrinking TTE
- separate books per option symbol

## Common themes in Prosperity 3 option writeups

### 1. Teams treated vouchers as actual call-option analogs

This sounds obvious, but it matters.

Top public teams did not frame the vouchers as "just weird extra symbols." They treated them as:

- calls on the underlying
- strike-dependent instruments
- time-sensitive instruments
- instruments whose pricing should be studied through implied vol and moneyness

### 2. Implied volatility was a central lens

Several public teams focused on:

- computing implied vol from market prices
- plotting implied vol versus strike or moneyness
- fitting some form of volatility smile

This appears again and again in the writeups.

### 3. Moneyness mattered more than raw strike alone

The public Prosperity 3 writeups often shifted from "strike K" to a normalized representation such as:

- relative strike
- moneyness
- moneyness scaled by time

The point is that teams were not just comparing `voucher_9500` to `voucher_10500` by raw price. They were trying to put all vouchers on the same option-surface view.

### 4. Hedging was considered, but not blindly accepted

Many teams considered delta hedging.

But the public writeups also show why hedging was not automatically optimal:

- hedging costs spreads
- position limits can prevent full hedge ratios
- gamma / theta realities depend on how far away expiry is
- a theoretically cleaner book can still produce lower realized PnL

This is a very important practical lesson.

## Repo-by-repo findings

## `TimoDiehm/imc-prosperity-3`

This repo is one of the strongest public writeups for understanding how top teams thought about the options round.

### Main options findings

- the team explicitly calls Volcanic Rock Vouchers **call-option analogs**
- they emphasize that without basic option theory, especially implied vol and pricing models, it would be hard to build a strong strategy
- they mention a wiki hint pushing people toward building a **volatility smile**
- they describe using **IV scalping**
- they also mention a **lightweight mean reversion component**

### Practical takeaway

The key takeaway is that they did not rely on one single modeling lens.

Their public explanation suggests a blend of:

- option-surface mispricing
- IV deviation / mean reversion
- some underlying or deep ITM mean reversion logic

This is useful because it implies the dashboard should support both:

- option-chain / smile analysis
- ordinary time-series analysis on the underlying and select vouchers

## `chrispyroberts/imc-prosperity-3`

This repo gives one of the cleanest public explanations of a volatility-smile workflow.

### Main options findings

- the team modeled vouchers as European-option-like contracts
- they computed moneyness as a function of strike, underlying price, and TTE
- they fit a **quadratic curve** to implied vol versus moneyness
- they initially used that fitted smile as a fair-IV model
- they then found a short rolling-window mean of IV worked even better in backtests
- they built an aggressive market maker around that fair-IV view
- they initially hedged, then analyzed the cost of hedging carefully
- they found hedging spread costs were very large
- they estimated the realistic downside of going unhedged and decided the trade-off was acceptable

### Practical takeaway

This repo strongly argues that a useful options dashboard should support:

- implied vol time series
- implied vol by strike
- moneyness transforms
- fair-IV versus market-IV deviation
- some way to inspect hedge cost or spread cost

It also shows that "hedge everything" is not automatically the right answer in these simulated markets.

## `angus4718/imc-prosperity-3-public`

This repo is compact but very useful because it describes a concrete smile-fitting workflow.

### Main options findings

- the team generated **separate volatility smiles for bid and ask**
- they fit **quadratic coefficients** for ask and bid smiles
- they used those fitted smiles directly in the trading strategy

### Practical takeaway

This is a strong hint that the dashboard should not only show a single mid-IV smile.

It would be better if our dashboard can support:

- mid smile
- bid smile
- ask smile

because the trading decision is often made off the executable side, not just off the midpoint.

## `CarterT27/imc-prosperity-3`

This repo gives a helpful high-level summary from another strong team.

### Main options findings

- they used **Black-Scholes**
- they computed implied volatility from market prices
- they maintained a **rolling volatility window**
- they looked for **arbitrage opportunities across vouchers with different strikes**
- they also considered using the average IV across vouchers when judging the underlying
- they experimented with a fitted volatility surface and with delta hedging, but those were harder to implement reliably

### Practical takeaway

This reinforces that cross-strike analysis matters.

The dashboard should not just treat each `VEV_*` as a standalone chart. It should let us inspect:

- all vouchers together
- relative price spacing between strikes
- relative IV spacing between strikes

## `jmerle/imc-prosperity-3-backtester`

This repo is less about strategy and more about environment structure.

### Main tooling findings

- the backtester explicitly includes `VOLCANIC_ROCK` and the voucher symbols in Round 3 data
- it supports running backtests across rounds and days
- it supports opening the result in a visualizer with `--vis`
- it documents order matching assumptions clearly

### Practical takeaway

Two things matter here:

1. Public teams valued a repeatable local replay environment
2. The visualizer was tightly coupled to the backtest log format

That suggests our own dashboard should be designed with log compatibility and replay in mind, even if we start with pure CSV exploration.

## What the tooling repos say about dashboards

## `jmerle/imc-prosperity-2` and `jmerle/imc-prosperity-3`

These repos are strong evidence that public Prosperity workflows often revolve around a toolchain:

- backtester
- visualizer
- submitter
- leaderboard helper

The important lesson is not one exact UI choice. The important lesson is that high-performing teams invested heavily in:

- reproducible replay
- log visualization
- day-by-day inspection
- product-by-product drilldown

## `TimoDiehm/imc-prosperity-3`

This repo contains the best public dashboard philosophy writeup I found.

### What their dashboard emphasized

- market microstructure, not just line charts
- explicit order-book level plotting
- rich hover information
- synchronized log view
- PnL panel
- position panel
- product and log selection controls
- overlaying logged indicators
- a normalization control
- trade filtering by trader group / trade type / quantity
- performance and downsampling controls

### Why this matters

This is the clearest public argument for a dashboard that is:

- functional first
- order-book aware
- inspection-heavy
- flexible in filtering

It also supports your instinct to keep the main page clean and move deeper relationship work into a separate view.

## `CarterT27/imc-prosperity-3`

This repo makes a smaller but important point:

- the public visualizer was valuable for stepping through one timestamp at a time
- it helped with PnL inspection
- it helped with position sizing inspection
- it helped with order-book debugging

This strongly supports building a dashboard that is not just "pretty charts," but also a practical debugging surface.

## `chrispyroberts/imc-prosperity-4`

This repo is the most modern public dashboard architecture in the set I reviewed.

### Main tooling findings

- it combines historical replay with Monte Carlo simulation
- it outputs structured dashboard bundles
- it exposes metrics such as profitability and stability
- it includes path-based boards and cross-product diagnostics
- it uses a separate frontend for visualization

### Why this matters

Even though it is not Round 3 voucher research directly, it shows a more mature way to think about tooling:

- one layer for raw replay
- another for robustness analysis
- dashboard outputs as data products, not just plots

We do not need to clone this architecture immediately, but it is a good north star.

## What these repos collectively imply for our Round 3 dashboard

## Minimum market-exploration layer

We should absolutely support:

- day selection
- product selection
- mid-price time series
- best bid / ask overlays
- trades overlay
- spread panel
- depth panel
- trade activity panel

That is the baseline.

## Minimum option / voucher layer

Because prior-year public repos consistently treated vouchers as option chains, we should also support:

- underlying + selected voucher comparison
- all-strikes voucher comparison at one timestamp
- strike ladder view
- rolling or point-in-time implied vol view
- moneyness-aware comparisons
- TTE display in the UI
- cross-strike relationship inspection

Even if we do not implement full option modeling on day one, the dashboard should at least make these comparisons easy.

## Dedicated relationship lab

This deserves its own view, not extra clutter on the main page.

That view should contain:

- normalized price comparison
- return comparison
- rolling correlation
- scatter comparison
- lag / lead exploration
- voucher ladder / chain panels
- correlation matrix for selected day

That lines up well with the public GitHub workflows and with your request to keep the main page clean.

## Strong candidate features to borrow

From all the public material, the most valuable borrowable ideas are:

### Borrow immediately

- separate market view and deeper research view
- product/day selectors
- explicit order-book-aware charts
- trade filtering
- timestamp-aware hover inspection
- summary cards
- synchronized PnL / position / price context when backtest logs are available

### Borrow soon after

- normalization relative to a chosen anchor
- strike-ladder or option-chain panel
- implied-vol charts
- best/worst session style summaries for backtests
- path comparison boards

### Borrow only if we need them later

- Monte Carlo route
- optimizer integration
- parameter sweep UI
- in-browser strategy editing

Those are powerful, but they are not necessary for a clean Round 3 exploration dashboard.

## What not to over-copy from public repos

There are a few cautions worth keeping in mind.

### 1. Many public writeups are post-hoc

These are useful, but they are still retrospective explanations. We should treat them as strong hints, not as ground truth.

### 2. Some public strategies depended on earlier years' exact market quirks

For example:

- specific hedge-cost assumptions
- exact spread behavior
- exact quote-wall patterns
- exact bot behavior

Those ideas may transfer only partially.

### 3. Tooling quality matters more than cloning someone else's model

Across the repos, one pattern shows up clearly:

- the best teams did not just have a formula
- they had a workflow for replaying data, inspecting fills, checking positions, and validating assumptions quickly

For us, the main lesson is to build a dashboard that helps us think clearly.

## Recommended immediate use of this research

This GitHub pass strongly supports the following direction:

1. Keep `algorithmic.md` focused on Round 3 mechanics
2. Build a dedicated clean visualizer instead of extending the current messy one
3. Give the visualizer two top-level views:
   - `Market View`
   - `Relationship Lab`
4. Treat vouchers as a true multi-strike option family in the dashboard design
5. Make sure the dashboard can answer both:
   - "What is happening in this one product right now?"
   - "How is this voucher family behaving relative to the underlying and to itself?"

## Useful public source list

- `https://github.com/ericcccsliu/imc-prosperity-2`
- `https://github.com/jmerle/imc-prosperity-2`
- `https://github.com/jmerle/imc-prosperity-2-backtester`
- `https://github.com/jmerle/imc-prosperity-3`
- `https://github.com/jmerle/imc-prosperity-3-backtester`
- `https://github.com/TimoDiehm/imc-prosperity-3`
- `https://github.com/chrispyroberts/imc-prosperity-3`
- `https://github.com/angus4718/imc-prosperity-3-public`
- `https://github.com/CarterT27/imc-prosperity-3`
- `https://github.com/nabayansaha/imc-prosperity-4-backtester`
- `https://github.com/chrispyroberts/imc-prosperity-4`

