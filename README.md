# IMC Prosperity 2025

Algorithmic & quantitative trading strategies, backtesters, and visualizers built for the
**IMC Prosperity** trading competition. This repository is a cleaned-up, curated archive: for
every round we keep only the **best-performing strategies**, one canonical backtester +
data model, the market data needed to reproduce results, and shared visualization tooling.

> Each round in Prosperity is a self-contained market with its own products, position limits,
> and mechanics. The competition mixes an **algorithmic** challenge (upload a Python `Trader`
> that trades on your behalf) with a separate **manual** trading challenge each round.

---

## Repository layout

```
IMC-Prosperity-2025/
├── README.md                  ← you are here
├── requirements.txt           ← unified Python dependencies
├── .gitignore
│
├── tools/                     ← shared, round-agnostic visualizers
│   ├── data_visualizer.py         interactive market-data dashboard (Plotly/Dash)
│   ├── log_visualizer.py          algorithm execution-log viewer
│   └── backtest_log_visualizer.py charts from backtester PnL/fills output
│
├── round1-tutorial/           ← Tutorial / Round 1  (EMERALD, TOMATO, OSMIUM, PEPPER)
├── round2/                    ← Round 2             (baskets / spreads, PEPPER + OSMIUM)
├── round3/                    ← Round 3             (HYDROGEL, VELVETFRUIT, VEV vouchers)
├── round4/                    ← Round 4             (R3 instruments + counterparty "Marks")
└── round5/                    ← Round 5             (50 new products, "cherry-pick winners")
```

Every `roundN/` folder is **self-contained** and follows the same shape:

```
roundN/
├── README.md          what the round was, the kept strategies, and why
├── datamodel.py       the data model this round's code was written against
├── backtester.py      the backtester for this round
├── strategies/        the best-performing Trader implementations (top ~5)
└── data/              prices_round_*.csv / trades_round_*.csv to run backtests
```

Some rounds also carry round-specific extras (`round4/visualizers/`, `round4/analysis/`
notebooks, `round5/research/`, `round5/docs/`).

> **Why a data model & backtester per round?** The platform's `datamodel.py` and the matching
> engine evolved between rounds (Round 5 moved to a leaner dataclass model; Round 4's backtester
> synthesizes counterparty fills for the "Mark" signals). Keeping each round pinned to the
> version its strategies were built against guarantees the backtests reproduce.

---

## Quick start

```bash
# 1. Set up the environment
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Run a backtest for a given round (run from inside that round's folder)
cd round5
../venv/bin/python backtester.py --submission strategies/submission_10_full_directional_pnl688k.py --data-dir data

# 3. Explore the market data interactively
../venv/bin/python ../tools/data_visualizer.py --data data --no-browser
```

> CLI flags vary slightly per round's backtester — run `python backtester.py --help`, or see
> each round's `README.md` for the exact commands.

---

## Results at a glance

Best backtested 3-day PnL per round (see each round README for the full ranking and methodology):

| Round | Headline products | Best kept strategy | Backtest PnL |
| :---- | :---------------- | :----------------- | -----------: |
| Tutorial / R1 | EMERALD, TOMATO, OSMIUM, PEPPER | `pepper_risk_taking.py` (inventory core) | dev-stage |
| Round 2 | PEPPER, OSMIUM (spreads) | `final_362616_pnl98k.py` | **+98,131** |
| Round 3 | HYDROGEL, VELVETFRUIT, VEV vouchers | `final_submission_pnl28k.py` | **+28,156** |
| Round 4 | R3 instruments + "Mark" counterparties | `final_submission_round4_pnl64k.py` | **+64,576** |
| Round 5 | 50 new products (10 groups) | `submission_10_full_directional_pnl688k.py` | **+687,942** |

PnL figures are from local backtests on the historical days noted in each round's README and are
not directly comparable across rounds (different products, position limits, and number of days).

---

## Notes

- **Manual trading challenges** are largely out of scope for this code archive (they were one-shot
  submissions); where relevant notes survive, they live in the round README or docs.
- Market-data CSVs are committed so backtests run out of the box (~110 MB lives in `round5/data`).
  To slim the repo, see the commented block in `.gitignore`.
- The upstream community backtester [`prosperity4bt`](https://github.com/jmerle/imc-prosperity-3-backtester)
  (`pip install prosperity4bt`) was also used during development; the per-round `backtester.py`
  here is the self-contained engine the strategies were tuned against.
