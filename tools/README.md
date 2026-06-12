# Shared visualization tools

Round-agnostic dashboards used across all rounds. They take a `--data` directory of
`prices_round_*.csv` / `trades_round_*.csv` files or a backtester output. These are the most
complete versions of the tools (consolidated from Round 5, where they were most developed).

| Tool | Purpose |
| :--- | :--- |
| `data_visualizer.py` | Interactive Plotly/Dash dashboard for raw market data — mid-price, order-book depth, spreads, MACD / z-score / Bollinger overlays. |
| `log_visualizer.py` | Parses an algorithm's execution log and visualizes positions, fills, and per-product PnL over time. |
| `backtest_log_visualizer.py` | Builds HTML charts from a backtester run's PnL / fills output per product. |

## Usage

```bash
# from inside a round folder, point at that round's data
python ../tools/data_visualizer.py --data data --no-browser
python ../tools/log_visualizer.py  --log path/to/run.log --no-browser
```

Most tools accept `--no-browser` to render to a static file instead of launching a server, and
`--port` to pick the Dash port. Run any tool with `--help` for its full flag set.

> Round-specific visualizers (e.g. Round 4's counterparty/"Mark" profiler and signal dashboard)
> live under that round's own folder, since they understand round-specific products and signals.
