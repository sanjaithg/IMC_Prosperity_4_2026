"""
visualize_strategy.py
=====================
Run a strategy through the backtester and produce an interactive HTML
showing:
  1. VELVETFRUIT mid price + every fill (BUY / SELL)
  2. VEV_5000 spread with markers at signal events (5 = buy, 7 = sell)
  3. VEV_4500 spread with markers at signal events (15 = buy, 17 = sell)
  4. Net position over time (per-product or total)
  5. Cumulative PnL

Usage:
    python visualize_strategy.py --submission submission23c_combined_signal_mm.py --day 1
    python visualize_strategy.py --submission submission22_dual_signal_mm.py --day 2
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import backtester  # noqa: E402


def load_strategy(path: str):
    spec = importlib.util.spec_from_file_location("strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Trader


def run_backtest(submission_path: str, day: int):
    Trader = load_strategy(submission_path)
    trader = Trader()
    data_dir = os.path.join(ROOT, "Datasets")
    prices, trades = backtester.load_day(data_dir, day)
    pnl_df, fills_df, _, _ = backtester.run_backtest_single_day(
        trader, prices, trades, day=day, passive_fills=True, verbose=False,
    )
    return pnl_df, fills_df, prices


def build_signal_markers(book: pd.DataFrame, product: str, value: int) -> pd.DataFrame:
    sub = book[book["product"] == product].copy()
    sub = sub.dropna(subset=["bid_price_1", "ask_price_1"])
    sub["spread"] = sub["ask_price_1"] - sub["bid_price_1"]
    return sub[sub["spread"] == value]


def build_figure(pnl_df, fills_df, book, day):
    vf = (book[book["product"] == "VELVETFRUIT_EXTRACT"]
          [["timestamp", "mid_price", "bid_price_1", "ask_price_1"]]
          .sort_values("timestamp").reset_index(drop=True))
    v5k = book[book["product"] == "VEV_5000"].copy()
    v5k = v5k.dropna(subset=["bid_price_1", "ask_price_1"])
    v5k["spread"] = v5k["ask_price_1"] - v5k["bid_price_1"]
    v45 = book[book["product"] == "VEV_4500"].copy()
    v45 = v45.dropna(subset=["bid_price_1", "ask_price_1"])
    v45["spread"] = v45["ask_price_1"] - v45["bid_price_1"]

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.30, 0.16, 0.16, 0.18, 0.20],
        subplot_titles=(
            f"VELVETFRUIT mid price + fills (Day {day})",
            "VEV_5000 spread (mode=6, signals at 5/7)",
            "VEV_4500 spread (mode=16, signals at 15/17)",
            "Net position by product",
            "Cumulative PnL ($)",
        ),
    )

    # ─── (1) VELVETFRUIT price + fills ────────────────────────────
    fig.add_trace(
        go.Scatter(x=vf["timestamp"], y=vf["mid_price"], mode="lines",
                   line=dict(color="#cbd5e1", width=0.8), name="VELVETFRUIT mid",
                   showlegend=True),
        row=1, col=1,
    )

    if not fills_df.empty:
        f_velv = fills_df[fills_df["product"] == "VELVETFRUIT_EXTRACT"].copy()
        buys = f_velv[f_velv["side"].str.startswith("BUY", na=False)]
        sells = f_velv[f_velv["side"].str.startswith("SELL", na=False)]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(x=buys["timestamp"], y=buys["price"], mode="markers",
                           marker=dict(color="#22c55e", size=6, symbol="triangle-up",
                                       line=dict(color="white", width=0.5)),
                           name=f"BUY fills ({len(buys)})",
                           hovertemplate="ts=%{x}<br>price=%{y}<br>qty=%{customdata}",
                           customdata=buys["quantity"]),
                row=1, col=1,
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(x=sells["timestamp"], y=sells["price"], mode="markers",
                           marker=dict(color="#ef4444", size=6, symbol="triangle-down",
                                       line=dict(color="white", width=0.5)),
                           name=f"SELL fills ({len(sells)})",
                           hovertemplate="ts=%{x}<br>price=%{y}<br>qty=%{customdata}",
                           customdata=sells["quantity"]),
                row=1, col=1,
            )

    # ─── (2) VEV_5000 spread + signal markers ─────────────────────
    fig.add_trace(
        go.Scatter(x=v5k["timestamp"], y=v5k["spread"], mode="lines",
                   line=dict(color="#94a3b8", width=0.5), name="VEV_5000 spread",
                   showlegend=True),
        row=2, col=1,
    )
    sig_5buy = v5k[v5k["spread"] == 5]
    sig_5sell = v5k[v5k["spread"] == 7]
    fig.add_trace(
        go.Scatter(x=sig_5buy["timestamp"], y=sig_5buy["spread"], mode="markers",
                   marker=dict(color="#22c55e", size=6, symbol="circle"),
                   name=f"5000 spread=5 BUY ({len(sig_5buy)})"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=sig_5sell["timestamp"], y=sig_5sell["spread"], mode="markers",
                   marker=dict(color="#ef4444", size=6, symbol="circle"),
                   name=f"5000 spread=7 SELL ({len(sig_5sell)})"),
        row=2, col=1,
    )

    # ─── (3) VEV_4500 spread + signal markers ─────────────────────
    fig.add_trace(
        go.Scatter(x=v45["timestamp"], y=v45["spread"], mode="lines",
                   line=dict(color="#94a3b8", width=0.5), name="VEV_4500 spread",
                   showlegend=True),
        row=3, col=1,
    )
    sig_45buy = v45[v45["spread"] == 15]
    sig_45sell = v45[v45["spread"] == 17]
    fig.add_trace(
        go.Scatter(x=sig_45buy["timestamp"], y=sig_45buy["spread"], mode="markers",
                   marker=dict(color="#22c55e", size=6, symbol="circle"),
                   name=f"4500 spread=15 BUY ({len(sig_45buy)})"),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=sig_45sell["timestamp"], y=sig_45sell["spread"], mode="markers",
                   marker=dict(color="#ef4444", size=6, symbol="circle"),
                   name=f"4500 spread=17 SELL ({len(sig_45sell)})"),
        row=3, col=1,
    )

    # ─── (4) Position by product ──────────────────────────────────
    pos_cols = [c for c in pnl_df.columns if c.startswith("pos_")]
    palette = ["#fbbf24", "#06b6d4", "#a78bfa", "#f472b6",
               "#84cc16", "#22c55e", "#ef4444", "#3b82f6"]
    for i, c in enumerate(pos_cols):
        prod = c.replace("pos_", "")
        if (pnl_df[c].abs().max() == 0):
            continue
        fig.add_trace(
            go.Scatter(x=pnl_df["timestamp"], y=pnl_df[c], mode="lines",
                       line=dict(color=palette[i % len(palette)], width=1),
                       name=f"pos {prod}"),
            row=4, col=1,
        )

    # ─── (5) Cumulative PnL ───────────────────────────────────────
    if "pnl" in pnl_df.columns:
        fig.add_trace(
            go.Scatter(x=pnl_df["timestamp"], y=pnl_df["pnl"], mode="lines",
                       line=dict(color="#22c55e", width=1.5), name="cumulative PnL",
                       fill="tozeroy", fillcolor="rgba(34,197,94,0.1)"),
            row=5, col=1,
        )

    fig.update_layout(
        height=1300, width=1500, hovermode="x unified",
        template="plotly_dark", showlegend=True,
        title=f"Strategy backtest visualization — Day {day}",
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)),
    )
    fig.update_xaxes(title_text="timestamp", row=5, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1, dtick=1)
    fig.update_yaxes(title_text="Spread", row=3, col=1, dtick=1)
    fig.update_yaxes(title_text="Position", row=4, col=1)
    fig.update_yaxes(title_text="$ PnL", row=5, col=1)

    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission", required=True, help="Path to submission.py")
    p.add_argument("--day", type=int, default=1, help="Day index (0/1/2)")
    p.add_argument("--out", default=None, help="Output HTML path")
    args = p.parse_args()

    name = os.path.basename(args.submission).replace(".py", "")
    out_path = args.out or os.path.join(ROOT, f"viz_{name}_day{args.day}.html")

    print(f"Running backtest: {args.submission} (day {args.day})…")
    pnl_df, fills_df, book = run_backtest(args.submission, args.day)
    print(f"  fills: {len(fills_df):,}")
    if not fills_df.empty:
        print(f"  fills by product:")
        print(fills_df["product"].value_counts().to_string())
    if "pnl" in pnl_df.columns:
        print(f"  final PnL: ${pnl_df['pnl'].iloc[-1]:,.0f}")

    fig = build_figure(pnl_df, fills_df, book, args.day)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\nVisualization → {out_path}")


if __name__ == "__main__":
    main()
