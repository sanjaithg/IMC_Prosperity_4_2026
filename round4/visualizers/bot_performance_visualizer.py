"""
IMC Prosperity 4 - Round 4 Bot Performance & Insider Scanner
=============================================================
Standalone Dash app on port 8090 dedicated to per-counterparty (Mark)
PnL, trade-log visualization, and insider-style scoring.

Why this exists
---------------
In Prosperity 3 a counterparty named "Olivia" turned out to be an insider
who consistently bought just before up-moves and sold just before down-moves
across multiple products. The same kind of trader could be hiding in the
Round 4 Marks. This dashboard quantifies each Mark's performance using:

  - Realized cash + MTM PnL (per Mark, per product, per day, plus totals)
  - Good-trade rate at horizons k=1, 5, 20, 50 ticks
        good = sign(Mark side) * (mid_{t+k} - mid_t) > 0
        baseline ~50%; significantly above suggests informational edge.
  - Informational PnL = sum over fills of:  side * qty * (mid_{t+k} - mid_t)
  - Visual trade log: buy/sell markers overlaid on mid-price per product.
  - Per-product and per-counterparty PnL attribution.

Run side-by-side with the other dashboards:
    python bot_performance_visualizer.py                  # port 8090
    python bot_performance_visualizer.py --port 8091
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from threading import Timer
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_visualizer import (  # noqa: E402  reuse — do NOT mutate base file
    load_data, find_data_folder,
    KNOWN_MARKS, MARK_COLOR,
)

# ----- theme -----
C = {
    "bg": "#0a0e17", "card": "#111827", "border": "#1f2937",
    "text": "#f1f5f9", "dim": "#64748b",
    "buy": "#34d399", "sell": "#f87171",
    "mid": "#38bdf8", "alt": "#fbbf24",
    "profit": "#22c55e", "loss": "#ef4444",
}
def _theme(fig: go.Figure, **layout) -> go.Figure:
    """Apply the dark theme cleanly without mixing kwargs/dicts (which Plotly
    can confuse and silently revert to its default light template)."""
    fig.update_layout(
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        font=dict(color=C["text"], size=11),
        margin=dict(l=60, r=30, t=44, b=46),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=C["card"], font_size=11),
        **layout,
    )
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"],
                     linecolor=C["border"], tickfont=dict(color=C["text"]))
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"],
                     linecolor=C["border"], tickfont=dict(color=C["text"]))
    return fig


def _empty_fig(msg: str = "") -> go.Figure:
    fig = go.Figure()
    if msg:
        fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=C["dim"], size=14))
    return _theme(fig)


TOOLBAR = dict(displaylogo=False, toImageButtonOptions=dict(format="png", scale=2))

HORIZONS = (1, 5, 20, 50)


# =====================================================================
# TICK-GRID PnL & POSITION CURVES (for plotting)
# =====================================================================
def compute_curves(prices: pd.DataFrame,
                   trades: pd.DataFrame
                   ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Build per-Mark **smooth** time series of:
      - PnL (cash + position * current mid, summed across products)
      - Position (one column per product traded)

    Sampled on a per-day tick grid (downsampled to ~600 points/day for speed).
    Positions reset at day boundaries (mirrors the live Prosperity simulator,
    where every day is a fresh 10,000-tick simulation starting from flat).

    Returns:
      pnl_curves[mark]  -> DataFrame[day, time_s, time_global_s, pnl, pnl_cum]
      pos_curves[mark]  -> DataFrame[day, time_s, time_global_s, <product>...]
    """
    pnl_out: Dict[str, pd.DataFrame] = {}
    pos_out: Dict[str, pd.DataFrame] = {}
    if trades.empty or prices.empty:
        return pnl_out, pos_out

    # Build per-day mid pivots (timestamp x product), forward-filled
    mid_pivots = {}
    for d, dfp in prices.groupby("day"):
        pv = (dfp.dropna(subset=["mid_price"])
                 .pivot_table(index="timestamp", columns="product",
                              values="mid_price", aggfunc="last"))
        pv = pv.sort_index().ffill().bfill()
        mid_pivots[int(d)] = pv

    days_sorted = sorted(mid_pivots.keys())

    for mark in KNOWN_MARKS:
        sub = trades[(trades["buyer"] == mark) | (trades["seller"] == mark)].copy()
        if sub.empty:
            continue
        sub["side_sign"] = np.where(sub["buyer"] == mark, 1, -1)
        sub["signed_qty"] = sub["side_sign"] * sub["quantity"]
        sub["cashflow"] = -sub["signed_qty"] * sub["price"]

        per_day_pnl_frames = []
        per_day_pos_frames = []
        prior_day_pnl = 0.0

        for d in days_sorted:
            day_trades = (sub[sub["day"] == d]
                          .sort_values("timestamp")
                          .reset_index(drop=True))
            mid_pv = mid_pivots[d]
            # Tick grid: every 20th timestamp (≈100 samples per 100k ticks; we
            # have 10k ticks/day so ~500 samples/day — plenty).
            grid_ts = mid_pv.index[::20]
            if len(grid_ts) == 0:
                continue
            # Cumulative cashflow at each grid point: cashflow events <= ts
            if not day_trades.empty:
                day_trades_sorted = day_trades.sort_values("timestamp")
                day_ts = day_trades_sorted["timestamp"].values
                day_cf = day_trades_sorted["cashflow"].values
                idx = np.searchsorted(day_ts, grid_ts, side="right")
                cum_cf = np.zeros(len(grid_ts))
                cum_cf_at_event = np.cumsum(day_cf)
                # cum_cf at grid ts = cum_cf_at_event[idx-1] when idx > 0
                cum_cf = np.where(idx > 0, cum_cf_at_event[np.clip(idx - 1, 0, None)], 0.0)
            else:
                cum_cf = np.zeros(len(grid_ts))

            # Per-product cumulative position at each grid point
            products_traded = day_trades["symbol"].unique() if not day_trades.empty else []
            pos_per_prod = {}
            for prod in products_traded:
                sub_p = day_trades[day_trades["symbol"] == prod]
                p_ts = sub_p["timestamp"].values
                p_dq = sub_p["signed_qty"].values
                idx = np.searchsorted(p_ts, grid_ts, side="right")
                cum = np.cumsum(p_dq)
                pos = np.where(idx > 0, cum[np.clip(idx - 1, 0, None)], 0).astype(float)
                pos_per_prod[prod] = pos

            # Mid at each grid_ts per product
            mids_at_grid = mid_pv.loc[grid_ts]

            # MTM = cum_cf + sum_p pos_p * mid_p_at_t
            mtm = cum_cf.astype(float).copy()
            for prod, pos in pos_per_prod.items():
                if prod in mids_at_grid.columns:
                    mids = mids_at_grid[prod].ffill().bfill().values
                    mtm = mtm + pos * mids

            # End-of-day MTM is the last grid value -> carry into next day
            day_pnl_frame = pd.DataFrame({
                "day": int(d),
                "time_s": grid_ts / 100.0,           # seconds within day
                "time_global_s": (int(d) - days_sorted[0]) * 10000 + grid_ts / 100.0,
                "pnl": mtm,
                "pnl_cum": mtm + prior_day_pnl,
            })
            per_day_pnl_frames.append(day_pnl_frame)

            # Position frame for the day
            day_pos = {
                "day": int(d),
                "time_s": grid_ts / 100.0,
                "time_global_s": (int(d) - days_sorted[0]) * 10000 + grid_ts / 100.0,
            }
            for prod, pos in pos_per_prod.items():
                day_pos[prod] = pos
            per_day_pos_frames.append(pd.DataFrame(day_pos))

            prior_day_pnl += float(mtm[-1])

        if per_day_pnl_frames:
            pnl_out[mark] = pd.concat(per_day_pnl_frames, ignore_index=True)
        if per_day_pos_frames:
            pos_out[mark] = pd.concat(per_day_pos_frames, ignore_index=True).fillna(0)

    return pnl_out, pos_out


# =====================================================================
# CORE: per-Mark PnL accounting
# =====================================================================
def _enrich_trades(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Add mid-at-trade and forward-mid columns to every trade row."""
    if trades.empty:
        return trades.copy()
    # Mid lookup keyed on (day, timestamp, product)
    mid_lookup = (prices[["day", "timestamp", "product", "mid_price"]]
                  .dropna(subset=["mid_price"])
                  .rename(columns={"product": "symbol"}))
    out = trades.merge(mid_lookup, on=["day", "timestamp", "symbol"], how="left")
    # Forward mids per horizon: for each row, mid at (day, timestamp + 100*k, symbol)
    for k in HORIZONS:
        fwd = mid_lookup.rename(columns={"mid_price": f"mid_fwd_{k}",
                                          "timestamp": "_ts_match"})
        fwd["_ts_match"] = fwd["_ts_match"] - 100 * k
        out = out.merge(fwd[["day", "_ts_match", "symbol", f"mid_fwd_{k}"]],
                        left_on=["day", "timestamp", "symbol"],
                        right_on=["day", "_ts_match", "symbol"],
                        how="left").drop(columns=["_ts_match"])
    return out


def compute_pnl_per_mark(prices: pd.DataFrame,
                         enriched_trades: pd.DataFrame
                         ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Build the per-Mark performance summary table AND a per-Mark dict of
    detailed trade-level dataframes (with cum_cash, cum_pos, cum_pnl,
    fwd-return scores) for plotting.

    PnL convention:
      cash flow on each fill: buy => -price*qty; sell => +price*qty
      MTM at end of day per product = cum_cash + final_position * final_mid_for_that_(mark, product, day)

    Total Mark PnL = sum across (product, day) of MTM.
    """
    if enriched_trades.empty:
        return pd.DataFrame(), {}

    # End-of-day mid per (day, product) for MTM
    eod_mid = (prices.dropna(subset=["mid_price"])
               .sort_values("timestamp")
               .groupby(["day", "product"], as_index=False)
               .tail(1)[["day", "product", "mid_price"]]
               .rename(columns={"mid_price": "eod_mid", "product": "symbol"}))

    detailed: Dict[str, pd.DataFrame] = {}
    summary_rows: List[dict] = []

    # Score helper
    def fwd_score(side_sign: pd.Series, mid: pd.Series, fwd: pd.Series) -> pd.Series:
        return side_sign * (fwd - mid)

    for mark in KNOWN_MARKS:
        sub = enriched_trades[(enriched_trades["buyer"] == mark) |
                              (enriched_trades["seller"] == mark)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["day", "timestamp"]).reset_index(drop=True)
        sub["side_sign"] = np.where(sub["buyer"] == mark, 1, -1)  # +1 buyer, -1 seller
        sub["signed_qty"] = sub["side_sign"] * sub["quantity"]
        sub["cashflow"] = -sub["signed_qty"] * sub["price"]   # buying => -cash
        sub["t_global"] = sub["day"].astype(int) * 1_000_000 + sub["timestamp"].astype(int)

        # Per (day, product) cumulative cash & position so we can MTM at EOD
        sub.sort_values(["symbol", "day", "timestamp"], inplace=True)
        sub["cum_cash_dp"] = sub.groupby(["symbol", "day"])["cashflow"].cumsum()
        sub["cum_pos_dp"] = sub.groupby(["symbol", "day"])["signed_qty"].cumsum()
        sub.sort_values("t_global", inplace=True)
        sub.reset_index(drop=True, inplace=True)

        # Whole-Mark cumulative PnL curve uses mid-at-trade for MTM
        sub["cum_cash_total"] = sub["cashflow"].cumsum()
        # position-by-product running map (need it to MTM each tick at current mid)
        # Easier proxy: PnL at each fill = cum_cash_total + sum_over_products(pos_dp * mid_at_trade)
        # We'll approximate with mid_price (the symbol of the current row) for that row's symbol
        # plus carrying open positions in other products at their last seen mid.
        # For simplicity in the curve, use cash + per-symbol pos * mid AT THIS ROW's product:
        sub["pnl_curve"] = sub["cum_cash_dp"] + sub["cum_pos_dp"] * sub["mid_price"]
        # That is per-(symbol,day) PnL. For total Mark PnL curve over time, sum across symbols:
        # Build a total-pnl curve via running totals across symbols.
        # Approach: for each row, the row's contribution to total = pnl_curve_thissymboldayrow
        # Total Mark PnL over time = sum_i for each (symbol,day) of latest pnl_curve up to that t.
        # We compute it cumulatively below.
        last_per_dp: Dict[Tuple, float] = {}
        running_total = []
        for _, r in sub.iterrows():
            key = (r["symbol"], r["day"])
            last_per_dp[key] = float(r["pnl_curve"])
            running_total.append(sum(last_per_dp.values()))
        sub["pnl_total_running"] = running_total

        # Final MTM per (symbol,day) using EOD mid (more accurate than last-trade mid)
        sub_dp = sub.groupby(["symbol", "day"], as_index=False).agg(
            cum_cash=("cashflow", "sum"),
            cum_pos=("signed_qty", "sum"),
        )
        sub_dp = sub_dp.merge(eod_mid, on=["symbol", "day"], how="left")
        sub_dp["mtm_pnl"] = sub_dp["cum_cash"] + sub_dp["cum_pos"] * sub_dp["eod_mid"]

        # Fwd-return scores per fill (for good-trade rate + info-PnL)
        for k in HORIZONS:
            fcol = f"mid_fwd_{k}"
            if fcol in sub.columns:
                sub[f"score_k{k}"] = sub["side_sign"] * (sub[fcol] - sub["mid_price"])
                sub[f"good_k{k}"] = (sub[f"score_k{k}"] > 0).astype(int)
            else:
                sub[f"score_k{k}"] = np.nan
                sub[f"good_k{k}"] = np.nan

        detailed[mark] = sub

        # ---------- summary row ----------
        n_trades = len(sub)
        total_qty = int(sub["quantity"].sum())
        side_bias = (sub["signed_qty"].sum() / total_qty) if total_qty else 0.0
        pnl_total = float(sub_dp["mtm_pnl"].sum())
        # info-PnL across horizons (qty-weighted forward score)
        info_pnl = {}
        good_rate = {}
        for k in HORIZONS:
            sc = sub[f"score_k{k}"].dropna()
            qty = sub.loc[sc.index, "quantity"]
            info_pnl[k] = float((sc * qty).sum())
            n = len(sc)
            good_rate[k] = float(sub[f"good_k{k}"].mean()) if n else np.nan

        # Edge-per-fill (avg) — favorable price relative to mid
        edge_per_fill = float((sub["side_sign"] * (sub["mid_price"] - sub["price"])).mean())

        # cross rate: lifted ask (buyer) or hit bid (seller). We don't have bid/ask
        # in the merged frame, so reuse mid-based approximation.
        sub["aggressive"] = ((sub["side_sign"] == 1) & (sub["price"] > sub["mid_price"])) | \
                            ((sub["side_sign"] == -1) & (sub["price"] < sub["mid_price"]))
        cross_rate = float(sub["aggressive"].mean())

        summary_rows.append(dict(
            mark=mark,
            n_trades=n_trades,
            qty=total_qty,
            side_bias=round(side_bias, 3),
            edge_per_fill=round(edge_per_fill, 3),
            cross_rate=round(cross_rate, 3),
            pnl_total=round(pnl_total, 1),
            info_pnl_k1=round(info_pnl[1], 1),
            info_pnl_k5=round(info_pnl[5], 1),
            info_pnl_k20=round(info_pnl[20], 1),
            info_pnl_k50=round(info_pnl[50], 1),
            good_rate_k1=round(good_rate[1] * 100, 1) if not np.isnan(good_rate[1]) else None,
            good_rate_k5=round(good_rate[5] * 100, 1) if not np.isnan(good_rate[5]) else None,
            good_rate_k20=round(good_rate[20] * 100, 1) if not np.isnan(good_rate[20]) else None,
            good_rate_k50=round(good_rate[50] * 100, 1) if not np.isnan(good_rate[50]) else None,
        ))

    summary = pd.DataFrame(summary_rows).sort_values("pnl_total", ascending=False).reset_index(drop=True)
    return summary, detailed


# =====================================================================
# FIGURE BUILDERS
# =====================================================================
def fig_cum_pnl_all_marks(pnl_curves: Dict[str, pd.DataFrame]) -> go.Figure:
    """All Marks' cumulative PnL on a clean per-day tick grid (smooth lines)."""
    fig = go.Figure()
    if not pnl_curves:
        return _empty_fig("No PnL curves computed")
    # Day-boundary annotations: vertical lines at end of day 1 and 2
    days = sorted({int(d) for c in pnl_curves.values() for d in c["day"].unique()})
    for mark in KNOWN_MARKS:
        if mark not in pnl_curves:
            continue
        d = pnl_curves[mark]
        fig.add_trace(go.Scatter(
            x=d["time_global_s"], y=d["pnl_cum"],
            mode="lines", name=mark,
            line=dict(color=MARK_COLOR.get(mark, "#94a3b8"), width=1.7),
            hovertemplate=(f"<b>{mark}</b><br>"
                           "day=%{customdata[0]}<br>"
                           "t in day=%{customdata[1]:.0f}s<br>"
                           "PnL=%{y:+,.1f}<extra></extra>"),
            customdata=np.stack([d["day"], d["time_s"]], axis=-1),
        ))
    # Day separators
    for i, d in enumerate(days[1:]):
        x_sep = (i + 1) * 10000  # each day is 10000s of grid time
        fig.add_vline(x=x_sep, line_color=C["dim"], line_dash="dot", line_width=1,
                      annotation_text=f"Day {d} start",
                      annotation_font=dict(color=C["dim"], size=10),
                      annotation_position="top right")
    fig.add_hline(y=0, line_color=C["border"], line_width=1)
    _theme(fig,
           title=dict(text="Cumulative PnL per Mark  (cash + MTM, positions reset each day)",
                      font=dict(size=13)),
           xaxis_title="Time across days (sec)",
           yaxis_title="Cumulative PnL (XIRECS)",
           hovermode="x unified")
    return fig


def fig_position_per_mark(pos_curves: Dict[str, pd.DataFrame],
                           mark: str) -> go.Figure:
    """For the selected Mark, plot per-product position over time (multi-line)."""
    fig = go.Figure()
    if mark not in pos_curves or pos_curves[mark].empty:
        return _empty_fig(f"{mark} has no position data")
    df = pos_curves[mark]
    product_cols = [c for c in df.columns
                    if c not in ("day", "time_s", "time_global_s")]
    if not product_cols:
        return _empty_fig(f"{mark} traded no products")
    palette = ["#60a5fa", "#34d399", "#fbbf24", "#fb923c", "#f87171",
               "#a78bfa", "#f472b6", "#22d3ee", "#facc15", "#94a3b8",
               "#84cc16", "#06b6d4"]
    for i, prod in enumerate(sorted(product_cols)):
        fig.add_trace(go.Scatter(
            x=df["time_global_s"], y=df[prod],
            mode="lines", name=prod,
            line=dict(color=palette[i % len(palette)], width=1.4),
            hovertemplate=f"<b>{prod}</b><br>day=%{{customdata[0]}}<br>"
                          "t=%{customdata[1]:.0f}s<br>position=%{y:+.0f}<extra></extra>",
            customdata=np.stack([df["day"], df["time_s"]], axis=-1),
        ))
    days = sorted(df["day"].unique())
    days_sorted = sorted({int(d) for d in days})
    for i, d in enumerate(days_sorted[1:]):
        x_sep = (i + 1) * 10000
        fig.add_vline(x=x_sep, line_color=C["dim"], line_dash="dot", line_width=1)
    fig.add_hline(y=0, line_color=C["border"], line_width=1)
    _theme(fig,
           title=dict(text=f"{mark}  -  position over time (per product)",
                      font=dict(size=13)),
           xaxis_title="Time across days (sec)",
           yaxis_title="Net position (qty)",
           hovermode="x unified")
    return fig


def fig_pnl_by_product(detailed: Dict[str, pd.DataFrame],
                        prices: pd.DataFrame,
                        mark: str) -> go.Figure:
    fig = go.Figure()
    if mark not in detailed:
        _theme(fig, title=f"No data for {mark}")
        return fig
    eod_mid = (prices.dropna(subset=["mid_price"])
               .sort_values("timestamp")
               .groupby(["day", "product"], as_index=False)
               .tail(1)[["day", "product", "mid_price"]]
               .rename(columns={"mid_price": "eod_mid", "product": "symbol"}))
    d = detailed[mark]
    grp = d.groupby(["symbol", "day"], as_index=False).agg(
        cum_cash=("cashflow", "sum"),
        cum_pos=("signed_qty", "sum"),
    ).merge(eod_mid, on=["symbol", "day"], how="left")
    grp["mtm_pnl"] = grp["cum_cash"] + grp["cum_pos"] * grp["eod_mid"]
    by_prod = grp.groupby("symbol")["mtm_pnl"].sum().sort_values()
    colors = [C["profit"] if v > 0 else C["loss"] for v in by_prod.values]
    fig.add_trace(go.Bar(
        x=by_prod.values, y=by_prod.index, orientation="h",
        marker_color=colors,
        text=[f"{v:+.0f}" for v in by_prod.values],
        textposition="outside",
        hovertemplate="%{y}: %{x:+.1f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=C["border"], line_width=1)
    _theme(fig,
           title=f"PnL attribution by product - {mark}",
           xaxis_title="PnL (XIRECS)", yaxis_title="Product")
    return fig


def fig_pnl_by_counterparty(detailed: Dict[str, pd.DataFrame],
                             mark: str) -> go.Figure:
    """For each counterparty C the selected Mark trades with, compute the Mark's MTM PnL on those fills."""
    fig = go.Figure()
    if mark not in detailed:
        _theme(fig, title=f"No data for {mark}")
        return fig
    d = detailed[mark].copy()
    # Counterparty = whoever's NOT this Mark on the row
    d["cp"] = np.where(d["buyer"] == mark, d["seller"], d["buyer"])
    # Score per fill: signed_qty * (eod_mid - price). Approximate eod_mid with mid_price
    # (we don't carry eod into per-row, but mid-at-trade is a fair proxy for "fair price now").
    d["fill_pnl"] = d["signed_qty"] * (d["mid_price"] - d["price"])
    by_cp = d.groupby("cp")["fill_pnl"].sum().sort_values()
    colors = [C["profit"] if v > 0 else C["loss"] for v in by_cp.values]
    fig.add_trace(go.Bar(
        x=by_cp.values, y=by_cp.index, orientation="h",
        marker_color=colors,
        text=[f"{v:+.0f}" for v in by_cp.values],
        textposition="outside",
        hovertemplate="vs %{y}: %{x:+.1f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=C["border"], line_width=1)
    _theme(fig,
                      title=f"Realized edge by counterparty - {mark}",
                      xaxis_title="Mark's edge (sum of side*(mid-price)*qty)",
                      yaxis_title="Counterparty")
    return fig


def fig_good_trade_rate_rolling(detailed: Dict[str, pd.DataFrame],
                                 mark: str,
                                 window: int = 30) -> go.Figure:
    """Rolling fraction of fills where Mark traded in the right direction at horizon k."""
    fig = go.Figure()
    if mark not in detailed:
        _theme(fig, title=f"No data for {mark}")
        return fig
    d = detailed[mark].sort_values("t_global").reset_index(drop=True)
    palette = ["#22d3ee", "#fbbf24", "#f97316", "#ef4444"]
    for ci, k in enumerate(HORIZONS):
        col = d[f"good_k{k}"]
        if col.isna().all():
            continue
        roll = col.rolling(window, min_periods=max(5, window // 3)).mean() * 100.0
        fig.add_trace(go.Scatter(
            x=d["t_global"], y=roll, mode="lines",
            name=f"k={k}",
            line=dict(color=palette[ci], width=1.6),
            hovertemplate=f"k={k}<br>good rate=%{{y:.1f}}%<extra></extra>",
        ))
    fig.add_hline(y=50, line_dash="dash", line_color=C["dim"],
                  annotation_text="random baseline (50%)",
                  annotation_font=dict(color=C["dim"], size=10))
    fig.add_hline(y=60, line_dash="dot", line_color=C["alt"], opacity=0.4)
    _theme(fig,
                      title=f"Good-trade rate (rolling {window}) - {mark}",
                      xaxis_title="t_global", yaxis_title="% good trades",
                      yaxis=dict(range=[0, 100], gridcolor=C["border"]))
    return fig


def fig_score_distribution(detailed: Dict[str, pd.DataFrame],
                            mark: str, k: int = 20) -> go.Figure:
    """Histogram of forward-return score per fill at horizon k for the Mark."""
    fig = go.Figure()
    if mark not in detailed:
        _theme(fig, title=f"No data for {mark}")
        return fig
    d = detailed[mark]
    s = d[f"score_k{k}"].dropna()
    if s.empty:
        _theme(fig, title="No forward-return data")
        return fig
    fig.add_trace(go.Histogram(
        x=s, nbinsx=60,
        marker_color=MARK_COLOR.get(mark, "#94a3b8"),
        opacity=0.85,
        hovertemplate="score=%{x:+.2f}<br>count=%{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=C["border"], line_width=1)
    mean_s = float(s.mean())
    fig.add_vline(x=mean_s, line_color=C["alt"], line_width=2, line_dash="dash",
                  annotation_text=f"mean={mean_s:+.3f}",
                  annotation_font=dict(color=C["alt"], size=11))
    _theme(fig,
                      title=f"Distribution of fwd-return scores at k={k} - {mark}",
                      xaxis_title=f"side * (mid_(t+{k}) - mid_t)",
                      yaxis_title="count")
    return fig


def trade_table_data(detailed: Dict[str, pd.DataFrame],
                     mark: str,
                     product_filter: str = None,
                     day_filter: str = None) -> List[dict]:
    """Compact per-Mark trade-log rows for the DataTable."""
    if mark not in detailed or detailed[mark].empty:
        return []
    d = detailed[mark].copy()
    if product_filter and product_filter != "__all__":
        d = d[d["symbol"] == product_filter]
    if day_filter and day_filter != "__all__":
        d = d[d["day_label"] == day_filter]
    if d.empty:
        return []
    d = d.sort_values(["day", "timestamp"]).reset_index(drop=True)
    rows = []
    for _, r in d.iterrows():
        side = "BUY" if r["side_sign"] == 1 else "SELL"
        cp = r["seller"] if r["side_sign"] == 1 else r["buyer"]
        edge = float(r["side_sign"]) * (float(r["mid_price"]) - float(r["price"]))
        fwd1 = (r.get("mid_fwd_1") - r["mid_price"]) if pd.notna(r.get("mid_fwd_1")) else None
        fwd20 = (r.get("mid_fwd_20") - r["mid_price"]) if pd.notna(r.get("mid_fwd_20")) else None
        score20 = (float(r["side_sign"]) * fwd20) if fwd20 is not None else None
        rows.append({
            "day": int(r["day"]),
            "time_s": int(r["timestamp"]) // 100,
            "product": r["symbol"],
            "side": side,
            "qty": int(r["quantity"]),
            "price": round(float(r["price"]), 2),
            "mid": round(float(r["mid_price"]), 2),
            "edge": round(edge, 2),
            "fwd_k1": round(float(fwd1), 2) if fwd1 is not None else None,
            "fwd_k20": round(float(fwd20), 2) if fwd20 is not None else None,
            "score_k20": round(float(score20), 2) if score20 is not None else None,
            "counterparty": cp,
        })
    return rows


def fig_trade_log(detailed: Dict[str, pd.DataFrame],
                  prices: pd.DataFrame,
                  mark: str, product: str, day: str) -> go.Figure:
    """Mid line for the (product, day) plus this Mark's buys (green up) and sells (red down)."""
    fig = go.Figure()
    p = prices[(prices["product"] == product) & (prices["day_label"] == day)]
    if p.empty:
        _theme(fig, title=f"No price data for {product} on {day}")
        return fig
    fig.add_trace(go.Scatter(
        x=p["timestamp"] / 100.0, y=p["mid_price"], mode="lines",
        line=dict(color=C["mid"], width=1.4),
        name="Mid",
    ))
    if mark in detailed:
        d = detailed[mark]
        d_pd = d[(d["symbol"] == product) & (d["day_label"] == day)]
        if not d_pd.empty:
            buys = d_pd[d_pd["side_sign"] == 1]
            sells = d_pd[d_pd["side_sign"] == -1]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["timestamp"] / 100.0, y=buys["price"], mode="markers",
                    marker=dict(symbol="triangle-up",
                                size=np.clip(buys["quantity"].astype(float) * 2, 8, 22),
                                color=C["buy"], line=dict(color="white", width=1)),
                    name=f"{mark} BUY ({len(buys)})",
                    hovertext=[f"BUY  qty={int(q)}  @ {p_:.1f}<br>vs {cp}"
                               for q, p_, cp in zip(buys["quantity"], buys["price"],
                                                    buys["seller"])],
                    hoverinfo="text",
                ))
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["timestamp"] / 100.0, y=sells["price"], mode="markers",
                    marker=dict(symbol="triangle-down",
                                size=np.clip(sells["quantity"].astype(float) * 2, 8, 22),
                                color=C["sell"], line=dict(color="white", width=1)),
                    name=f"{mark} SELL ({len(sells)})",
                    hovertext=[f"SELL  qty={int(q)}  @ {p_:.1f}<br>vs {cp}"
                               for q, p_, cp in zip(sells["quantity"], sells["price"],
                                                    sells["buyer"])],
                    hoverinfo="text",
                ))
    _theme(fig,
                      title=f"Trade log: {mark} on {product} ({day})",
                      xaxis_title="Time (s)", yaxis_title="Price")
    return fig


# =====================================================================
# APP
# =====================================================================
def make_scoreboard_table(summary: pd.DataFrame) -> dash_table.DataTable:
    """Highlight cells: PnL color, good-rate cells colored if > 60%."""
    cols = [
        {"name": "Mark", "id": "mark"},
        {"name": "# trades", "id": "n_trades", "type": "numeric"},
        {"name": "qty", "id": "qty", "type": "numeric"},
        {"name": "bias", "id": "side_bias", "type": "numeric"},
        {"name": "edge/fill", "id": "edge_per_fill", "type": "numeric"},
        {"name": "cross %", "id": "cross_rate", "type": "numeric"},
        {"name": "TOTAL PnL", "id": "pnl_total", "type": "numeric"},
        {"name": "info PnL k=1", "id": "info_pnl_k1", "type": "numeric"},
        {"name": "info PnL k=5", "id": "info_pnl_k5", "type": "numeric"},
        {"name": "info PnL k=20", "id": "info_pnl_k20", "type": "numeric"},
        {"name": "info PnL k=50", "id": "info_pnl_k50", "type": "numeric"},
        {"name": "good %k=1", "id": "good_rate_k1", "type": "numeric"},
        {"name": "good %k=5", "id": "good_rate_k5", "type": "numeric"},
        {"name": "good %k=20", "id": "good_rate_k20", "type": "numeric"},
        {"name": "good %k=50", "id": "good_rate_k50", "type": "numeric"},
    ]
    style_rules = [
        {"if": {"filter_query": "{pnl_total} > 0", "column_id": "pnl_total"},
         "color": C["profit"], "fontWeight": "700"},
        {"if": {"filter_query": "{pnl_total} < 0", "column_id": "pnl_total"},
         "color": C["loss"], "fontWeight": "700"},
    ]
    for k in HORIZONS:
        col = f"good_rate_k{k}"
        style_rules.extend([
            {"if": {"filter_query": f"{{{col}}} >= 60", "column_id": col},
             "backgroundColor": "rgba(34,197,94,0.18)", "color": C["profit"], "fontWeight": "700"},
            {"if": {"filter_query": f"{{{col}}} <= 40", "column_id": col},
             "backgroundColor": "rgba(239,68,68,0.18)", "color": C["loss"]},
        ])
        ipnl = f"info_pnl_k{k}"
        style_rules.extend([
            {"if": {"filter_query": f"{{{ipnl}}} > 0", "column_id": ipnl}, "color": C["profit"]},
            {"if": {"filter_query": f"{{{ipnl}}} < 0", "column_id": ipnl}, "color": C["loss"]},
        ])
    return dash_table.DataTable(
        id="scoreboard",
        columns=cols,
        data=summary.to_dict("records"),
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": C["card"], "color": C["text"],
            "fontWeight": "700", "fontSize": "11px",
            "border": f"1px solid {C['border']}",
        },
        style_cell={
            "backgroundColor": C["bg"], "color": C["text"],
            "fontFamily": "Inter, monospace", "fontSize": "12px",
            "padding": "6px 10px", "border": f"1px solid {C['border']}",
            "textAlign": "right",
        },
        style_data_conditional=style_rules,
    )


def card(title: str, gid: str, h: str = "380px") -> html.Div:
    return html.Div(style={
        "background": C["card"], "border": f"1px solid {C['border']}",
        "borderRadius": "8px", "padding": "12px 16px",
    }, children=[
        html.Div(title, style={
            "fontSize": "13px", "fontWeight": "700",
            "letterSpacing": "1px", "color": C["text"], "marginBottom": "6px",
        }),
        dcc.Graph(id=gid, style={"height": h}, config=TOOLBAR),
    ])


def create_app(prices: pd.DataFrame, trades: pd.DataFrame) -> Dash:
    print("  enriching trades with mid-at-trade and forward mids...")
    enriched = _enrich_trades(prices, trades)
    print(f"  computing PnL summary across {len(KNOWN_MARKS)} marks...")
    summary, detailed = compute_pnl_per_mark(prices, enriched)
    print("  building tick-grid PnL & position curves...")
    pnl_curves, pos_curves = compute_curves(prices, trades)
    print("  done.")

    products = sorted(prices["product"].unique()) if not prices.empty else []
    days = sorted(prices["day_label"].unique()) if not prices.empty else []
    marks_present = [m for m in KNOWN_MARKS if m in detailed]

    app = Dash(__name__, title="Round 4 - Bot Performance Scanner",
               suppress_callback_exceptions=True)

    app.layout = html.Div(style={
        "backgroundColor": C["bg"], "color": C["text"],
        "fontFamily": "Inter, -apple-system, sans-serif", "minHeight": "100vh",
    }, children=[
        html.Div(style={
            "background": f"linear-gradient(135deg, {C['card']}, {C['bg']})",
            "borderBottom": f"1px solid {C['border']}",
            "padding": "16px 28px", "display": "flex",
            "justifyContent": "space-between", "alignItems": "center",
        }, children=[
            html.Div([
                html.Span("PROSPERITY 4  -  BOT PERFORMANCE & INSIDER SCANNER",
                          style={"fontSize": "18px", "fontWeight": "800",
                                 "letterSpacing": "3px", "color": C["mid"]}),
                html.Div("Looking for an Olivia: Marks with persistently > 50% good-trade rate",
                         style={"fontSize": "11px", "color": C["dim"],
                                "marginTop": "4px", "letterSpacing": "1px"}),
            ]),
            html.Div(style={"display": "flex", "gap": "16px", "alignItems": "center"}, children=[
                html.Div([
                    html.Label("Mark", style={"fontSize": "10px", "color": C["dim"],
                        "textTransform": "uppercase", "letterSpacing": "1px"}),
                    dcc.Dropdown(id="b-mark",
                                 options=[{"label": m, "value": m} for m in marks_present],
                                 value=marks_present[0] if marks_present else None,
                                 clearable=False, style={"width": "150px"}),
                ]),
                html.Div([
                    html.Label("Product (trade log)", style={"fontSize": "10px",
                        "color": C["dim"], "textTransform": "uppercase",
                        "letterSpacing": "1px"}),
                    dcc.Dropdown(id="b-product",
                                 options=[{"label": p, "value": p} for p in products],
                                 value="VELVETFRUIT_EXTRACT" if "VELVETFRUIT_EXTRACT" in products
                                       else (products[0] if products else None),
                                 clearable=False, style={"width": "210px"}),
                ]),
                html.Div([
                    html.Label("Day (trade log)", style={"fontSize": "10px",
                        "color": C["dim"], "textTransform": "uppercase",
                        "letterSpacing": "1px"}),
                    dcc.Dropdown(id="b-day",
                                 options=[{"label": d, "value": d} for d in days],
                                 value=days[0] if days else None,
                                 clearable=False, style={"width": "120px"}),
                ]),
                html.Div([
                    html.Label("Roll window", style={"fontSize": "10px", "color": C["dim"],
                        "textTransform": "uppercase", "letterSpacing": "1px"}),
                    dcc.Slider(id="b-window", min=10, max=120, step=10, value=30,
                               marks={10: "10", 30: "30", 60: "60", 120: "120"},
                               tooltip={"placement": "bottom"}),
                ], style={"width": "240px"}),
            ]),
        ]),

        # Column glossary
        html.Div(style={"padding": "16px 28px 0"}, children=[
            html.Div("WHAT EACH COLUMN MEANS", style={
                "fontSize": "12px", "fontWeight": "700",
                "letterSpacing": "2px", "color": C["dim"],
                "marginBottom": "8px"}),
            html.Div(style={
                "background": C["card"], "border": f"1px solid {C['border']}",
                "borderRadius": "8px", "padding": "12px 16px",
                "fontSize": "12px", "lineHeight": "1.7",
            }, children=[
                html.Div([html.B("# trades"), " - number of fills the Mark appears in (as buyer or seller)."]),
                html.Div([html.B("qty"), " - total volume traded (sum of quantity)."]),
                html.Div([html.B("bias"), " - (buy_qty - sell_qty) / total_qty. +1 = pure buyer, -1 = pure seller, 0 = balanced market-maker."]),
                html.Div([html.B("edge/fill"), " - average ", html.Code("side * (mid - price)"), ". POSITIVE = trades favorably (passive maker). NEGATIVE = pays the spread (taker)."]),
                html.Div([html.B("cross %"), " - fraction of fills where they crossed the spread (lifted ask as buyer or hit bid as seller). High % = aggressive taker; 0% = pure passive maker."]),
                html.Div([html.Span("TOTAL PnL", style={"color": C["alt"], "fontWeight": 700}),
                          " - their actual profit: ",
                          html.Code("cash + position * EOD_mid"), ", summed across all products and days. THIS IS THE BOTTOM LINE."]),
                html.Div([html.B("info PnL k"), " - informational PnL at horizon k ticks: sum over fills of ",
                          html.Code("side * qty * (mid_(t+k) - mid_t)"),
                          ". Quantifies how much they make from being right about direction. Big positive = informed. Negative = consistently wrong."]),
                html.Div([html.B("good %k"), " - % of fills where ",
                          html.Code("side * (mid_(t+k) - mid_t) > 0"),
                          ". Random baseline = 50%. ",
                          html.Span(">60% over 100+ trades is statistically impossible by chance.",
                                    style={"color": C["alt"], "fontWeight": 700})]),
            ]),
        ]),

        # Insider scoreboard
        html.Div(style={"padding": "18px 28px"}, children=[
            html.Div("INSIDER SCOREBOARD  -  sort by any column",
                     style={"fontSize": "12px", "fontWeight": "700",
                            "letterSpacing": "2px", "color": C["dim"],
                            "marginBottom": "8px"}),
            make_scoreboard_table(summary),
        ]),

        # Cumulative PnL + position across all marks
        html.Div(style={"padding": "8px 28px 4px",
                        "display": "flex", "flexDirection": "column",
                        "gap": "14px"}, children=[
            card("All-marks cumulative PnL  (cash + MTM, positions reset at day boundary)",
                 "b-cum-all", "460px"),
            card("Selected Mark's POSITION over time  (per product)",
                 "b-position", "400px"),
        ]),

        # Per-Mark cards
        html.Div(style={"padding": "8px 28px", "display": "flex",
                        "flexDirection": "column", "gap": "14px"}, children=[
            html.Div("FOCUS: SELECTED MARK", style={
                "fontSize": "12px", "fontWeight": "700",
                "letterSpacing": "2px", "color": C["dim"],
                "marginTop": "6px", "marginBottom": "-4px",
            }),
            card("Trade log  (Mark X buys / sells overlaid on mid-price)",
                 "b-trade-log", "460px"),

            # Sortable / filterable trade table
            html.Div(style={
                "background": C["card"], "border": f"1px solid {C['border']}",
                "borderRadius": "8px", "padding": "12px 16px",
            }, children=[
                html.Div([
                    html.Span("Trade table  -  every fill the selected Mark appears in",
                              style={"fontSize": "13px", "fontWeight": "700",
                                     "letterSpacing": "1px", "color": C["text"]}),
                    html.Span("   (click column headers to sort, type in row 2 to filter)",
                              style={"fontSize": "11px", "color": C["dim"],
                                     "marginLeft": "10px"}),
                ], style={"marginBottom": "8px"}),
                dash_table.DataTable(
                    id="b-trade-table",
                    columns=[
                        {"name": "day", "id": "day", "type": "numeric"},
                        {"name": "time (s)", "id": "time_s", "type": "numeric"},
                        {"name": "product", "id": "product"},
                        {"name": "side", "id": "side"},
                        {"name": "qty", "id": "qty", "type": "numeric"},
                        {"name": "price", "id": "price", "type": "numeric"},
                        {"name": "mid", "id": "mid", "type": "numeric"},
                        {"name": "edge", "id": "edge", "type": "numeric"},
                        {"name": "fwd k=1", "id": "fwd_k1", "type": "numeric"},
                        {"name": "fwd k=20", "id": "fwd_k20", "type": "numeric"},
                        {"name": "score k=20", "id": "score_k20", "type": "numeric"},
                        {"name": "counterparty", "id": "counterparty"},
                    ],
                    data=[],
                    sort_action="native",
                    filter_action="native",
                    page_size=20,
                    style_table={"overflowX": "auto", "maxHeight": "560px"},
                    fixed_rows={"headers": True},
                    style_header={
                        "backgroundColor": C["card"], "color": C["text"],
                        "fontWeight": "700", "fontSize": "11px",
                        "border": f"1px solid {C['border']}",
                    },
                    style_filter={
                        "backgroundColor": C["bg"], "color": C["text"],
                        "border": f"1px solid {C['border']}",
                    },
                    style_cell={
                        "backgroundColor": C["bg"], "color": C["text"],
                        "fontFamily": "Inter, monospace", "fontSize": "12px",
                        "padding": "4px 8px", "border": f"1px solid {C['border']}",
                        "textAlign": "right",
                    },
                    style_cell_conditional=[
                        {"if": {"column_id": c}, "textAlign": "left"}
                        for c in ("product", "side", "counterparty")
                    ],
                    style_data_conditional=[
                        {"if": {"filter_query": "{side} = BUY", "column_id": "side"},
                         "color": C["buy"], "fontWeight": "700"},
                        {"if": {"filter_query": "{side} = SELL", "column_id": "side"},
                         "color": C["sell"], "fontWeight": "700"},
                        {"if": {"filter_query": "{edge} > 0", "column_id": "edge"},
                         "color": C["profit"]},
                        {"if": {"filter_query": "{edge} < 0", "column_id": "edge"},
                         "color": C["loss"]},
                        {"if": {"filter_query": "{score_k20} > 0", "column_id": "score_k20"},
                         "backgroundColor": "rgba(34,197,94,0.18)", "color": C["profit"]},
                        {"if": {"filter_query": "{score_k20} < 0", "column_id": "score_k20"},
                         "backgroundColor": "rgba(239,68,68,0.18)", "color": C["loss"]},
                    ],
                ),
                html.Div([
                    html.Span("edge", style={"color": C["alt"], "fontWeight": "600"}),
                    " = side*(mid - price). Positive ⇒ traded favorably.   ",
                    html.Span("score k=20", style={"color": C["alt"], "fontWeight": "600"}),
                    " = side*(mid_(t+20) - mid_t). Positive ⇒ direction was right at horizon 20 ticks.",
                ], style={"fontSize": "11px", "color": C["dim"], "marginTop": "8px"}),
            ]),

            card("PnL attribution by product",
                 "b-pnl-prod", "360px"),
            card("Realized edge by counterparty  (who Mark X makes / loses money against)",
                 "b-pnl-cp", "360px"),
            card("Good-trade rate (rolling, %)  -  multiple horizons",
                 "b-good-rate", "380px"),
            card("Forward-return score distribution  (k=20 ticks)",
                 "b-score-dist", "340px"),
        ]),

        html.Div(style={"padding": "10px 28px 22px", "fontSize": "11px",
                        "color": C["dim"]}, children=[
            "Counterparty colors: ",
            *[html.Span(f" {m} ", style={
                "color": MARK_COLOR.get(m, C["text"]),
                "fontWeight": "600", "marginRight": "6px",
            }) for m in marks_present],
        ]),
    ])

    app.index_string = '''<!DOCTYPE html><html><head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
    *{box-sizing:border-box}body{margin:0}
    ::-webkit-scrollbar{width:6px}
    ::-webkit-scrollbar-track{background:#0a0e17}
    ::-webkit-scrollbar-thumb{background:#1f2937;border-radius:3px}
    .Select-control{background-color:#111827!important;border-color:#1f2937!important}
    .Select-menu-outer{background-color:#111827!important;border-color:#1f2937!important}
    .Select-option{background-color:#111827!important;color:#f1f5f9!important}
    .Select-option.is-focused{background-color:#1f2937!important}
    .Select-value-label{color:#f1f5f9!important}
    .Select-input input{color:#f1f5f9!important}
    </style></head><body>
    {%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body></html>'''

    # ---------- callback ----------
    @app.callback(
        [Output("b-cum-all", "figure"),
         Output("b-position", "figure"),
         Output("b-trade-log", "figure"),
         Output("b-trade-table", "data"),
         Output("b-pnl-prod", "figure"),
         Output("b-pnl-cp", "figure"),
         Output("b-good-rate", "figure"),
         Output("b-score-dist", "figure")],
        [Input("b-mark", "value"), Input("b-product", "value"),
         Input("b-day", "value"), Input("b-window", "value")],
    )
    def update(mark, product, day, window):
        if not mark:
            empty = _empty_fig("Select a Mark to see details")
            return empty, empty, empty, [], empty, empty, empty, empty
        f1 = fig_cum_pnl_all_marks(pnl_curves)
        f2 = fig_position_per_mark(pos_curves, mark)
        f3 = fig_trade_log(detailed, prices, mark, product, day)
        # Trade table mirrors the same product/day selection but pass
        # "__all__" sentinels so the user can override with the table's
        # native filter row to broaden the view.
        table_rows = trade_table_data(detailed, mark,
                                       product_filter=product,
                                       day_filter=day)
        f4 = fig_pnl_by_product(detailed, prices, mark)
        f5 = fig_pnl_by_counterparty(detailed, mark)
        f6 = fig_good_trade_rate_rolling(detailed, mark, window=window or 30)
        f7 = fig_score_distribution(detailed, mark, k=20)
        return f1, f2, f3, table_rows, f4, f5, f6, f7

    return app


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Round 4 Bot Performance Scanner")
    parser.add_argument("--data", "-d", default=None,
                        help="Data folder (default: auto-detect Round 4)")
    parser.add_argument("--port", "-p", type=int, default=8090)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    data_dir = args.data or find_data_folder()
    if not data_dir:
        print("No data folder found. Pass --data DIR.")
        sys.exit(1)

    print(f"Loading: {data_dir}")
    prices, trades = load_data(data_dir)
    print(f"  Prices: {len(prices):,} rows | Trades: {len(trades):,} rows")
    print(f"  Days: {sorted(prices['day_label'].unique())}")

    app = create_app(prices, trades)

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    print(f"\n  http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
