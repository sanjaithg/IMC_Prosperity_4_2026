"""
IMC Prosperity 4 - Round 4 Featured Signal Visualizer
======================================================
Standalone Dash app focused on counterparty-conditioned and options-pricing
signals derived from the Round 4 dataset. Reuses load_data and BS helpers
from data_visualizer.py without modifying it.

Run side-by-side with the base visualizer (different port):
    python datavisualiser_featured.py                    # port 8060
    python datavisualiser_featured.py --port 8061
    python datavisualiser_featured.py --data Dataset/ROUND_4

Signal plots
------------
  1. Per-Mark cumulative VWAP - mid  (who's making spread vs paying it)
  2. Dealer book imbalance           (Mark 22 sold cum - Mark 01 bought cum,
                                      across voucher chain = synthetic vol)
  3. Mark activity heatmap           (Mark x time bucket, trade count)
  4. Mark-conditioned forward returns
                                     (mean spot move at k ticks after each
                                      Mark's buy vs sell, vs random baseline)
  5. ATM IV vs realized vol          (fitted ATM IV vs Parkinson RV)
  6. Voucher fair-value residual     (BS price - market mid, by strike)
  7. Per-Mark net qty (selected product)
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from threading import Timer

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_visualizer import (  # noqa: E402  reuse — do NOT mutate base file
    load_data, find_data_folder,
    KNOWN_MARKS, MARK_COLOR,
    bs_call_price_delta_gamma_vega, implied_vol_call,
    calc_realized_vol_parkinson, calc_realized_vol_close_to_close,
)

# --- Theme (matches base visualizer aesthetic) ----------------------------
C = {
    "bg": "#0a0e17", "card": "#111827", "border": "#1f2937",
    "text": "#f1f5f9", "dim": "#64748b",
    "buy": "#34d399", "sell": "#f87171",
    "mid": "#38bdf8", "alt": "#fbbf24",
    "profit": "#22c55e", "loss": "#ef4444",
}
BASE = dict(
    plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
    font=dict(color=C["text"], size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"]),
    legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=C["card"], font_size=11),
)
TOOLBAR = dict(
    displaylogo=False,
    modeBarButtonsToAdd=["drawline", "drawrect", "eraseshape"],
    toImageButtonOptions=dict(format="png", scale=2),
)

UNDERLYING = "VELVETFRUIT_EXTRACT"
VOUCHER_RE = r"^VEV_(\d+)$"


# =====================================================================
# CARD WRAPPER
# =====================================================================
def card(title, gid, h="360px"):
    return html.Div(style={
        "background": C["card"], "border": f"1px solid {C['border']}",
        "borderRadius": "8px", "padding": "12px 16px",
    }, children=[
        html.Div(title, style={
            "fontSize": "13px", "fontWeight": "700",
            "letterSpacing": "1px", "color": C["text"],
            "marginBottom": "6px",
        }),
        dcc.Graph(id=gid, style={"height": h}, config=TOOLBAR),
    ])


# =====================================================================
# SIGNAL COMPUTATIONS
# =====================================================================
def fig_mark_vwap_edge(td: pd.DataFrame, mid_lookup: pd.Series,
                       product: str) -> go.Figure:
    """
    Per-Mark cumulative (VWAP - mid) edge for the selected product.
    Side-aware:
      - When the Mark is the BUYER, edge per fill = (mid - price) * qty
        (positive => bought below mid).
      - When the Mark is the SELLER, edge per fill = (price - mid) * qty
        (positive => sold above mid).
    Cumulating that gives total spread captured (or paid) over time.
    """
    fig = go.Figure()
    if td.empty:
        fig.add_annotation(text="No trades for selected product/day",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=C["dim"], size=14))
        fig.update_layout(**BASE)
        return fig

    td = td.copy()
    td["mid"] = td["timestamp"].map(mid_lookup)
    td = td.dropna(subset=["mid"])
    if td.empty:
        fig.update_layout(**BASE,
                          title="No mid data aligned to trade timestamps")
        return fig

    for mark in KNOWN_MARKS:
        as_buyer = td[td["buyer"] == mark].copy()
        as_buyer["signed_edge"] = (as_buyer["mid"] - as_buyer["price"]) * as_buyer["quantity"]
        as_seller = td[td["seller"] == mark].copy()
        as_seller["signed_edge"] = (as_seller["price"] - as_seller["mid"]) * as_seller["quantity"]
        events = pd.concat([as_buyer[["time_s", "signed_edge"]],
                            as_seller[["time_s", "signed_edge"]]])
        if events.empty:
            continue
        events = events.sort_values("time_s")
        events["cum_edge"] = events["signed_edge"].cumsum()
        fig.add_trace(go.Scatter(
            x=events["time_s"], y=events["cum_edge"], mode="lines",
            line=dict(color=MARK_COLOR.get(mark, "#94a3b8"), width=1.6),
            name=mark,
            hovertemplate=f"{mark}<br>t=%{{x:.0f}}s<br>cum edge=%{{y:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=C["border"], line_width=1)
    fig.update_layout(**BASE,
                      xaxis_title="Time (s)",
                      yaxis_title=f"Cumulative (VWAP - mid) * qty  for  {product}")
    return fig


def fig_dealer_imbalance(trades_day: pd.DataFrame) -> go.Figure:
    """
    Cumulative (Mark 22 sold qty) - (Mark 01 bought qty) summed across the
    voucher chain over the selected day. Positive = dealer net short more
    OTM call exposure than they've offset on the long-call leg.
    """
    fig = go.Figure()
    vouchers = trades_day[trades_day["symbol"].str.match(VOUCHER_RE)].sort_values("time_s").copy()
    if vouchers.empty:
        fig.update_layout(**BASE, title="No voucher trades on selected day")
        return fig

    vouchers["m22_sold"] = np.where(vouchers["seller"] == "Mark 22",
                                     vouchers["quantity"], 0)
    vouchers["m01_bought"] = np.where(vouchers["buyer"] == "Mark 01",
                                       vouchers["quantity"], 0)
    vouchers["delta"] = vouchers["m22_sold"] - vouchers["m01_bought"]
    vouchers["cum_imbalance"] = vouchers["delta"].cumsum()
    vouchers["cum_m22"] = vouchers["m22_sold"].cumsum()
    vouchers["cum_m01"] = vouchers["m01_bought"].cumsum()

    fig.add_trace(go.Scatter(x=vouchers["time_s"], y=vouchers["cum_m22"],
                             mode="lines", line=dict(color=MARK_COLOR["Mark 22"], width=1.4, dash="dot"),
                             name="Mark 22 cum sold"))
    fig.add_trace(go.Scatter(x=vouchers["time_s"], y=vouchers["cum_m01"],
                             mode="lines", line=dict(color=MARK_COLOR["Mark 01"], width=1.4, dash="dot"),
                             name="Mark 01 cum bought"))
    fig.add_trace(go.Scatter(x=vouchers["time_s"], y=vouchers["cum_imbalance"],
                             mode="lines", line=dict(color=C["alt"], width=2.2),
                             name="Imbalance = sold - bought"))
    fig.add_hline(y=0, line_color=C["border"], line_width=1)
    fig.update_layout(**BASE,
                      xaxis_title="Time (s)",
                      yaxis_title="Cumulative voucher qty (across all strikes)")
    return fig


def fig_mark_activity_heatmap(td_day: pd.DataFrame,
                               n_buckets: int = 50) -> go.Figure:
    """Mark x time-bucket trade-count heatmap for the selected DAY (all products)."""
    fig = go.Figure()
    if td_day.empty:
        fig.update_layout(**BASE, title="No trades")
        return fig

    t_min, t_max = td_day["time_s"].min(), td_day["time_s"].max()
    if t_max <= t_min:
        fig.update_layout(**BASE, title="Insufficient time range")
        return fig
    bucket_w = (t_max - t_min) / n_buckets
    td_day = td_day.copy()
    td_day["bucket"] = ((td_day["time_s"] - t_min) // bucket_w).clip(upper=n_buckets - 1)

    # Count trades where each Mark appears (buyer OR seller).
    counts = np.zeros((len(KNOWN_MARKS), n_buckets))
    for i, mark in enumerate(KNOWN_MARKS):
        sub = td_day[(td_day["buyer"] == mark) | (td_day["seller"] == mark)]
        if sub.empty:
            continue
        c = sub.groupby("bucket").size().reindex(range(n_buckets), fill_value=0)
        counts[i] = c.values

    bucket_centers = t_min + (np.arange(n_buckets) + 0.5) * bucket_w
    fig.add_trace(go.Heatmap(
        z=counts, x=bucket_centers, y=KNOWN_MARKS,
        colorscale="Viridis",
        hovertemplate="mark=%{y}<br>t~%{x:.0f}s<br>trades=%{z}<extra></extra>",
        colorbar=dict(title="trades"),
    ))
    fig.update_layout(**BASE,
                      xaxis_title="Time (s)",
                      yaxis_title="Counterparty (Mark)")
    return fig


def fig_mark_forward_returns(td_day: pd.DataFrame,
                              spot_mid_by_ts: pd.Series,
                              horizons=(1, 5, 20, 50)) -> go.Figure:
    """
    For each Mark, plot mean forward spot move at k ticks after their fill,
    separated by buy and sell side. One sub-bar per horizon.
    """
    fig = go.Figure()
    if td_day.empty or spot_mid_by_ts.empty:
        fig.update_layout(**BASE, title="No data for forward-return computation")
        return fig

    spot_velvet = td_day[td_day["symbol"] == UNDERLYING]
    if spot_velvet.empty:
        fig.update_layout(**BASE, title=f"No {UNDERLYING} trades on selected day")
        return fig

    rows = []
    for mark in KNOWN_MARKS:
        for side, sub in (("buy", spot_velvet[spot_velvet["buyer"] == mark]),
                          ("sell", spot_velvet[spot_velvet["seller"] == mark])):
            if sub.empty:
                continue
            for k in horizons:
                fwd = []
                for ts in sub["timestamp"]:
                    nxt = spot_mid_by_ts.get(ts + 100 * k, np.nan)
                    cur = spot_mid_by_ts.get(ts, np.nan)
                    if not (np.isnan(nxt) or np.isnan(cur)):
                        fwd.append(nxt - cur)
                if fwd:
                    rows.append(dict(mark=mark, side=side, k=k,
                                     mean=float(np.mean(fwd)),
                                     n=len(fwd)))
    if not rows:
        fig.update_layout(**BASE, title="Insufficient samples for any Mark")
        return fig
    df = pd.DataFrame(rows)

    # Bar groups: x = mark, color = side+k. Compactly, one trace per (side, k).
    for side in ("buy", "sell"):
        for k in horizons:
            sub = df[(df["side"] == side) & (df["k"] == k)]
            if sub.empty:
                continue
            color = C["buy"] if side == "buy" else C["sell"]
            opacity = {1: 0.45, 5: 0.6, 20: 0.78, 50: 0.95}.get(k, 0.7)
            fig.add_trace(go.Bar(
                x=sub["mark"], y=sub["mean"],
                name=f"{side} k={k}",
                marker_color=color, opacity=opacity,
                hovertemplate=(f"%{{x}} | {side} | k={k}<br>"
                               "mean fwd move=%{y:+.3f}<br>"
                               "n=%{customdata}<extra></extra>"),
                customdata=sub["n"],
            ))
    fig.add_hline(y=0, line_color=C["border"], line_width=1)
    fig.update_layout(**BASE, barmode="group",
                      xaxis_title="Mark (counterparty)",
                      yaxis_title=f"Mean forward {UNDERLYING} mid move (price units)")
    return fig


def fig_iv_vs_rv(prices_day: pd.DataFrame, day_int: int) -> go.Figure:
    """
    Time series of fitted ATM IV (from VEV_5200/5300/5400 chain near ATM)
    vs Parkinson realized vol on the underlying.
    Both annualized to compare.
    """
    fig = go.Figure()
    und = prices_day[prices_day["product"] == UNDERLYING].sort_values("timestamp").reset_index(drop=True)
    if und.empty:
        fig.update_layout(**BASE, title=f"No {UNDERLYING} prices on day")
        return fig

    # Pull TTE from the day's voucher rows (already populated by load_data).
    voucher_rows = prices_day[prices_day["product"].str.match(VOUCHER_RE)]
    if voucher_rows.empty:
        fig.update_layout(**BASE, title="No voucher prices on day")
        return fig
    tte_days = float(voucher_rows["tte_days"].iloc[0])
    T = max(tte_days / 365.0, 1e-6)

    # Fit ATM IV at each underlying tick: pick the strike nearest current mid
    # from {5200, 5300, 5400}, get its mid, invert BS.
    atm_strikes = [5200, 5300, 5400]
    voucher_mids = {}
    for K in atm_strikes:
        sym = f"VEV_{K}"
        v = prices_day[prices_day["product"] == sym].sort_values("timestamp")
        if v.empty:
            continue
        voucher_mids[K] = v.set_index("timestamp")["mid_price"]

    if not voucher_mids:
        fig.update_layout(**BASE, title="No ATM voucher data")
        return fig

    iv_ts, iv_vals = [], []
    # Sub-sample for speed on long days.
    step = max(1, len(und) // 1000)
    for _, row in und.iloc[::step].iterrows():
        ts = int(row["timestamp"])
        S = float(row["mid_price"])
        # Closest strike to S among atm_strikes that has a quote at ts (or near).
        best_iv = np.nan
        for K in atm_strikes:
            if K not in voucher_mids:
                continue
            ser = voucher_mids[K]
            try:
                idx = ser.index.get_indexer([ts], method="nearest")[0]
                if idx < 0:
                    continue
                price = float(ser.iloc[idx])
            except Exception:
                continue
            iv = implied_vol_call(price, S, float(K), T)
            if not np.isnan(iv) and 0.05 <= iv <= 1.5:
                best_iv = iv
                break
        if not np.isnan(best_iv):
            iv_ts.append(ts / 100.0)
            iv_vals.append(best_iv * 100.0)

    if iv_vals:
        fig.add_trace(go.Scatter(x=iv_ts, y=iv_vals, mode="lines",
                                 line=dict(color=C["alt"], width=2),
                                 name="ATM IV (fitted, %)"))

    # Parkinson RV on the underlying (if high/low not available, fall back to
    # close-to-close on mid).
    if {"bid_price_1", "ask_price_1"}.issubset(und.columns):
        hl = pd.DataFrame({
            "high": und["ask_price_1"], "low": und["bid_price_1"]
        })
        rv = calc_realized_vol_parkinson(hl, window_ticks=200)
    else:
        rv = calc_realized_vol_close_to_close(und["mid_price"], window_ticks=200)
    if rv is not None and len(rv) > 0:
        # Annualize: Prosperity day == 10000 ticks. Interpret window=200 ticks.
        # Both helpers return per-period vol; multiply by sqrt(periods/year).
        # Treat one Prosperity day as 1/365 of a year for simplicity.
        ann = np.sqrt(365.0 * (10000.0 / 200.0))
        fig.add_trace(go.Scatter(
            x=und["timestamp"] / 100.0, y=(np.asarray(rv, dtype=float) * ann * 100.0),
            mode="lines", line=dict(color=C["mid"], width=1.4),
            name="Realized vol (Parkinson, ann %)",
        ))

    fig.update_layout(**BASE,
                      xaxis_title="Time (s)",
                      yaxis_title="Vol (% annualized)")
    return fig


def fig_voucher_residual(prices_day: pd.DataFrame) -> go.Figure:
    """
    Per-strike (BS price under fitted ATM IV) - (market mid) over the day.
    A positive residual at strike K means BS thinks K is worth more than the
    market — i.e. that strike is CHEAP and you'd buy it.
    """
    fig = go.Figure()
    und = prices_day[prices_day["product"] == UNDERLYING].sort_values("timestamp").reset_index(drop=True)
    voucher_prods = sorted(prices_day.loc[prices_day["product"].str.match(VOUCHER_RE), "product"].unique(),
                           key=lambda s: int(s.split("_")[1]))
    if und.empty or not voucher_prods:
        fig.update_layout(**BASE, title="Need underlying + voucher data")
        return fig

    tte_days = float(prices_day.loc[prices_day["product"].str.match(VOUCHER_RE), "tte_days"].iloc[0])
    T = max(tte_days / 365.0, 1e-6)
    spot_at = und.set_index("timestamp")["mid_price"]

    # Single mid-day fitted IV from VEV_5300 (closest to typical S~5245).
    fit_sym = "VEV_5300" if "VEV_5300" in voucher_prods else voucher_prods[len(voucher_prods) // 2]
    fit_v = prices_day[prices_day["product"] == fit_sym].sort_values("timestamp")
    if fit_v.empty:
        fig.update_layout(**BASE, title="No data to fit IV")
        return fig
    mid_idx = len(fit_v) // 2
    S_fit = float(spot_at.reindex([fit_v.iloc[mid_idx]["timestamp"]]).iloc[0])
    K_fit = float(fit_sym.split("_")[1])
    P_fit = float(fit_v.iloc[mid_idx]["mid_price"])
    sigma = implied_vol_call(P_fit, S_fit, K_fit, T)
    if np.isnan(sigma) or sigma <= 0:
        sigma = 0.5
    title_iv = f"fitted sigma = {sigma:.3f}  (from {fit_sym} mid-day)"

    for sym in voucher_prods:
        K = float(sym.split("_")[1])
        v = prices_day[prices_day["product"] == sym].sort_values("timestamp").reset_index(drop=True)
        if v.empty:
            continue
        # Compute residual at each tick (sub-sample).
        step = max(1, len(v) // 1000)
        rows = v.iloc[::step]
        ts = rows["timestamp"].values
        S_arr = spot_at.reindex(ts).ffill().bfill().values
        bs = np.array([bs_call_price_delta_gamma_vega(s, K, T, sigma)[0]
                       for s in S_arr])
        residual = bs - rows["mid_price"].values
        fig.add_trace(go.Scatter(
            x=rows["timestamp"] / 100.0, y=residual, mode="lines",
            name=f"{sym} (K={int(K)})",
            line=dict(width=1.3),
            hovertemplate=f"{sym}<br>t=%{{x:.0f}}s<br>BS-mid=%{{y:+.2f}}<extra></extra>",
        ))
    fig.add_hline(y=0, line_color=C["border"], line_width=1)
    fig.update_layout(**BASE,
                      title=title_iv,
                      xaxis_title="Time (s)",
                      yaxis_title="BS price - market mid  (positive = cheap)")
    return fig


def fig_per_mark_net_qty(td_pd: pd.DataFrame, product: str) -> go.Figure:
    """Cumulative net qty (buy - sell) per Mark for the selected product."""
    fig = go.Figure()
    if td_pd.empty:
        fig.update_layout(**BASE, title=f"No trades for {product}")
        return fig
    for mark in KNOWN_MARKS:
        buys = td_pd[td_pd["buyer"] == mark][["time_s", "quantity"]].copy()
        sells = td_pd[td_pd["seller"] == mark][["time_s", "quantity"]].copy()
        if buys.empty and sells.empty:
            continue
        sells["quantity"] = -sells["quantity"]
        events = pd.concat([buys, sells]).sort_values("time_s")
        events["cum"] = events["quantity"].cumsum()
        fig.add_trace(go.Scatter(
            x=events["time_s"], y=events["cum"], mode="lines",
            line=dict(color=MARK_COLOR.get(mark, "#94a3b8"), width=1.5),
            name=mark,
        ))
    fig.add_hline(y=0, line_color=C["border"], line_width=1)
    fig.update_layout(**BASE,
                      xaxis_title="Time (s)",
                      yaxis_title=f"Cumulative net qty in {product}")
    return fig


# =====================================================================
# SPREAD PLOTS
# =====================================================================
def fig_spread_dist_by_product(prices_day: pd.DataFrame) -> go.Figure:
    """Box per product of L1 bid-ask spread distribution for the day."""
    fig = go.Figure()
    if prices_day.empty:
        fig.update_layout(**BASE, title="No data")
        return fig
    pdf = prices_day.dropna(subset=["bid_price_1", "ask_price_1"]).copy()
    pdf["spread"] = pdf["ask_price_1"] - pdf["bid_price_1"]
    # Sort: spot first, then vouchers by strike
    def order_key(p):
        if p.startswith("VEV_"):
            return (1, int(p.split("_")[1]))
        return (0, p)
    products = sorted(pdf["product"].unique(), key=order_key)
    for prod in products:
        s = pdf.loc[pdf["product"] == prod, "spread"]
        if s.empty:
            continue
        fig.add_trace(go.Box(
            y=s, name=prod, boxpoints=False,
            marker_color=C["mid"] if prod == UNDERLYING else C["alt"],
            line=dict(width=1),
            hovertemplate=f"{prod}<br>spread=%{{y}}<extra></extra>",
        ))
    fig.update_layout(**BASE, showlegend=False,
                      xaxis_title="Product",
                      yaxis_title="Bid-Ask spread (price units)")
    return fig


def fig_rv_comparison(prices_day: pd.DataFrame, window_ticks: int = 200) -> go.Figure:
    """
    Compare rolling Parkinson RV (annualized %) across all products for the selected day.
    """
    fig = go.Figure()
    if prices_day.empty:
        fig.update_layout(**BASE, title="No data")
        return fig

    # Annualization factor: Prosperity day ~ 10,000 ticks.
    ann = np.sqrt(365.0 * (10000.0 / window_ticks))

    # Sort products: Underlying first, then VEVs, then others
    def order_key(p):
        if p == UNDERLYING: return (0, p)
        if p.startswith("VEV_"): return (1, int(p.split("_")[1]))
        return (2, p)

    products = sorted(prices_day["product"].unique(), key=order_key)

    palette = ["#38bdf8", "#34d399", "#fbbf24", "#fb923c", "#f87171",
               "#a78bfa", "#f472b6", "#22d3ee", "#facc15", "#94a3b8",
               "#84cc16", "#06b6d4"]

    for i, prod in enumerate(products):
        sub = prices_day[prices_day["product"] == prod].sort_values("timestamp")
        if sub.empty or "bid_price_1" not in sub.columns or "ask_price_1" not in sub.columns:
            continue

        hl = pd.DataFrame({
            "high": sub["ask_price_1"], "low": sub["bid_price_1"]
        })
        rv = calc_realized_vol_parkinson(hl, window_ticks=window_ticks)
        if rv is not None and len(rv) > 0:
            fig.add_trace(go.Scatter(
                x=sub["timestamp"] / 100.0,
                y=(np.asarray(rv, dtype=float) * ann * 100.0),
                mode="lines",
                name=prod,
                line=dict(color=palette[i % len(palette)], width=1.4),
                hovertemplate=f"<b>{prod}</b><br>t=%{{x:.0f}}s<br>RV=%{{y:.1f}}%<extra></extra>"
            ))

    fig.update_layout(**BASE,
                      xaxis_title="Time (s)",
                      yaxis_title="Realized Vol (% annualized)",
                      hovermode="x unified")
    return fig


def fig_per_mark_half_spread(td_pd: pd.DataFrame,
                              mid_lookup: pd.Series,
                              product: str) -> go.Figure:
    """
    Per-Mark distribution of effective half-spread = |price - mid| at trade time
    for the selected product. Aggressive takers cluster high; passive makers
    cluster near zero.
    """
    fig = go.Figure()
    if td_pd.empty:
        fig.update_layout(**BASE, title=f"No trades for {product}")
        return fig
    td = td_pd.copy()
    td["mid"] = td["timestamp"].map(mid_lookup)
    td = td.dropna(subset=["mid"])
    if td.empty:
        fig.update_layout(**BASE, title="No mid-aligned trades")
        return fig
    td["abs_half_spread"] = (td["price"] - td["mid"]).abs()
    # Stack rows: each Mark contributes |hsp| from rows where it's buyer or seller.
    rows = []
    for mark in KNOWN_MARKS:
        sub = td[(td["buyer"] == mark) | (td["seller"] == mark)]
        if sub.empty:
            continue
        for v in sub["abs_half_spread"].values:
            rows.append((mark, float(v)))
    if not rows:
        fig.update_layout(**BASE, title=f"No counterparty trades on {product}")
        return fig
    df = pd.DataFrame(rows, columns=["mark", "hsp"])
    for mark in KNOWN_MARKS:
        sub = df[df["mark"] == mark]
        if sub.empty:
            continue
        fig.add_trace(go.Violin(
            y=sub["hsp"], name=mark, box_visible=True, meanline_visible=True,
            line_color=MARK_COLOR.get(mark, "#94a3b8"),
            fillcolor=MARK_COLOR.get(mark, "#94a3b8"),
            opacity=0.55, points=False,
        ))
    fig.update_layout(**BASE, showlegend=False,
                      xaxis_title="Counterparty",
                      yaxis_title=f"|trade_price - mid|  on  {product}")
    return fig


def fig_spread_vs_activity(prices_day: pd.DataFrame,
                            trades_day: pd.DataFrame,
                            product: str) -> go.Figure:
    """
    Bin the day into 100 buckets, compute (mean L1 spread) and (trade count)
    per bucket for the selected product. Scatter the two; if spread widens
    with activity, market-makers raise quotes during volume bursts.
    """
    fig = go.Figure()
    p = prices_day[prices_day["product"] == product].copy()
    t = trades_day[trades_day["symbol"] == product].copy()
    if p.empty:
        fig.update_layout(**BASE, title=f"No prices for {product}")
        return fig
    p["spread"] = p["ask_price_1"] - p["bid_price_1"]
    t_min, t_max = p["time_s"].min(), p["time_s"].max()
    if t_max <= t_min:
        fig.update_layout(**BASE, title="Insufficient time range")
        return fig
    bw = (t_max - t_min) / 100.0
    p["b"] = ((p["time_s"] - t_min) // bw).clip(upper=99)
    t["b"] = ((t["time_s"] - t_min) // bw).clip(upper=99) if not t.empty else pd.Series(dtype=int)
    spread_b = p.groupby("b")["spread"].mean()
    count_b = (t.groupby("b").size() if not t.empty else pd.Series(0, index=spread_b.index))
    df = pd.concat([spread_b, count_b.rename("count")], axis=1).fillna(0)
    fig.add_trace(go.Scatter(
        x=df["count"], y=df["spread"], mode="markers",
        marker=dict(color=df.index, colorscale="Plasma", size=8,
                    showscale=True, colorbar=dict(title="time bucket")),
        hovertemplate="bucket=%{marker.color}<br>trades=%{x}<br>avg spread=%{y:.2f}<extra></extra>",
    ))
    if (df["count"] > 0).any() and df["spread"].std() > 0:
        x = df["count"].values.astype(float); y = df["spread"].values.astype(float)
        try:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept,
                                     mode="lines", name=f"trend: slope={slope:+.4f}",
                                     line=dict(color=C["alt"], dash="dash")))
        except Exception:
            pass
    fig.update_layout(**BASE,
                      xaxis_title=f"Trade count per ~{bw:.0f}s bucket",
                      yaxis_title="Avg L1 spread per bucket")
    return fig


# =====================================================================
# IV PLOTS
# =====================================================================
def _voucher_iv_grid(prices_day: pd.DataFrame, n_samples: int = 80):
    """
    Compute IV(t, K) sub-sampled across the day for all VEV strikes.
    Returns: (timestamps_sec, strikes, iv_matrix [n_strikes x n_samples], spot_at_t).
    """
    und = prices_day[prices_day["product"] == UNDERLYING].sort_values("timestamp").reset_index(drop=True)
    voucher_prods = sorted(prices_day.loc[prices_day["product"].str.match(VOUCHER_RE), "product"].unique(),
                           key=lambda s: int(s.split("_")[1]))
    if und.empty or not voucher_prods:
        return None
    tte_days = float(prices_day.loc[prices_day["product"].str.match(VOUCHER_RE), "tte_days"].iloc[0])
    T = max(tte_days / 365.0, 1e-6)
    # Sample timestamps
    if len(und) <= n_samples:
        sample_idx = list(range(len(und)))
    else:
        sample_idx = list(range(0, len(und), len(und) // n_samples))[:n_samples]
    times = und.iloc[sample_idx]["timestamp"].values
    spots = und.iloc[sample_idx]["mid_price"].values
    strikes = [int(s.split("_")[1]) for s in voucher_prods]
    voucher_idx = {}
    for prod in voucher_prods:
        v = prices_day[prices_day["product"] == prod].sort_values("timestamp")
        voucher_idx[prod] = v.set_index("timestamp")["mid_price"]
    iv = np.full((len(strikes), len(times)), np.nan)
    for i, prod in enumerate(voucher_prods):
        ser = voucher_idx[prod]
        if ser.empty:
            continue
        K = float(strikes[i])
        for j, (ts, S) in enumerate(zip(times, spots)):
            try:
                idx = ser.index.get_indexer([ts], method="nearest")[0]
                if idx < 0:
                    continue
                price = float(ser.iloc[idx])
            except Exception:
                continue
            v = implied_vol_call(price, float(S), K, T)
            if not np.isnan(v) and 0.05 <= v <= 1.5:
                iv[i, j] = v * 100.0  # store as %
    return dict(times_sec=times / 100.0, strikes=strikes, iv=iv, spots=spots)


def fig_iv_smile_snapshots(prices_day: pd.DataFrame,
                            n_snapshots: int = 5) -> go.Figure:
    """N evenly-spaced snapshots of the IV smile (IV vs strike)."""
    fig = go.Figure()
    grid = _voucher_iv_grid(prices_day, n_samples=80)
    if grid is None:
        fig.update_layout(**BASE, title="No IV data")
        return fig
    iv, strikes, times_sec, spots = grid["iv"], grid["strikes"], grid["times_sec"], grid["spots"]
    # Pick snapshot columns evenly across time
    n = iv.shape[1]
    pick = np.linspace(0, n - 1, n_snapshots).astype(int)
    cmap = ["#3b82f6", "#22d3ee", "#facc15", "#f97316", "#ef4444"]
    for ci, j in enumerate(pick):
        col = iv[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        ts = times_sec[j]; S = spots[j]
        fig.add_trace(go.Scatter(
            x=np.array(strikes)[valid], y=col[valid],
            mode="lines+markers",
            line=dict(color=cmap[ci % len(cmap)], width=2),
            marker=dict(size=7),
            name=f"t={ts:.0f}s  S={S:.0f}",
        ))
    fig.update_layout(**BASE, xaxis_title="Strike", yaxis_title="Implied vol (%)")
    return fig


def fig_per_strike_iv_ts(prices_day: pd.DataFrame) -> go.Figure:
    """One IV(t) line per strike."""
    fig = go.Figure()
    grid = _voucher_iv_grid(prices_day, n_samples=120)
    if grid is None:
        fig.update_layout(**BASE, title="No IV data")
        return fig
    iv, strikes, times_sec = grid["iv"], grid["strikes"], grid["times_sec"]
    palette = ["#60a5fa", "#34d399", "#fbbf24", "#fb923c", "#f87171",
               "#a78bfa", "#f472b6", "#22d3ee", "#facc15", "#94a3b8"]
    for i, K in enumerate(strikes):
        col = iv[i]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        fig.add_trace(go.Scatter(
            x=times_sec[valid], y=col[valid], mode="lines",
            name=f"K={K}",
            line=dict(color=palette[i % len(palette)], width=1.4),
        ))
    fig.update_layout(**BASE, xaxis_title="Time (s)", yaxis_title="IV (%)")
    return fig


def fig_iv_skew_curvature(prices_day: pd.DataFrame) -> go.Figure:
    """
    Skew = IV(K_high) - IV(K_low)   (positive => OTM call IV > ITM call IV)
    Curvature (butterfly) = IV(K_high) + IV(K_low) - 2*IV(K_mid)
    Built from K_low=5100, K_mid=5300, K_high=5500 (or nearest available).
    """
    fig = go.Figure()
    grid = _voucher_iv_grid(prices_day, n_samples=120)
    if grid is None:
        fig.update_layout(**BASE, title="No IV data")
        return fig
    strikes = grid["strikes"]
    desired = [5100, 5300, 5500]
    indices = []
    for d in desired:
        if d in strikes:
            indices.append(strikes.index(d))
        else:
            indices.append(int(np.argmin(np.abs(np.array(strikes) - d))))
    iL, iM, iH = indices
    iv = grid["iv"]
    skew = iv[iH] - iv[iL]
    curv = iv[iH] + iv[iL] - 2 * iv[iM]
    t = grid["times_sec"]
    fig.add_trace(go.Scatter(x=t, y=skew, mode="lines", name=f"skew (K{strikes[iH]} - K{strikes[iL]})",
                             line=dict(color=C["alt"], width=1.6)))
    fig.add_trace(go.Scatter(x=t, y=curv, mode="lines", name=f"curvature (K{strikes[iH]} + K{strikes[iL]} - 2*K{strikes[iM]})",
                             line=dict(color=C["mid"], width=1.6)))
    fig.add_hline(y=0, line_color=C["border"], line_width=1)
    fig.update_layout(**BASE, xaxis_title="Time (s)",
                      yaxis_title="IV difference (%)")
    return fig


# =====================================================================
# OTHER PLOTS
# =====================================================================
def fig_volume_per_strike(trades_day: pd.DataFrame) -> go.Figure:
    """Total qty traded per voucher strike, on the selected day."""
    fig = go.Figure()
    voucher_t = trades_day[trades_day["symbol"].str.match(VOUCHER_RE)].copy()
    if voucher_t.empty:
        fig.update_layout(**BASE, title="No voucher trades")
        return fig
    voucher_t["strike"] = voucher_t["symbol"].str.extract(VOUCHER_RE).astype(int)
    by_strike = voucher_t.groupby("strike")["quantity"].sum().sort_index()
    fig.add_trace(go.Bar(
        x=by_strike.index.astype(str), y=by_strike.values,
        marker_color=C["alt"], opacity=0.85,
        text=[f"{int(v):,}" for v in by_strike.values],
        textposition="outside",
    ))
    fig.update_layout(**BASE, xaxis_title="Strike", yaxis_title="Total qty traded")
    return fig


def fig_per_mark_clip_dist(trades_day: pd.DataFrame) -> go.Figure:
    """Distribution of trade quantities per Mark (violin) for the selected day."""
    fig = go.Figure()
    if trades_day.empty:
        fig.update_layout(**BASE, title="No trades")
        return fig
    rows = []
    for mark in KNOWN_MARKS:
        sub = trades_day[(trades_day["buyer"] == mark) | (trades_day["seller"] == mark)]
        if sub.empty:
            continue
        for q in sub["quantity"].astype(float).values:
            rows.append((mark, q))
    if not rows:
        fig.update_layout(**BASE, title="No counterparty trades")
        return fig
    df = pd.DataFrame(rows, columns=["mark", "qty"])
    for mark in KNOWN_MARKS:
        sub = df[df["mark"] == mark]
        if sub.empty:
            continue
        fig.add_trace(go.Violin(
            y=sub["qty"], name=mark, points=False,
            box_visible=True, meanline_visible=True,
            line_color=MARK_COLOR.get(mark, "#94a3b8"),
            fillcolor=MARK_COLOR.get(mark, "#94a3b8"),
            opacity=0.55,
        ))
    fig.update_layout(**BASE, showlegend=False,
                      xaxis_title="Counterparty",
                      yaxis_title="Trade quantity (clip size)")
    return fig


def fig_voucher_greeks_eod(prices_day: pd.DataFrame) -> go.Figure:
    """End-of-day delta and gamma per voucher strike from BS at the day's close mid."""
    fig = go.Figure()
    und = prices_day[prices_day["product"] == UNDERLYING].sort_values("timestamp")
    voucher_prods = sorted(prices_day.loc[prices_day["product"].str.match(VOUCHER_RE), "product"].unique(),
                           key=lambda s: int(s.split("_")[1]))
    if und.empty or not voucher_prods:
        fig.update_layout(**BASE, title="No data")
        return fig
    tte_days = float(prices_day.loc[prices_day["product"].str.match(VOUCHER_RE), "tte_days"].iloc[0])
    T = max(tte_days / 365.0, 1e-6)
    S_eod = float(und["mid_price"].iloc[-1])
    rows = []
    for prod in voucher_prods:
        K = float(prod.split("_")[1])
        v = prices_day[prices_day["product"] == prod].sort_values("timestamp")
        if v.empty:
            continue
        market_mid = float(v["mid_price"].iloc[-1])
        sigma = implied_vol_call(market_mid, S_eod, K, T)
        if np.isnan(sigma) or sigma <= 0:
            sigma = 0.5
        bs_price, delta, gamma, vega = bs_call_price_delta_gamma_vega(S_eod, K, T, sigma)
        rows.append(dict(strike=int(K), sigma=sigma, delta=delta,
                         gamma=gamma, vega=vega))
    if not rows:
        fig.update_layout(**BASE, title="No greeks computed")
        return fig
    df = pd.DataFrame(rows).sort_values("strike")
    # Bar with secondary y-axis: delta on left (0..1), gamma on right
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["strike"].astype(str), y=df["delta"],
                          name="delta", marker_color=C["mid"], opacity=0.8),
                  secondary_y=False)
    fig.add_trace(go.Bar(x=df["strike"].astype(str), y=df["gamma"],
                          name="gamma", marker_color=C["alt"], opacity=0.8),
                  secondary_y=True)
    fig.update_layout(**BASE, barmode="group",
                      title=f"S_eod = {S_eod:.0f}, T = {tte_days:.0f}d")
    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="Delta", secondary_y=False, range=[0, 1.05])
    fig.update_yaxes(title_text="Gamma", secondary_y=True)
    return fig


# =====================================================================
# APP
# =====================================================================
def create_app(prices: pd.DataFrame, trades: pd.DataFrame) -> Dash:
    app = Dash(__name__, title="Round 4 - Featured Signals",
               suppress_callback_exceptions=True)

    products = sorted(prices["product"].unique()) if not prices.empty else []
    days = sorted(prices["day_label"].unique()) if not prices.empty else []
    default_product = UNDERLYING if UNDERLYING in products else (products[0] if products else None)
    default_day = days[0] if days else None

    app.layout = html.Div(style={
        "backgroundColor": C["bg"], "color": C["text"],
        "fontFamily": "Inter, -apple-system, sans-serif", "minHeight": "100vh",
    }, children=[
        # Header
        html.Div(style={
            "background": f"linear-gradient(135deg, {C['card']}, {C['bg']})",
            "borderBottom": f"1px solid {C['border']}",
            "padding": "16px 28px", "display": "flex",
            "justifyContent": "space-between", "alignItems": "center",
        }, children=[
            html.Span("PROSPERITY 4  -  ROUND 4 FEATURED SIGNALS", style={
                "fontSize": "18px", "fontWeight": "800",
                "letterSpacing": "3px", "color": C["mid"],
            }),
            html.Div(style={"display": "flex", "gap": "16px",
                            "alignItems": "center"}, children=[
                html.Div([
                    html.Label("Product", style={"fontSize": "10px",
                        "color": C["dim"], "textTransform": "uppercase",
                        "letterSpacing": "1px"}),
                    dcc.Dropdown(id="f-product",
                                 options=[{"label": p, "value": p} for p in products],
                                 value=default_product,
                                 style={"width": "180px"}),
                ]),
                html.Div([
                    html.Label("Day", style={"fontSize": "10px",
                        "color": C["dim"], "textTransform": "uppercase",
                        "letterSpacing": "1px"}),
                    dcc.Dropdown(id="f-day",
                                 options=[{"label": d, "value": d} for d in days],
                                 value=default_day,
                                 style={"width": "120px"}),
                ]),
                html.Div([
                    html.Label("Activity buckets", style={"fontSize": "10px",
                        "color": C["dim"], "textTransform": "uppercase",
                        "letterSpacing": "1px"}),
                    dcc.Slider(id="f-buckets", min=20, max=120, step=10, value=50,
                               marks={20: "20", 50: "50", 100: "100"},
                               tooltip={"placement": "bottom"}),
                ], style={"width": "240px"}),
            ]),
        ]),

        # Cards
        html.Div(style={"padding": "16px 28px", "display": "flex",
                        "flexDirection": "column", "gap": "14px"}, children=[
            card("1. Per-Mark cumulative VWAP - mid edge  (selected product/day)",
                 "f-edge", "380px"),
            card("2. Dealer book imbalance  (Mark 22 sold cum vs Mark 01 bought cum across voucher chain)",
                 "f-imb", "380px"),
            card("3. Mark activity heatmap  (selected day, all products)",
                 "f-heat", "320px"),
            card("4. Mark-conditioned forward spot returns  (k=1/5/20/50 ticks after fill)",
                 "f-fwd", "420px"),
            card("5. ATM IV  vs  realized vol  (annualized, %)",
                 "f-ivrv", "380px"),
            card("6. Voucher fair-value residual  (BS_price - market_mid by strike)",
                 "f-resid", "420px"),
            card("7. Per-Mark cumulative net qty  (selected product)",
                 "f-net", "360px"),

            html.Div("SPREAD signals", style={
                "fontSize": "12px", "fontWeight": "700",
                "letterSpacing": "2px", "color": C["dim"],
                "marginTop": "10px", "marginBottom": "-4px",
            }),
            card("8. Spread distribution by product  (boxplot, selected day)",
                 "f-spread-dist", "360px"),
            card("9. Per-Mark effective half-spread |price - mid|  (selected product)",
                 "f-mark-hsp", "380px"),
            card("10. Spread vs trade activity  (per-bucket scatter, selected product)",
                 "f-spread-act", "360px"),

            html.Div("IV signals", style={
                "fontSize": "12px", "fontWeight": "700",
                "letterSpacing": "2px", "color": C["dim"],
                "marginTop": "10px", "marginBottom": "-4px",
            }),
            card("11. Volatility smile snapshots through the day  (IV vs strike)",
                 "f-smile-snaps", "400px"),
            card("12. Per-strike implied vol time series  (one line per strike)",
                 "f-strike-iv", "400px"),
            card("13. IV skew & curvature time series  (wing - body, butterfly)",
                 "f-iv-skew", "360px"),

            html.Div("OTHER signals", style={
                "fontSize": "12px", "fontWeight": "700",
                "letterSpacing": "2px", "color": C["dim"],
                "marginTop": "10px", "marginBottom": "-4px",
            }),
            card("14. Total voucher qty per strike  (selected day)",
                 "f-vol-strike", "340px"),
            card("15. Per-Mark trade clip-size distribution  (selected day)",
                 "f-mark-clip", "360px"),
            card("16. End-of-day delta & gamma per strike  (selected day)",
                 "f-greeks-eod", "360px"),
            card("17. Realized volatility comparison across all products (annualized %)",
                 "f-rv-comp", "400px"),
        ]),

        html.Div(style={"padding": "10px 28px 20px", "fontSize": "11px",
                        "color": C["dim"]}, children=[
            "Counterparty colors: ",
            *[html.Span(f" {m} ", style={
                "color": MARK_COLOR.get(m, C["text"]),
                "fontWeight": "600", "marginRight": "6px",
            }) for m in KNOWN_MARKS],
        ]),
    ])

    # ---------- callback ----------
    @app.callback(
        [Output("f-edge", "figure"), Output("f-imb", "figure"),
         Output("f-heat", "figure"), Output("f-fwd", "figure"),
         Output("f-ivrv", "figure"), Output("f-resid", "figure"),
         Output("f-net", "figure"),
         Output("f-spread-dist", "figure"), Output("f-mark-hsp", "figure"),
         Output("f-spread-act", "figure"),
         Output("f-smile-snaps", "figure"), Output("f-strike-iv", "figure"),
         Output("f-iv-skew", "figure"),
         Output("f-vol-strike", "figure"), Output("f-mark-clip", "figure"),
         Output("f-greeks-eod", "figure"), Output("f-rv-comp", "figure")],
        [Input("f-product", "value"), Input("f-day", "value"),
         Input("f-buckets", "value")],
    )
    def update(product, day, n_buckets):
        empty = go.Figure().update_layout(**BASE)
        if not product or not day:
            return [empty] * 17

        # Slice
        p_day_all = prices[prices["day_label"] == day]
        t_day_all = trades[trades["day_label"] == day]
        p_pd = p_day_all[p_day_all["product"] == product]
        t_pd = t_day_all[t_day_all["symbol"] == product]

        mid_lookup = (p_pd.dropna(subset=["mid_price"])
                          .set_index("timestamp")["mid_price"])
        spot_mid = (p_day_all[p_day_all["product"] == UNDERLYING]
                    .dropna(subset=["mid_price"])
                    .set_index("timestamp")["mid_price"])

        # 1-7: existing signal plots
        f_edge = fig_mark_vwap_edge(t_pd, mid_lookup, product)
        f_imb = fig_dealer_imbalance(t_day_all)
        f_heat = fig_mark_activity_heatmap(t_day_all, n_buckets=n_buckets or 50)
        f_fwd = fig_mark_forward_returns(t_day_all, spot_mid)
        f_ivrv = fig_iv_vs_rv(p_day_all, day_int=int(p_day_all["day"].iloc[0])
                              if not p_day_all.empty else 0)
        f_resid = fig_voucher_residual(p_day_all)
        f_net = fig_per_mark_net_qty(t_pd, product)

        # 8-10: spread cluster
        f_spread_dist = fig_spread_dist_by_product(p_day_all)
        f_mark_hsp = fig_per_mark_half_spread(t_pd, mid_lookup, product)
        f_spread_act = fig_spread_vs_activity(p_day_all, t_day_all, product)

        # 11-13: IV cluster
        f_smile_snaps = fig_iv_smile_snapshots(p_day_all)
        f_strike_iv = fig_per_strike_iv_ts(p_day_all)
        f_iv_skew = fig_iv_skew_curvature(p_day_all)

        # 14-16: other
        f_vol_strike = fig_volume_per_strike(t_day_all)
        f_mark_clip = fig_per_mark_clip_dist(t_day_all)
        f_greeks_eod = fig_voucher_greeks_eod(p_day_all)
        f_rv_comp = fig_rv_comparison(p_day_all)

        return (f_edge, f_imb, f_heat, f_fwd, f_ivrv, f_resid, f_net,
                f_spread_dist, f_mark_hsp, f_spread_act,
                f_smile_snaps, f_strike_iv, f_iv_skew,
                f_vol_strike, f_mark_clip, f_greeks_eod, f_rv_comp)

    return app


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Round 4 Featured Signal Visualizer")
    parser.add_argument("--data", "-d", default=None,
                        help="Data folder (default: auto-detect Round 4)")
    parser.add_argument("--port", "-p", type=int, default=8060)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    data_dir = args.data or find_data_folder()
    if not data_dir:
        print("No data folder found. Pass --data DIR.")
        sys.exit(1)

    print(f"Loading: {data_dir}")
    prices, trades = load_data(data_dir)
    print(f"  Prices: {len(prices):,} rows | Trades: {len(trades):,} rows")
    print(f"  Products: {len(prices['product'].unique())} | Days: {sorted(prices['day_label'].unique())}")

    app = create_app(prices, trades)

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    print(f"\n  http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
