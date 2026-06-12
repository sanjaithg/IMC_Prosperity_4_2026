"""
backtest_log_visualizer.py
==========================
Interactive backtest visualizer (Dash).

Primary mode (no files generated):
  Run a backtest in-memory from a submission + dataset and visualize directly.

Fallback mode:
  Load existing backtester CSV outputs with --pnl and --fills.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import webbrowser
from threading import Timer
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, ctx, no_update
from dash.dependencies import Input, Output, State, ALL

# Reuse local backtester core to avoid writing csv files.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.append(ROOT)
from backtester import (  # noqa: E402
    discover_available_days,
    discover_data_dir,
    load_day,
    load_trader,
    run_backtest_single_day,
)


def run_backtest_frames(
    submission: str,
    data_dir: str,
    days: Optional[List[int]],
    no_passive: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub_path = os.path.abspath(submission)
    data_path = os.path.abspath(data_dir)

    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission not found: {sub_path}")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data dir not found: {data_path}")

    use_days = days if days else discover_available_days(data_path)
    if not use_days:
        raise FileNotFoundError(f"No prices_round_*_day_*.csv found in {data_path}")

    all_pnl: List[pd.DataFrame] = []
    all_fills: List[pd.DataFrame] = []

    for day in use_days:
        trader = load_trader(sub_path)
        prices, trades, round_id = load_day(data_path, day)
        pnl_df, fills_df, _, _ = run_backtest_single_day(
            trader,
            prices,
            trades,
            day=day,
            round_id=round_id,
            passive_fills=not no_passive,
            verbose=False,
        )
        pnl_df = attach_market_columns(pnl_df, prices)
        all_pnl.append(pnl_df)
        all_fills.append(fills_df)

    pnl = pd.concat(all_pnl, ignore_index=True) if all_pnl else pd.DataFrame()
    fills = pd.concat(all_fills, ignore_index=True) if all_fills else pd.DataFrame()
    return pnl, fills


def attach_market_columns(pnl_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add mid/spread columns per product so the backtest dashboard can show execution context."""
    if pnl_df.empty or prices.empty:
        return pnl_df
    required = {"timestamp", "product", "mid_price", "bid_price_1", "ask_price_1"}
    if not required.issubset(prices.columns):
        return pnl_df

    market = prices[["timestamp", "product", "mid_price", "bid_price_1", "ask_price_1"]].copy()
    market["spread"] = market["ask_price_1"] - market["bid_price_1"]
    mid = market.pivot(index="timestamp", columns="product", values="mid_price").add_prefix("mid_")
    spread = market.pivot(index="timestamp", columns="product", values="spread").add_prefix("spread_")
    bid = market.pivot(index="timestamp", columns="product", values="bid_price_1").add_prefix("bid_")
    ask = market.pivot(index="timestamp", columns="product", values="ask_price_1").add_prefix("ask_")
    wide = pd.concat([mid, spread, bid, ask], axis=1).reset_index()
    return pnl_df.merge(wide, on="timestamp", how="left")


def compute_stats(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {"final": 0.0, "peak": 0.0, "trough": 0.0, "max_dd": 0.0, "sharpe": float("nan")}

    peak = series.cummax()
    ret = series.diff().dropna()
    sharpe = (ret.mean() / ret.std()) * math.sqrt(len(ret)) if ret.std() > 0 else float("nan")
    return {
        "final": float(series.iloc[-1]),
        "peak": float(series.max()),
        "trough": float(series.min()),
        "max_dd": float((peak - series).max()),
        "sharpe": float(sharpe),
    }


def _signed_fill_qty(side: str, qty: float) -> float:
    if isinstance(side, str) and side.startswith("BUY"):
        return abs(float(qty))
    return -abs(float(qty))


def _position_limit(product: str) -> int:
    return 10


def _sparkline_figure(x: pd.Series, y: pd.Series, c: Dict[str, str]) -> go.Figure:
    fig = go.Figure()
    if len(x) and len(y):
        line_color = c["good"] if float(y.iloc[-1]) >= 0 else c["bad"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=line_color, width=1.2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def make_theme() -> Dict[str, str]:
    return {
        "bg": "#090d14",
        "panel": "#101722",
        "ink": "#e5edf7",
        "muted": "#8ea0b8",
        "border": "#243246",
        "grid": "#223047",
        "accent": "#38bdf8",
        "accent2": "#f59e0b",
        "good": "#22c55e",
        "bad": "#ef4444",
    }


def apply_figure_theme(fig: go.Figure, c: Dict[str, str], title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=c["panel"],
        plot_bgcolor=c["panel"],
        font=dict(color=c["ink"], family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=10),
        margin=dict(l=44, r=24, t=50, b=72),
        title=dict(text=title, font=dict(size=13), x=0.015, y=0.98),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            x=0,
            y=-0.22,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9),
        ),
    )
    fig.update_xaxes(gridcolor=c["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=c["grid"], zeroline=False)
    return fig


def graph_panel(graph_id: str, c: Dict[str, str], height: str = "360px"):
    return html.Div(
        style={
            "backgroundColor": c["panel"],
            "border": f"1px solid {c['border']}",
            "borderRadius": "8px",
            "padding": "8px",
            "marginBottom": "12px",
        },
        children=[dcc.Graph(id=graph_id, config={"displaylogo": False}, style={"height": height, "width": "100%"})],
    )


def create_app(pnl: pd.DataFrame, fills: pd.DataFrame, title: str) -> Dash:
    c = make_theme()
    app = Dash(__name__, title="Round 5 Backtest Explorer", suppress_callback_exceptions=True)

    if pnl.empty:
        raise ValueError("Empty pnl dataframe")

    pnl = pnl.sort_values(["day", "timestamp"]).reset_index(drop=True)
    fills = fills.sort_values(["day", "timestamp"]).reset_index(drop=True) if not fills.empty else fills

    pos_cols = [x for x in pnl.columns if x.startswith("pos_")]
    assets = [x.replace("pos_", "") for x in pos_cols]
    assets = sorted(assets)
    days = sorted(pnl["day"].unique())

    default_asset = assets[0]

    app.layout = html.Div(
        style={"backgroundColor": c["bg"], "minHeight": "100vh", "padding": "22px", "color": c["ink"]},
        children=[
            html.Div(
                style={
                    "backgroundColor": c["panel"],
                    "border": f"1px solid {c['border']}",
                    "borderRadius": "14px",
                    "padding": "16px 18px",
                    "marginBottom": "14px",
                },
                children=[
                    html.Div("Backtest Explorer", style={"fontSize": "24px", "fontWeight": 800, "color": c["ink"]}),
                    html.Div(title, style={"fontSize": "12px", "color": c["muted"], "marginTop": "4px"}),
                    html.Div(
                        style={"display": "flex", "gap": "12px", "marginTop": "14px", "alignItems": "end", "flexWrap": "wrap"},
                        children=[
                            html.Div(
                                style={"minWidth": "190px"},
                                children=[
                                    html.Div("Day", style={"fontSize": "10px", "color": c["muted"], "marginBottom": "6px"}),
                                    dcc.Dropdown(
                                        id="day",
                                        options=[{"label": f"Day {int(d)}", "value": int(d)} for d in days],
                                        value=int(days[0]),
                                        clearable=False,
                                        className="dark-dropdown",
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"minWidth": "520px"},
                                children=[
                                    html.Div("Asset", style={"fontSize": "10px", "color": c["muted"], "marginBottom": "6px"}),
                                    dcc.Dropdown(
                                        id="asset",
                                        options=[{"label": a, "value": a} for a in assets],
                                        value=default_asset,
                                        clearable=False,
                                        className="dark-dropdown",
                                        style={"fontSize": "13px"},
                                    ),
                                ],
                            ),
                            html.Div(
                                style={"minWidth": "520px"},
                                children=[
                                    html.Div("Compare tickers", style={"fontSize": "10px", "color": c["muted"], "marginBottom": "6px"}),
                                    dcc.Dropdown(
                                        id="compare-assets",
                                        options=[{"label": a, "value": a} for a in assets],
                                        value=[default_asset],
                                        clearable=True,
                                        multi=True,
                                        className="dark-dropdown",
                                        style={"fontSize": "13px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "backgroundColor": c["panel"],
                    "border": f"1px solid {c['border']}",
                    "borderRadius": "12px",
                    "padding": "10px 12px",
                    "marginBottom": "14px",
                },
                children=[
                    html.Div("Products PnL Table", style={"fontSize": "14px", "fontWeight": 700, "marginBottom": "8px"}),
                    html.Div(
                        f"Click ticker to load product analysis. Colored PnL and mini sparkline shown for each product.",
                        style={"fontSize": "11px", "color": c["muted"], "marginBottom": "8px"},
                    ),
                    html.Div(id="pnl-table"),
                ],
            ),
            html.Div(
                id="stat-row",
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))", "gap": "10px", "marginBottom": "14px"},
            ),
            graph_panel("price-chart", c, "390px"),
            graph_panel("pnl-chart", c, "360px"),
            graph_panel("drawdown-chart", c, "300px"),
            graph_panel("pos-chart", c, "340px"),
            graph_panel("flow-chart", c, "320px"),
            graph_panel("product-chart", c, "360px"),
        ],
    )

    app.index_string = """<!DOCTYPE html><html><head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
    *{box-sizing:border-box}body{margin:0;background:#090d14;color:#e5edf7}
    ::-webkit-scrollbar{width:8px;height:8px}
    ::-webkit-scrollbar-track{background:#090d14}
    ::-webkit-scrollbar-thumb{background:#243246;border-radius:4px}
    .dark-dropdown .Select-control,
    .dark-dropdown .Select-menu-outer,
    .dark-dropdown .Select-menu,
    .dark-dropdown .Select-option,
    .dark-dropdown .Select-value,
    .dark-dropdown .Select-placeholder,
    .dark-dropdown .Select-input,
    .dark-dropdown .Select-input input{
      background:#101722!important;
      color:#e5edf7!important;
    }
    .dark-dropdown .Select-control{border:1px solid #243246!important;box-shadow:none!important}
    .dark-dropdown .Select-menu-outer{border:1px solid #243246!important;box-shadow:0 14px 32px rgba(0,0,0,.38)!important;z-index:9999!important}
    .dark-dropdown .Select-option{color:#e5edf7!important}
    .dark-dropdown .Select-option.is-focused,
    .dark-dropdown .Select-option.is-selected{background:#1e293b!important;color:#fff!important}
    .dark-dropdown .Select-value-label,
    .dark-dropdown .Select--single>.Select-control .Select-value{color:#e5edf7!important}
    .dark-dropdown .Select-placeholder{color:#8ea0b8!important}
    .dark-dropdown .Select--multi .Select-value{background:#1e293b!important;border-color:#334155!important;color:#e5edf7!important}
    .dark-dropdown .Select--multi .Select-value-label{color:#e5edf7!important}
    .dark-dropdown .Select-clear-zone{color:#8ea0b8!important}
    .dark-dropdown .Select-arrow-zone .Select-arrow{border-top-color:#8ea0b8!important}
    .modebar-btn path{fill:#8ea0b8!important}
    .modebar-btn:hover path{fill:#38bdf8!important}
    .modebar-btn.active path{fill:#38bdf8!important}
    </style></head><body>
    {%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body></html>"""

    @app.callback(
        [
            Output("pnl-table", "children"),
            Output("asset", "options"),
            Output("compare-assets", "options"),
            Output("stat-row", "children"),
            Output("price-chart", "figure"),
            Output("pnl-chart", "figure"),
            Output("drawdown-chart", "figure"),
            Output("pos-chart", "figure"),
            Output("flow-chart", "figure"),
            Output("product-chart", "figure"),
        ],
        [Input("day", "value"), Input("asset", "value"), Input("compare-assets", "value")],
    )
    def update(day: int, asset: str, compare_assets: Optional[List[str]]):
        day_df = pnl[pnl["day"] == day].copy().sort_values("timestamp")
        if day_df.empty:
            empty = apply_figure_theme(go.Figure(), c, "No data")
            return html.Div("No data"), [], [], [], empty, empty, empty, empty, empty, empty

        pos_col = f"pos_{asset}"
        ppnl_col = f"ppnl_{asset}"
        if pos_col not in day_df.columns:
            day_df[pos_col] = 0
        if ppnl_col not in day_df.columns:
            day_df[ppnl_col] = 0.0
        pos_limit = _position_limit(asset)
        inv_used = (day_df[pos_col].abs() / pos_limit) * 100.0

        total_stats = compute_stats(day_df["pnl"])
        asset_stats = compute_stats(day_df[ppnl_col])

        card = lambda label, val, color: html.Div(
            style={
                "backgroundColor": c["panel"],
                "border": f"1px solid {c['border']}",
                "borderRadius": "12px",
                "padding": "10px 12px",
            },
            children=[
                html.Div(label, style={"fontSize": "10px", "color": c["muted"]}),
                html.Div(val, style={"fontSize": "18px", "fontWeight": 800, "color": color, "marginTop": "4px"}),
            ],
        )

        fills_a = fills[(fills["day"] == day) & (fills["product"] == asset)].copy() if not fills.empty else pd.DataFrame()
        n_fills = int(len(fills_a))
        if not fills_a.empty:
            fills_a["signed_qty"] = [_signed_fill_qty(s, q) for s, q in zip(fills_a["side"], fills_a["quantity"])]
            fills_a["cum_signed_qty"] = fills_a["signed_qty"].cumsum()
        final_pos = float(day_df[pos_col].iloc[-1])
        n_passive = int(fills_a["side"].astype(str).str.contains("PASSIVE").sum()) if not fills_a.empty else 0
        n_aggressive = n_fills - n_passive
        final_inv = float(inv_used.iloc[-1]) if not inv_used.empty else 0.0
        max_inv = float(inv_used.max()) if not inv_used.empty else 0.0

        # Top persistent table with per-product pnl + sparkline.
        ppnl_cols_all = [col for col in day_df.columns if col.startswith("ppnl_")]
        products_with_pnl = [col.replace("ppnl_", "") for col in ppnl_cols_all]
        product_final = {
            p: float(day_df[f"ppnl_{p}"].iloc[-1]) if f"ppnl_{p}" in day_df.columns else 0.0
            for p in products_with_pnl
        }
        sorted_products = sorted(products_with_pnl, key=lambda p: product_final.get(p, 0.0), reverse=True)

        table_header = html.Thead(
            html.Tr(
                [
                    html.Th("Ticker", style={"textAlign": "left", "padding": "8px", "fontSize": "12px"}),
                    html.Th("PnL", style={"textAlign": "right", "padding": "8px", "fontSize": "12px"}),
                    html.Th("Spark", style={"textAlign": "left", "padding": "8px", "fontSize": "12px"}),
                ]
            )
        )
        table_rows = []
        for p in sorted_products:
            series = day_df[f"ppnl_{p}"] if f"ppnl_{p}" in day_df.columns else pd.Series(np.zeros(len(day_df)))
            p_final = float(series.iloc[-1]) if len(series) else 0.0
            pnl_color = c["good"] if p_final >= 0 else c["bad"]
            spark = _sparkline_figure(day_df["timestamp"], series, c)
            table_rows.append(
                html.Tr(
                    [
                        html.Td(
                            html.Button(
                                p,
                                id={"type": "asset-btn", "index": p},
                                n_clicks=0,
                                style={
                                    "background": "transparent",
                                    "color": c["ink"],
                                    "border": f"1px solid {c['border']}",
                                    "borderRadius": "6px",
                                    "padding": "6px 10px",
                                    "fontSize": "13px",
                                    "textAlign": "left",
                                    "cursor": "pointer",
                                    "minWidth": "360px",
                                },
                            ),
                            style={"padding": "6px 8px"},
                        ),
                        html.Td(
                            f"{p_final:+,.1f}",
                            style={"padding": "6px 8px", "textAlign": "right", "fontWeight": 700, "color": pnl_color, "fontSize": "13px"},
                        ),
                        html.Td(
                            dcc.Graph(figure=spark, config={"displayModeBar": False}, style={"height": "32px", "width": "140px"}),
                            style={"padding": "4px 8px"},
                        ),
                    ]
                )
            )
        pnl_table = html.Table(
            [table_header, html.Tbody(table_rows)],
            style={"width": "100%", "borderCollapse": "collapse", "tableLayout": "auto"},
        )

        asset_options = [{"label": f"{p} | {product_final.get(p, 0.0):+,.1f}", "value": p} for p in sorted_products]
        compare_options = asset_options

        stat_cards = [
            card("Day Total PnL", f"{total_stats['final']:+,.1f}", c["good"] if total_stats["final"] >= 0 else c["bad"]),
            card(f"{asset} PnL", f"{asset_stats['final']:+,.1f}", c["good"] if asset_stats["final"] >= 0 else c["bad"]),
            card("Max Drawdown", f"{asset_stats['max_dd']:,.1f}", c["bad"]),
            card("Sharpe (asset)", f"{asset_stats['sharpe']:.3f}", c["ink"]),
            card("Position", f"{final_pos:+.0f}", c["good"] if final_pos > 0 else (c["bad"] if final_pos < 0 else c["muted"])),
            card("Inventory Used", f"{final_inv:.1f}% / max {max_inv:.1f}%", c["bad"] if max_inv >= 80 else c["ink"]),
            card("Fills", f"{n_fills}", c["ink"]),
            card("Passive / Agg", f"{n_passive} / {n_aggressive}", c["ink"]),
        ]

        # Price chart: selected asset market context plus executions when market columns are available.
        fig_price = go.Figure()
        mid_col = f"mid_{asset}"
        bid_col = f"bid_{asset}"
        ask_col = f"ask_{asset}"
        spread_col = f"spread_{asset}"
        if mid_col in day_df.columns:
            if bid_col in day_df.columns and ask_col in day_df.columns:
                fig_price.add_trace(go.Scatter(
                    x=day_df["timestamp"], y=day_df[ask_col], mode="lines",
                    line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig_price.add_trace(go.Scatter(
                    x=day_df["timestamp"], y=day_df[bid_col], mode="lines",
                    line=dict(width=0), fill="tonexty", fillcolor="rgba(148,163,184,0.08)",
                    name="bid-ask band", hoverinfo="skip",
                ))
                fig_price.add_trace(go.Scatter(
                    x=day_df["timestamp"], y=day_df[bid_col], mode="lines",
                    name="best bid", line=dict(color=c["good"], width=1, dash="dash"), opacity=0.7,
                ))
                fig_price.add_trace(go.Scatter(
                    x=day_df["timestamp"], y=day_df[ask_col], mode="lines",
                    name="best ask", line=dict(color=c["bad"], width=1, dash="dash"), opacity=0.7,
                ))
            fig_price.add_trace(go.Scatter(
                x=day_df["timestamp"], y=day_df[mid_col], mode="lines",
                name="mid price", line=dict(color=c["accent"], width=2.2),
                customdata=np.stack([inv_used], axis=-1),
                hovertemplate="timestamp=%{x}<br>mid=%{y:.1f}<br>inv used=%{customdata[0]:.1f}%<extra></extra>",
            ))
            if spread_col in day_df.columns:
                fig_price.add_trace(go.Scatter(
                    x=day_df["timestamp"], y=day_df[spread_col], mode="lines",
                    name="spread", line=dict(color=c["accent2"], width=1.5), yaxis="y2",
                    hovertemplate="timestamp=%{x}<br>spread=%{y:.2f}<extra></extra>",
                ))
            if not fills_a.empty:
                buy_mask = fills_a["side"].astype(str).str.startswith("BUY")
                colors = np.where(buy_mask, c["good"], c["bad"])
                symbols = np.where(buy_mask, "triangle-up", "triangle-down")
                fig_price.add_trace(go.Scatter(
                    x=fills_a["timestamp"], y=fills_a["price"], mode="markers",
                    name="fills",
                    marker=dict(symbol=symbols, size=10, color=colors, line=dict(color="#ffffff", width=1)),
                    customdata=np.stack([fills_a["side"], fills_a["quantity"]], axis=-1),
                    hovertemplate="timestamp=%{x}<br>%{customdata[0]} %{customdata[1]} @ %{y:.1f}<extra></extra>",
                ))
        else:
            fig_price.add_annotation(
                text="Price context is available for in-memory runs. CSV mode needs mid_/bid_/ask_ columns.",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color=c["muted"]),
            )
        apply_figure_theme(fig_price, c, f"Day {day} {asset} price and executions")
        fig_price.update_xaxes(title="Timestamp")
        fig_price.update_yaxes(title="Price")
        if spread_col in day_df.columns:
            fig_price.update_layout(yaxis2=dict(title="Spread", overlaying="y", side="right", showgrid=False, zeroline=False))

        # PnL chart: day total + selected asset pnl.
        fig_top = go.Figure()
        fig_top.add_trace(go.Scatter(
            x=day_df["timestamp"], y=day_df["pnl"], mode="lines", name="Day total PnL",
            line=dict(color="#94a3b8", width=2),
            customdata=inv_used,
            hovertemplate="timestamp=%{x}<br>Total PnL=%{y:+,.1f}<br>Inv used=%{customdata:.1f}%<extra></extra>",
        ))
        fig_top.add_trace(go.Scatter(
            x=day_df["timestamp"], y=day_df[ppnl_col], mode="lines", name=f"{asset} PnL",
            line=dict(color=c["accent"], width=2.5),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.10)",
            customdata=inv_used,
            hovertemplate=f"timestamp=%{{x}}<br>{asset} PnL=%{{y:+,.1f}}<br>Inv used=%{{customdata:.1f}}%<extra></extra>",
        ))
        apply_figure_theme(fig_top, c, f"Day {day} PnL curves")
        fig_top.update_xaxes(title="Timestamp")
        fig_top.update_yaxes(title="PnL")

        # Drawdown chart.
        fig_dd = go.Figure()
        total_dd = day_df["pnl"] - day_df["pnl"].cummax()
        asset_dd = day_df[ppnl_col] - day_df[ppnl_col].cummax()
        fig_dd.add_trace(go.Scatter(
            x=day_df["timestamp"], y=total_dd, mode="lines",
            name="total drawdown", line=dict(color="#94a3b8", width=1.8),
            fill="tozeroy", fillcolor="rgba(148,163,184,0.10)",
            hovertemplate="timestamp=%{x}<br>total drawdown=%{y:+,.1f}<extra></extra>",
        ))
        fig_dd.add_trace(go.Scatter(
            x=day_df["timestamp"], y=asset_dd, mode="lines",
            name=f"{asset} drawdown", line=dict(color=c["bad"], width=2),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.13)",
            hovertemplate=f"timestamp=%{{x}}<br>{asset} drawdown=%{{y:+,.1f}}<extra></extra>",
        ))
        fig_dd.add_hline(y=0, line=dict(color=c["grid"], width=1))
        apply_figure_theme(fig_dd, c, f"Day {day} drawdown")
        fig_dd.update_xaxes(title="Timestamp")
        fig_dd.update_yaxes(title="Drawdown")

        # Position chart: full state, long/short fills, and normalized inventory usage.
        fig_pos = go.Figure()
        pos_v = day_df[pos_col].astype(float)
        pos_long = pos_v.where(pos_v > 0, np.nan)
        pos_short = pos_v.where(pos_v < 0, np.nan)
        pos_flat = pos_v.where(pos_v == 0, np.nan)
        fig_pos.add_trace(go.Scatter(
            x=day_df["timestamp"], y=pos_v.clip(lower=0), mode="none",
            fill="tozeroy", fillcolor="rgba(34,197,94,0.14)", showlegend=False, hoverinfo="skip",
        ))
        fig_pos.add_trace(go.Scatter(
            x=day_df["timestamp"], y=pos_v.clip(upper=0), mode="none",
            fill="tozeroy", fillcolor="rgba(239,68,68,0.14)", showlegend=False, hoverinfo="skip",
        ))
        fig_pos.add_trace(go.Scatter(
            x=day_df["timestamp"], y=pos_long, mode="lines",
            line=dict(color=c["good"], width=2.4), name="long",
            customdata=inv_used,
            hovertemplate="timestamp=%{x}<br>long pos=%{y:.0f}<br>inv used=%{customdata:.1f}%<extra></extra>",
        ))
        fig_pos.add_trace(go.Scatter(
            x=day_df["timestamp"], y=pos_short, mode="lines",
            line=dict(color=c["bad"], width=2.4), name="short",
            customdata=inv_used,
            hovertemplate="timestamp=%{x}<br>short pos=%{y:.0f}<br>inv used=%{customdata:.1f}%<extra></extra>",
        ))
        fig_pos.add_trace(go.Scatter(
            x=day_df["timestamp"], y=pos_flat, mode="lines",
            line=dict(color=c["muted"], width=1.5, dash="dot"), name="flat",
            customdata=inv_used,
            hovertemplate="timestamp=%{x}<br>flat<br>inv used=%{customdata:.1f}%<extra></extra>",
        ))
        fig_pos.add_trace(go.Scatter(
            x=day_df["timestamp"], y=inv_used, mode="lines",
            line=dict(color="#7c3aed", width=1.6, dash="dot"), name="Inventory Used %",
            yaxis="y2",
            hovertemplate="timestamp=%{x}<br>inv used=%{y:.1f}%<extra></extra>",
        ))
        if not fills_a.empty:
            buy_mask = fills_a["signed_qty"] > 0
            for label, mask, color, symbol in [
                ("buy fills", buy_mask, c["good"], "triangle-up"),
                ("sell fills", ~buy_mask, c["bad"], "triangle-down"),
            ]:
                ff = fills_a[mask].copy()
                if ff.empty:
                    continue
                fill_pos = pd.merge_asof(
                    ff.sort_values("timestamp"),
                    day_df[["timestamp", pos_col]].sort_values("timestamp"),
                    on="timestamp",
                    direction="backward",
                )
                fig_pos.add_trace(go.Scatter(
                    x=fill_pos["timestamp"], y=fill_pos[pos_col], mode="markers",
                    name=label,
                    marker=dict(symbol=symbol, size=9, color=color, line=dict(color="#ffffff", width=1)),
                    customdata=np.stack([fill_pos["side"], fill_pos["quantity"]], axis=-1),
                    hovertemplate="timestamp=%{x}<br>%{customdata[0]} %{customdata[1]}<br>pos=%{y:+.0f}<extra></extra>",
                ))
        fig_pos.add_hline(y=0, line=dict(color=c["grid"], width=1, dash="dot"))
        apply_figure_theme(fig_pos, c, f"{asset} position and inventory usage")
        fig_pos.update_xaxes(title="Timestamp")
        fig_pos.update_yaxes(title="Position")
        fig_pos.update_layout(
            yaxis2=dict(title="Inventory Used (%)", overlaying="y", side="right", range=[0, 105], showgrid=False, zeroline=False),
        )

        # Fill flow chart.
        fig_flow = go.Figure()
        if not fills_a.empty:
            colors = [c["good"] if q > 0 else c["bad"] for q in fills_a["signed_qty"]]
            fig_flow.add_trace(go.Bar(
                x=fills_a["timestamp"], y=fills_a["signed_qty"], marker_color=colors,
                name="signed fills", opacity=0.62,
                customdata=np.stack([fills_a["side"], fills_a["price"], fills_a["quantity"]], axis=-1),
                hovertemplate="timestamp=%{x}<br>%{customdata[0]} %{customdata[2]} @ %{customdata[1]:.1f}<extra></extra>",
            ))
            fig_flow.add_trace(go.Scatter(
                x=fills_a["timestamp"], y=fills_a["cum_signed_qty"], mode="lines",
                line=dict(color="#a78bfa", width=2), name="cumulative signed qty",
                hovertemplate="timestamp=%{x}<br>cumulative signed qty=%{y:+.0f}<extra></extra>",
            ))
            fig_flow.add_trace(go.Scatter(
                x=fills_a["timestamp"], y=fills_a["price"], mode="lines+markers",
                line=dict(color=c["accent"], width=1.4, dash="dot"),
                marker=dict(size=5, color=colors),
                name="fill price", yaxis="y2",
                hovertemplate="timestamp=%{x}<br>fill price=%{y:.1f}<extra></extra>",
            ))
        else:
            fig_flow.add_annotation(
                text="No fills for selected product/day",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=12, color=c["muted"]),
            )
        apply_figure_theme(fig_flow, c, f"{asset} fill flow")
        fig_flow.update_xaxes(title="Timestamp")
        fig_flow.update_yaxes(title="Signed quantity")
        fig_flow.update_layout(yaxis2=dict(title="Fill price", overlaying="y", side="right", showgrid=False, zeroline=False))

        # Per-product / comparison chart.
        fig_product = go.Figure()
        compare_list = [a for a in (compare_assets or []) if f"ppnl_{a}" in day_df.columns]
        if not compare_list:
            compare_list = [asset]
        for a in compare_list:
            col = f"ppnl_{a}"
            a_final = float(day_df[col].iloc[-1]) if len(day_df[col]) else 0.0
            color = c["accent"] if a == asset else (c["good"] if a_final >= 0 else c["bad"])
            fig_product.add_trace(go.Scatter(
                x=day_df["timestamp"], y=day_df[col], mode="lines", name=f"{a} ({a_final:+,.1f})",
                line=dict(color=color, width=2.5 if a == asset else 1.6),
                hovertemplate=f"timestamp=%{{x}}<br>{a} PnL=%{{y:+,.1f}}<extra></extra>",
            ))
        fig_product.add_hline(y=0, line=dict(color=c["grid"], width=1))
        apply_figure_theme(fig_product, c, f"Day {day} ticker compare PnL")
        fig_product.update_xaxes(title="Timestamp")
        fig_product.update_yaxes(title="PnL")

        return pnl_table, asset_options, compare_options, stat_cards, fig_price, fig_top, fig_dd, fig_pos, fig_flow, fig_product

    @app.callback(
        Output("asset", "value"),
        [Input({"type": "asset-btn", "index": ALL}, "n_clicks")],
        [State({"type": "asset-btn", "index": ALL}, "id"), State("asset", "value")],
        prevent_initial_call=True,
    )
    def pick_asset_from_table(n_clicks: List[int], ids: List[Dict[str, str]], current_asset: str):
        if not n_clicks or not ids:
            return no_update
        if ctx.triggered_id and isinstance(ctx.triggered_id, dict):
            return ctx.triggered_id.get("index", current_asset)
        return no_update

    return app


def _limit_for_asset(asset: str) -> int:
    return 10


def print_backtest_diagnostics(pnl: pd.DataFrame, fills: pd.DataFrame) -> None:
    """Quick risk/execution summary in terminal."""
    if pnl.empty:
        print("  [diag] empty pnl frame")
        return
    assets = [c.replace("ppnl_", "") for c in pnl.columns if c.startswith("ppnl_")]
    if not assets:
        print("  [diag] no per-asset ppnl_* columns found")
        return

    print("  [diag] per-asset PnL / DD / inventory")
    for asset in sorted(assets):
        series = pnl[f"ppnl_{asset}"].astype(float)
        final = float(series.iloc[-1]) if not series.empty else 0.0
        dd = float((series.cummax() - series).max()) if not series.empty else 0.0
        f = fills[fills["product"] == asset].sort_values("timestamp").copy() if not fills.empty else pd.DataFrame()
        if not f.empty:
            side = f["side"].astype(str)
            signed = f["quantity"].where(side.str.startswith("BUY"), -f["quantity"])
            pos = signed.cumsum()
            max_inv = float((pos.abs().max() / _limit_for_asset(asset)) * 100.0)
            n = len(f)
        else:
            max_inv = 0.0
            n = 0
        print(f"    {asset:<24s} pnl={final:>10.1f} dd={dd:>10.1f} fills={n:>5d} max_inv={max_inv:>5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive backtest visualizer")

    parser.add_argument("--pnl", default=None, help="Path to pnl csv (optional)")
    parser.add_argument("--fills", default=None, help="Path to fills csv (optional)")

    parser.add_argument("--submission", default=None, help="Submission path for in-memory backtest")
    parser.add_argument("--data-dir", default=None, help="Dataset directory for in-memory backtest")
    parser.add_argument("--days", nargs="+", type=int, default=None)
    parser.add_argument("--no-passive", action="store_true")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8071)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--diag", action="store_true",
                        help="Print terminal diagnostics for PnL/drawdown/inventory by asset")
    args = parser.parse_args()

    title = ""
    if args.pnl and os.path.exists(args.pnl):
        pnl = pd.read_csv(args.pnl)
        fills = pd.read_csv(args.fills) if args.fills and os.path.exists(args.fills) else pd.DataFrame()
        title = f"CSV mode  |  {os.path.basename(args.pnl)}"
    else:
        if not args.submission:
            raise SystemExit("Provide --submission (or --pnl).")
        data_dir = args.data_dir or discover_data_dir()
        if not data_dir:
            raise SystemExit("Could not auto-detect data dir. Pass --data-dir.")
        pnl, fills = run_backtest_frames(args.submission, data_dir, args.days, args.no_passive)
        title = f"In-memory backtest  |  {os.path.basename(args.submission)}"

    if args.diag:
        print_backtest_diagnostics(pnl, fills)

    app = create_app(pnl, fills, title)

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    print(f"\n  http://{args.host}:{args.port}\n")
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
