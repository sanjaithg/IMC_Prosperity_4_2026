"""
Prosperity Round 5 algorithm log visualizer
=============================================
Usage:
    python visualizer.py Logs/Tutorial/trader_bollinger/74814.log
    python visualizer.py Logs/Tutorial/trader_sma/59742.log --port 8051
"""

import os
import sys
import json
import argparse
import webbrowser
from io import StringIO
from threading import Timer

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State, ALL
from dash import ctx, no_update


# ═══════════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════════


def resolve_log_path(path):
    """Accept either a .log file or a submission logs folder."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None

    candidates = [os.path.join(path, name) for name in sorted(os.listdir(path)) if name.endswith(".log")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    for root, _dirs, files in os.walk(path):
        nested = [os.path.join(root, name) for name in files if name.endswith(".log")]
        if nested:
            nested.sort(key=os.path.getmtime, reverse=True)
            return nested[0]
    return None

def parse_log(log_path):
    with open(log_path) as f:
        data = json.load(f)

    activities = pd.read_csv(StringIO(data["activitiesLog"]), delimiter=";")
    activities["time_s"] = activities["timestamp"] / 100
    activities["day_label"] = "Day " + activities["day"].astype(str)
    # Replace mid_price with NaN when either bid or ask is missing
    empty_book = activities["bid_price_1"].isna() | activities["ask_price_1"].isna()
    activities.loc[empty_book, "mid_price"] = np.nan
    activities["spread"] = activities["ask_price_1"] - activities["bid_price_1"]

    trade_history = pd.DataFrame(data.get("tradeHistory", []))
    if not trade_history.empty:
        trade_history["time_s"] = trade_history["timestamp"] / 100
        if "day" in trade_history.columns:
            trade_history["day_label"] = "Day " + trade_history["day"].astype(str)
        else:
            first_day = activities["day"].iloc[0] if not activities.empty else 0
            trade_history["day_label"] = f"Day {int(first_day)}"
        trade_history["side"] = trade_history.apply(
            lambda r: "BUY" if r.get("buyer") == "SUBMISSION"
            else ("SELL" if r.get("seller") == "SUBMISSION" else "MARKET"), axis=1)
        trade_history["signed_qty"] = trade_history.apply(
            lambda r: r["quantity"] if r["side"] == "BUY"
            else (-r["quantity"] if r["side"] == "SELL" else 0), axis=1)

    json_path = log_path.replace(".log", ".json")
    companion = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            companion = json.load(f)

    graph_log = pd.DataFrame()
    if companion.get("graphLog"):
        graph_log = pd.read_csv(StringIO(companion["graphLog"]), delimiter=";")
        graph_log["time_s"] = graph_log["timestamp"] / 100

    return {
        "activities": activities,
        "trades": trade_history,
        "graph_log": graph_log,
        "total_profit": companion.get("profit"),
        "final_positions": companion.get("positions", []),
        "submission_id": data.get("submissionId", ""),
    }


def compute_running_position(trades_df, symbol, all_times):
    """
    Build a full-timeline position series.
    Position is 0 before any trade, then steps at each trade,
    and carries forward (ffill) across all ticks.
    """
    if trades_df.empty:
        return pd.DataFrame({"time_s": all_times, "position": 0})

    sym = trades_df[
        (trades_df["symbol"] == symbol) & (trades_df["side"].isin(["BUY", "SELL"]))
    ].sort_values("timestamp").copy()

    if sym.empty:
        return pd.DataFrame({"time_s": all_times, "position": 0})

    sym["position"] = sym["signed_qty"].cumsum()

    # Build a full timeline: start at 0, step at each trade, ffill the rest
    full = pd.DataFrame({"time_s": all_times})
    # Merge trade positions onto full timeline
    trade_pos = sym[["time_s", "position"]].drop_duplicates("time_s", keep="last")
    full = full.merge(trade_pos, on="time_s", how="left")
    # Before first trade = 0
    first_trade_time = sym["time_s"].iloc[0]
    full.loc[full["time_s"] < first_trade_time, "position"] = 0
    full["position"] = full["position"].ffill().fillna(0).astype(int)

    return full


def position_limit_for_product(product: str) -> int:
    """Round 5 products all have a position limit of 10."""
    return 10


def print_log_diagnostics(parsed):
    """Console diagnostics for faster post-run debugging."""
    activities = parsed["activities"]
    trades = parsed["trades"]
    if trades.empty:
        print("  [diag] no trade history in log")
        return

    ours = trades[trades["side"].isin(["BUY", "SELL"])].copy()
    if ours.empty:
        print("  [diag] no SUBMISSION trades in this log")
        return

    print("  [diag] execution / inventory summary")
    rows = []
    for sym in sorted(ours["symbol"].unique()):
        sym_tr = ours[ours["symbol"] == sym].sort_values("timestamp").copy()
        sym_tr["signed"] = np.where(sym_tr["side"] == "BUY", sym_tr["quantity"], -sym_tr["quantity"])
        sym_tr["pos"] = sym_tr["signed"].cumsum()
        limit = position_limit_for_product(sym)
        max_inv = (sym_tr["pos"].abs().max() / max(1, limit)) * 100.0
        final_pos = int(sym_tr["pos"].iloc[-1])
        buys = int((sym_tr["side"] == "BUY").sum())
        sells = int((sym_tr["side"] == "SELL").sum())

        mid = activities[activities["product"] == sym][["timestamp", "mid_price"]].dropna().sort_values("timestamp")
        if not mid.empty:
            # Match closest prior position to each tick to estimate saturation time.
            pos_on_ticks = np.interp(
                mid["timestamp"].to_numpy(),
                sym_tr["timestamp"].to_numpy(),
                sym_tr["pos"].to_numpy(),
                left=0,
                right=sym_tr["pos"].iloc[-1],
            )
            sat_pct = float((np.abs(pos_on_ticks) >= 0.8 * limit).mean() * 100.0)
        else:
            sat_pct = float((sym_tr["pos"].abs() >= 0.8 * limit).mean() * 100.0)

        rows.append((sym, buys, sells, final_pos, max_inv, sat_pct))

    for sym, buys, sells, final_pos, max_inv, sat_pct in rows:
        print(
            f"    {sym:<24s} buys={buys:>4d} sells={sells:>4d} "
            f"final_pos={final_pos:>5d} max_inv={max_inv:>5.1f}% sat(>=80%)={sat_pct:>5.1f}%"
        )


# ═══════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════

C = {
    "bg":       "#0a0e17",
    "card":     "#111827",
    "border":   "#1f2937",
    "text":     "#f1f5f9",
    "dim":      "#64748b",
    "mid":      "#38bdf8",       # sky blue — mid price
    "best_bid": "#4ade80",       # bright green — best bid line
    "best_ask": "#f87171",       # bright red/coral — best ask line
    "your_buy": "#22c55e",       # solid green — your buy markers
    "your_sell":"#ef4444",       # solid red — your sell markers
    "bot_buy":  "#a78bfa",       # purple — other bots buy
    "bot_sell":  "#fb923c",      # orange — other bots sell
    "profit":   "#22c55e",
    "loss":     "#ef4444",
    "long_fill":"rgba(34,197,94,0.15)",
    "short_fill":"rgba(239,68,68,0.15)",
    "flat_line": "#64748b",
    "grid":     "#1e293b",
}

TOOLBAR = dict(
    displayModeBar=True,
    scrollZoom=True,
    modeBarButtonsToAdd=["drawline", "drawopenpath", "eraseshape"],
    displaylogo=False,
)

BASE = dict(
    template="plotly_dark",
    paper_bgcolor=C["card"], plot_bgcolor=C["bg"],
    font=dict(family="Inter, -apple-system, sans-serif", size=10, color=C["text"]),
    margin=dict(l=55, r=15, t=18, b=68),
    xaxis=dict(gridcolor=C["grid"], zeroline=False, showgrid=True, gridwidth=1),
    yaxis=dict(gridcolor=C["grid"], zeroline=False, showgrid=True, gridwidth=1),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=9),
        orientation="h",
        x=0,
        y=-0.22,
        xanchor="left",
        yanchor="top",
    ),
    dragmode="zoom",
)


def card_div(title, gid, h="280px"):
    return html.Div(style={
        "backgroundColor": C["card"], "borderRadius": "10px",
        "border": f"1px solid {C['border']}", "overflow": "hidden",
    }, children=[
        html.Div(title, style={
            "padding": "8px 14px", "fontSize": "10px", "fontWeight": "600",
            "color": C["dim"], "letterSpacing": "1.5px", "textTransform": "uppercase",
            "borderBottom": f"1px solid {C['border']}",
        }),
        dcc.Graph(id=gid, style={"height": h, "width": "100%"}, config=TOOLBAR),
    ])


def stat_badge(label, value, color=None):
    color = color or C["text"]
    return html.Div(style={
        "backgroundColor": C["bg"], "borderRadius": "8px",
        "padding": "8px 14px", "minWidth": "110px",
    }, children=[
        html.Div(label, style={"fontSize": "9px", "color": C["dim"],
                                "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "2px"}),
        html.Div(value, style={"fontSize": "13px", "fontWeight": "700", "color": color}),
    ])


def mini_sparkline(x, y, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=1.4), hoverinfo="skip"))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════════

def create_app(parsed, log_name):
    app = Dash(__name__, title="Round 5 Log Visualizer", suppress_callback_exceptions=True)

    activities = parsed["activities"]
    trades = parsed["trades"]
    graph_log = parsed["graph_log"]
    total_profit = parsed["total_profit"]

    products = sorted(activities["product"].unique())
    traded_products = []
    if not trades.empty:
        traded_products = sorted(
            trades.loc[trades["side"].isin(["BUY", "SELL"]), "symbol"].dropna().unique().tolist()
        )
    focus_products = traded_products if traded_products else products
    default_product = focus_products[0] if focus_products else (products[0] if products else None)
    days = sorted(activities["day_label"].unique())

    app.layout = html.Div(style={
        "backgroundColor": C["bg"], "color": C["text"],
        "fontFamily": "Inter, -apple-system, sans-serif", "minHeight": "100vh",
    }, children=[
        # Header
        html.Div(style={
            "background": f"linear-gradient(135deg, {C['card']}, {C['bg']})",
            "borderBottom": f"1px solid {C['border']}",
            "padding": "18px 28px", "display": "flex",
            "justifyContent": "space-between", "alignItems": "center",
        }, children=[
            html.Div([
                html.Span("PROSPERITY ROUND 5", style={
                    "fontSize": "16px", "fontWeight": "800",
                    "letterSpacing": "3px", "color": C["mid"]}),
                html.Span(f"  {log_name}", style={
                    "fontSize": "11px", "color": C["dim"], "marginLeft": "12px"}),
            ]),
            html.Div([
                html.Div([
                    html.Label("Day ", style={"fontSize": "10px", "color": C["dim"], "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="day",
                        options=[{"label": d, "value": d} for d in days],
                        value=days[0] if days else None,
                        className="dark-dropdown",
                        style={"width": "120px", "display": "inline-block", "verticalAlign": "middle"},
                    ),
                ], style={"display": "inline-block", "marginRight": "10px"}),
                html.Div([
                    html.Label("Product ", style={"fontSize": "10px", "color": C["dim"], "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="product",
                        options=[{"label": p, "value": p} for p in products],
                        value=default_product,
                        className="dark-dropdown",
                        style={"width": "460px", "display": "inline-block", "verticalAlign": "middle", "fontSize": "13px"},
                    ),
                ], style={"display": "inline-block"}),
                html.Div([
                    html.Label("Compare ", style={"fontSize": "10px", "color": C["dim"], "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="compare-products",
                        options=[{"label": p, "value": p} for p in products],
                        value=[default_product] if default_product else [],
                        className="dark-dropdown",
                        multi=True,
                        style={"width": "460px", "display": "inline-block", "verticalAlign": "middle", "fontSize": "13px"},
                    ),
                ], style={"display": "inline-block", "marginLeft": "10px"}),
            ]),
        ]),

        # Persistent top table
        html.Div(style={
            "padding": "12px 28px",
            "borderBottom": f"1px solid {C['border']}",
            "backgroundColor": C["card"],
        }, children=[
            html.Div("Products PnL Table", style={"fontSize": "14px", "fontWeight": 700, "marginBottom": "8px"}),
            html.Div("Click ticker to load product analysis. Table remains visible for all selections.",
                     style={"fontSize": "11px", "color": C["dim"], "marginBottom": "8px"}),
            html.Div(id="pnl-table"),
        ]),

        # Stats
        html.Div(id="stats", style={
            "padding": "14px 28px", "display": "flex", "gap": "12px",
            "flexWrap": "wrap", "borderBottom": f"1px solid {C['border']}",
        }),

        # Charts
        html.Div(style={"padding": "18px 28px", "display": "flex", "flexDirection": "column", "gap": "14px"}, children=[
            card_div("Price  /  Best Bid & Ask  /  Trades (You + Bots)", "ch-price", "390px"),
            card_div("Position — Long / Flat / Short", "ch-pos", "320px"),
            card_div("Profit & Loss", "ch-pnl", "320px"),
            card_div("Your Trades — Volume & Price", "ch-vol", "320px"),
            card_div("Spread", "ch-spread", "300px"),
            card_div("Total PnL — All Products", "ch-total", "260px"),
        ]),

        # Legend helper
        html.Div(style={
            "padding": "10px 28px 20px", "display": "flex", "gap": "20px",
            "flexWrap": "wrap", "fontSize": "11px",
        }, children=[
            html.Span(["Tip: Use toolbar above each chart — ",
                html.B("Zoom"), " (drag), ",
                html.B("Pan"), " (shift+drag), ",
                html.B("Box Select"), ", ",
                html.B("Lasso"), ", ",
                html.B("Draw"), " lines on chart"],
                style={"color": C["dim"]}),
            html.Span(f"Focused products: {', '.join(focus_products)}", style={"color": C["dim"]}),
        ]),
    ])

    # CSS
    app.index_string = '''<!DOCTYPE html><html><head>
    {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
    <style>
    *{box-sizing:border-box}body{margin:0}
    ::-webkit-scrollbar{width:6px}
    ::-webkit-scrollbar-track{background:#0a0e17}
    ::-webkit-scrollbar-thumb{background:#1f2937;border-radius:3px}
    .Select-control,
    .Select-menu-outer,
    .Select-menu,
    .Select-option,
    .Select-value,
    .Select-placeholder,
    .Select-input,
    .Select-input input{
      background:#111827!important;
      color:#f1f5f9!important;
    }
    .Select-control{border:1px solid #1f2937!important;box-shadow:none!important}
    .Select-menu-outer{border:1px solid #1f2937!important;box-shadow:0 14px 32px rgba(0,0,0,.38)!important;z-index:9999!important}
    .Select-option{color:#f1f5f9!important}
    .Select-option.is-focused,.Select-option.is-selected{background:#1f2937!important;color:#fff!important}
    .Select-value-label,.Select--single>.Select-control .Select-value{color:#f1f5f9!important}
    .Select-placeholder{color:#94a3b8!important}
    .Select--multi .Select-value{background:#1f2937!important;border-color:#334155!important;color:#f1f5f9!important}
    .Select--multi .Select-value-label{color:#f1f5f9!important}
    .Select-clear-zone{color:#94a3b8!important}
    .Select-arrow-zone .Select-arrow{border-top-color:#94a3b8!important}
    /* Make modebar visible on dark bg */
    .modebar-btn path{fill:#64748b!important}
    .modebar-btn:hover path{fill:#38bdf8!important}
    .modebar-btn.active path{fill:#38bdf8!important}
    </style></head><body>
    {%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body></html>'''

    # ── Callback ──
    @app.callback(
        [Output("pnl-table", "children"),
         Output("product", "options"),
         Output("compare-products", "options"),
         Output("ch-price", "figure"), Output("ch-pos", "figure"),
         Output("ch-pnl", "figure"), Output("ch-vol", "figure"),
         Output("ch-spread", "figure"), Output("ch-total", "figure"),
         Output("stats", "children")],
        [Input("product", "value"), Input("day", "value"), Input("compare-products", "value")],
    )
    def update(product, day, compare_products):
        empty = go.Figure().update_layout(**BASE)
        if not product or not day:
            return html.Div("No data"), [], [], *([empty]*6), []

        mask = (activities["product"] == product) & (activities["day_label"] == day)
        p = activities[mask].sort_values("time_s").reset_index(drop=True)
        if p.empty:
            return html.Div("No data"), [], [], *([empty]*6), []

        t = p["time_s"]
        mid = p["mid_price"]
        sprd = p["spread"]

        day_all = activities[activities["day_label"] == day].sort_values("time_s")
        day_products = sorted(day_all["product"].unique().tolist())
        # Sort products by final pnl descending, but force 0 pnl names to the end.
        pnl_map = {}
        for pr in day_products:
            s = day_all.loc[day_all["product"] == pr, "profit_and_loss"]
            pnl_map[pr] = float(s.iloc[-1]) if not s.empty else 0.0
        non_zero = [pr for pr in day_products if abs(pnl_map.get(pr, 0.0)) > 1e-12]
        zeros = [pr for pr in day_products if abs(pnl_map.get(pr, 0.0)) <= 1e-12]
        sorted_products = sorted(non_zero, key=lambda pr: pnl_map.get(pr, 0.0), reverse=True) + sorted(zeros)
        product_options = [{"label": f"{pr}  |  {pnl_map.get(pr, 0.0):+,.1f}", "value": pr} for pr in sorted_products]
        compare_options = [{"label": f"{pr}  |  {pnl_map.get(pr, 0.0):+,.1f}", "value": pr} for pr in sorted_products]

        # Build persistent top pnl table with colored values and mini spark lines.
        # Render as 2 columns to better use horizontal space.
        def _row_for_product(pr: str):
            pr_df = day_all[day_all["product"] == pr]
            pr_t = pr_df["time_s"]
            pr_y = pr_df["profit_and_loss"]
            val = pnl_map.get(pr, 0.0)
            pnl_color = C["profit"] if val >= 0 else C["loss"]
            spark_color = C["profit"] if val >= 0 else C["loss"]
            spark = mini_sparkline(pr_t, pr_y, spark_color) if not pr_df.empty else mini_sparkline([0, 1], [0, 0], C["dim"])
            return html.Tr([
                html.Td(
                    html.Button(
                        pr,
                        id={"type": "ticker-btn", "index": pr},
                        n_clicks=0,
                        style={
                            "background": "transparent",
                            "border": f"1px solid {C['border']}",
                            "borderRadius": "6px",
                            "padding": "6px 10px",
                            "color": C["text"],
                            "fontSize": "13px",
                            "cursor": "pointer",
                            "minWidth": "320px",
                            "textAlign": "left",
                        },
                    ),
                    style={"padding": "6px 8px"},
                ),
                html.Td(
                    f"{val:+,.1f}",
                    style={"padding": "6px 8px", "color": pnl_color, "fontWeight": 700, "fontSize": "13px", "textAlign": "right"},
                ),
                html.Td(
                    dcc.Graph(figure=spark, config={"displayModeBar": False}, style={"height": "28px", "width": "130px"}),
                    style={"padding": "4px 8px"},
                ),
            ])

        def _make_table(rows):
            return html.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Ticker", style={"textAlign": "left", "padding": "8px"}),
                        html.Th("PnL", style={"textAlign": "right", "padding": "8px"}),
                        html.Th("Spark", style={"textAlign": "left", "padding": "8px"}),
                    ])),
                    html.Tbody(rows),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            )

        split_idx = (len(sorted_products) + 1) // 2
        left_products = sorted_products[:split_idx]
        right_products = sorted_products[split_idx:]
        left_rows = [_row_for_product(pr) for pr in left_products]
        right_rows = [_row_for_product(pr) for pr in right_products]
        pnl_table = html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
            children=[
                _make_table(left_rows),
                _make_table(right_rows),
            ],
        )

        day_trades = pd.DataFrame()
        if not trades.empty:
            if "day_label" in trades.columns:
                day_trades = trades[trades["day_label"] == day].copy()
            else:
                day_trades = trades.copy()

        our = pd.DataFrame()
        if not day_trades.empty:
            our = day_trades[
                (day_trades["symbol"] == product)
                & day_trades["side"].isin(["BUY", "SELL"])
            ].sort_values("time_s")

        mkt = pd.DataFrame()
        if not day_trades.empty:
            mkt = day_trades[
                (day_trades["symbol"] == product)
                & (day_trades["side"] == "MARKET")
            ].sort_values("time_s")

        pos_full = compute_running_position(day_trades, product, t.values)
        pos_limit = position_limit_for_product(product)
        pos_lookup = pos_full.set_index("time_s")["position"]
        inv_lookup = (pos_lookup.abs() / pos_limit) * 100.0

        # ═══════════════════════════════════════
        # PRICE CHART
        # ═══════════════════════════════════════
        fig_price = go.Figure()

        # Bid-ask shaded band (subtle)
        fig_price.add_trace(go.Scatter(
            x=t, y=p["ask_price_1"], mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip"))
        fig_price.add_trace(go.Scatter(
            x=t, y=p["bid_price_1"], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(148,163,184,0.06)",
            name="Bid–Ask Band", hoverinfo="skip"))

        # Best bid line — bright green dashed
        fig_price.add_trace(go.Scatter(
            x=t, y=p["bid_price_1"], mode="lines",
            line=dict(color=C["best_bid"], width=1, dash="dash"),
            name="Best Bid", opacity=0.7))

        # Best ask line — bright red dashed
        fig_price.add_trace(go.Scatter(
            x=t, y=p["ask_price_1"], mode="lines",
            line=dict(color=C["best_ask"], width=1, dash="dash"),
            name="Best Ask", opacity=0.7))

        # Mid price — solid blue
        fig_price.add_trace(go.Scatter(
            x=t, y=mid, mode="lines",
            line=dict(color=C["mid"], width=2), name="Mid Price",
            customdata=np.column_stack([t.map(inv_lookup).fillna(0.0).values]),
            hovertemplate="time=%{x:.1f}s<br>Mid: %{y:.1f}<br>Inv used: %{customdata[0]:.1f}%<extra></extra>"))

        # Other bots' trades
        if not mkt.empty:
            fig_price.add_trace(go.Scatter(
                x=mkt["time_s"], y=mkt["price"], mode="markers",
                marker=dict(symbol="diamond", size=6, color=C["bot_buy"],
                            opacity=0.5, line=dict(width=0.5, color="white")),
                name="Bot Trades",
                customdata=np.column_stack([mkt["time_s"].map(inv_lookup).fillna(0.0).values]),
                text=[f"BOT  qty={q}  @ {pr}" for q, pr in zip(mkt["quantity"], mkt["price"])],
                hovertemplate="time=%{x:.1f}s<br>%{text}<br>Inv used: %{customdata[0]:.1f}%<extra></extra>"))

        # Your BUY — green triangle up
        if not our.empty:
            buys = our[our["side"] == "BUY"]
            sells = our[our["side"] == "SELL"]
            if not buys.empty:
                fig_price.add_trace(go.Scatter(
                    x=buys["time_s"], y=buys["price"], mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color=C["your_buy"],
                                line=dict(width=1.5, color="white")),
                    name="Your BUY",
                    customdata=np.column_stack([buys["time_s"].map(inv_lookup).fillna(0.0).values]),
                    text=[f"YOU BUY {q} @ {pr} (from ask)" for q, pr in zip(buys["quantity"], buys["price"])],
                    hovertemplate="time=%{x:.1f}s<br>%{text}<br>Inv used: %{customdata[0]:.1f}%<extra></extra>"))
            if not sells.empty:
                fig_price.add_trace(go.Scatter(
                    x=sells["time_s"], y=sells["price"], mode="markers",
                    marker=dict(symbol="triangle-down", size=12, color=C["your_sell"],
                                line=dict(width=1.5, color="white")),
                    name="Your SELL",
                    customdata=np.column_stack([sells["time_s"].map(inv_lookup).fillna(0.0).values]),
                    text=[f"YOU SELL {q} @ {pr} (to bid)" for q, pr in zip(sells["quantity"], sells["price"])],
                    hovertemplate="time=%{x:.1f}s<br>%{text}<br>Inv used: %{customdata[0]:.1f}%<extra></extra>"))

        fig_price.update_layout(**BASE, xaxis_title="Time (s)", yaxis_title="Price")

        # ═══════════════════════════════════════
        # POSITION (full timeline, shows flat too)
        # ═══════════════════════════════════════
        fig_pos = go.Figure()
        pos_t = pos_full["time_s"]
        pos_v = pos_full["position"]
        inv_used_pct = (pos_v.abs() / pos_limit) * 100.0
        inv_signed_pct = (pos_v / pos_limit) * 100.0

        # Green fill for long, red fill for short
        fig_pos.add_trace(go.Scatter(
            x=pos_t, y=pos_v.clip(lower=0), mode="none",
            fill="tozeroy", fillcolor=C["long_fill"],
            showlegend=False, hoverinfo="skip"))
        fig_pos.add_trace(go.Scatter(
            x=pos_t, y=pos_v.clip(upper=0), mode="none",
            fill="tozeroy", fillcolor=C["short_fill"],
            showlegend=False, hoverinfo="skip"))

        # Position line colored by state
        # We draw 3 separate traces for legend clarity
        long_mask = pos_v > 0
        short_mask = pos_v < 0
        flat_mask = pos_v == 0

        # Use NaN to break lines where state changes
        long_y = pos_v.copy().astype(float)
        long_y[~long_mask] = np.nan
        short_y = pos_v.copy().astype(float)
        short_y[~short_mask] = np.nan
        flat_y = pos_v.copy().astype(float)
        flat_y[~flat_mask] = np.nan

        fig_pos.add_trace(go.Scatter(
            x=pos_t, y=long_y, mode="lines",
            line=dict(color=C["your_buy"], width=2.5), name="LONG",
            customdata=np.column_stack([inv_signed_pct.values, inv_used_pct.values]),
            hovertemplate="time=%{x:.1f}s<br>LONG %{y} qty<br>Inv signed: %{customdata[0]:.1f}%<br>Inv used: %{customdata[1]:.1f}%<extra></extra>"))
        fig_pos.add_trace(go.Scatter(
            x=pos_t, y=short_y, mode="lines",
            line=dict(color=C["your_sell"], width=2.5), name="SHORT",
            customdata=np.column_stack([inv_signed_pct.values, inv_used_pct.values]),
            hovertemplate="time=%{x:.1f}s<br>SHORT %{y} qty<br>Inv signed: %{customdata[0]:.1f}%<br>Inv used: %{customdata[1]:.1f}%<extra></extra>"))
        fig_pos.add_trace(go.Scatter(
            x=pos_t, y=flat_y, mode="lines",
            line=dict(color=C["flat_line"], width=1.5, dash="dot"), name="FLAT",
            customdata=np.column_stack([inv_used_pct.values]),
            hovertemplate="time=%{x:.1f}s<br>FLAT<br>Inv used: %{customdata[0]:.1f}%<extra></extra>"))

        # Full position line (thin, connects everything)
        fig_pos.add_trace(go.Scatter(
            x=pos_t, y=pos_v, mode="lines",
            line=dict(color="rgba(148,163,184,0.3)", width=1),
            showlegend=False, hoverinfo="skip"))

        # Normalized inventory usage overlay (absolute)
        fig_pos.add_trace(go.Scatter(
            x=pos_t, y=inv_used_pct, mode="lines",
            line=dict(color="#f59e0b", width=1.6, dash="dot"),
            name="Inventory Used %",
            yaxis="y2",
            hovertemplate="time=%{x:.1f}s<br>Inv used: %{y:.1f}%<extra></extra>"))

        # Trade markers on position chart
        if not our.empty:
            buys = our[our["side"] == "BUY"]
            sells = our[our["side"] == "SELL"]
            # Look up position at each trade time
            if not buys.empty:
                buy_pos = pos_full[pos_full["time_s"].isin(buys["time_s"])]["position"]
                if len(buy_pos) == len(buys):
                    buy_inv = (buy_pos.abs() / pos_limit) * 100.0
                    fig_pos.add_trace(go.Scatter(
                        x=buys["time_s"].values, y=buy_pos.values, mode="markers",
                        marker=dict(symbol="triangle-up", size=9, color=C["your_buy"],
                                    line=dict(width=1, color="white")),
                        name="Buy Trade",
                        text=[f"BUY {q} @ {pr}" for q, pr in zip(buys["quantity"].values, buys["price"].values)],
                        customdata=np.column_stack([buy_inv.values]),
                        hovertemplate="time=%{x:.1f}s<br>%{text}<br>Inv used: %{customdata[0]:.1f}%<extra></extra>"))
            if not sells.empty:
                sell_pos = pos_full[pos_full["time_s"].isin(sells["time_s"])]["position"]
                if len(sell_pos) == len(sells):
                    sell_inv = (sell_pos.abs() / pos_limit) * 100.0
                    fig_pos.add_trace(go.Scatter(
                        x=sells["time_s"].values, y=sell_pos.values, mode="markers",
                        marker=dict(symbol="triangle-down", size=9, color=C["your_sell"],
                                    line=dict(width=1, color="white")),
                        name="Sell Trade",
                        text=[f"SELL {q} @ {pr}" for q, pr in zip(sells["quantity"].values, sells["price"].values)],
                        customdata=np.column_stack([sell_inv.values]),
                        hovertemplate="time=%{x:.1f}s<br>%{text}<br>Inv used: %{customdata[0]:.1f}%<extra></extra>"))

        # Annotations
        if pos_v.max() > 0:
            i = pos_v.idxmax()
            fig_pos.add_annotation(x=pos_t.iloc[i], y=pos_v.iloc[i],
                text=f"Peak Long: {pos_v.iloc[i]}", showarrow=True, arrowhead=2,
                font=dict(color=C["your_buy"], size=10), arrowcolor=C["your_buy"],
                bgcolor=C["bg"], borderpad=3)
        if pos_v.min() < 0:
            i = pos_v.idxmin()
            fig_pos.add_annotation(x=pos_t.iloc[i], y=pos_v.iloc[i],
                text=f"Peak Short: {pos_v.iloc[i]}", showarrow=True, arrowhead=2,
                font=dict(color=C["your_sell"], size=10), arrowcolor=C["your_sell"],
                bgcolor=C["bg"], borderpad=3)

        fig_pos.add_hline(y=0, line_color=C["border"], line_width=1)
        fig_pos.update_layout(
            **BASE,
            xaxis_title="Time (s)",
            yaxis_title="Position (qty)",
            yaxis2=dict(
                title="Inventory Used (%)",
                overlaying="y",
                side="right",
                range=[0, 105],
                showgrid=False,
                zeroline=False,
                tickmode="array",
                tickvals=[0, 25, 50, 75, 100],
            ),
        )

        # ═══════════════════════════════════════
        # PNL
        # ═══════════════════════════════════════
        fig_pnl = go.Figure()
        pnl = p["profit_and_loss"]
        final_pnl = pnl.iloc[-1]
        pnl_c = C["profit"] if final_pnl >= 0 else C["loss"]

        fig_pnl.add_trace(go.Scatter(
            x=t, y=pnl.clip(lower=0), mode="none",
            fill="tozeroy", fillcolor="rgba(34,197,94,0.12)",
            showlegend=False, hoverinfo="skip"))
        fig_pnl.add_trace(go.Scatter(
            x=t, y=pnl.clip(upper=0), mode="none",
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
            showlegend=False, hoverinfo="skip"))
        fig_pnl.add_trace(go.Scatter(
            x=t, y=pnl, mode="lines", line=dict(color=pnl_c, width=2), name="PnL",
            customdata=np.column_stack([t.map(inv_lookup).fillna(0.0).values]),
            hovertemplate="time=%{x:.1f}s<br>PnL: %{y:+.1f}<br>Inv used: %{customdata[0]:.1f}%<extra></extra>"))

        peak = pnl.max()
        peak_idx = pnl.idxmax()
        fig_pnl.add_annotation(x=t.iloc[p.index.get_loc(peak_idx)], y=peak,
            text=f"Peak: {peak:+.1f}", showarrow=True, arrowhead=2,
            font=dict(color=C["profit"], size=10), arrowcolor=C["profit"],
            bgcolor=C["bg"], borderpad=3)
        fig_pnl.add_hline(y=0, line_color=C["border"], line_width=1)
        fig_pnl.update_layout(**BASE, xaxis_title="Time (s)", yaxis_title="PnL")

        # ═══════════════════════════════════════
        # TRADE VOLUME + PRICE (dual axis)
        # ═══════════════════════════════════════
        fig_vol = go.Figure()
        if not our.empty:
            buys = our[our["side"] == "BUY"]
            sells = our[our["side"] == "SELL"]

            # Individual trade bars with price labels
            if not buys.empty:
                fig_vol.add_trace(go.Bar(
                    x=buys["time_s"], y=buys["quantity"],
                    marker=dict(color=C["your_buy"], opacity=0.8),
                    name="Buy Vol",
                    text=[f"@ {pr}" for pr in buys["price"]],
                    textposition="outside", textfont=dict(size=9, color=C["your_buy"]),
                    hovertemplate="time=%{x:.1f}s<br>BUY %{y} qty @ %{text}<extra></extra>",
                    width=8,
                ))
            if not sells.empty:
                fig_vol.add_trace(go.Bar(
                    x=sells["time_s"], y=-sells["quantity"],
                    marker=dict(color=C["your_sell"], opacity=0.8),
                    name="Sell Vol",
                    text=[f"@ {pr}" for pr in sells["price"]],
                    textposition="outside", textfont=dict(size=9, color=C["your_sell"]),
                    hovertemplate="time=%{x:.1f}s<br>SELL %{customdata} qty @ %{text}<extra></extra>",
                    customdata=sells["quantity"],
                    width=8,
                ))

            # Price line on secondary y-axis
            all_our = our.sort_values("time_s")
            fig_vol.add_trace(go.Scatter(
                x=all_our["time_s"], y=all_our["price"],
                mode="lines+markers",
                line=dict(color=C["mid"], width=1.5, dash="dot"),
                marker=dict(size=5, color=[C["your_buy"] if s == "BUY" else C["your_sell"] for s in all_our["side"]],
                            line=dict(width=0.5, color="white")),
                name="Trade Price",
                yaxis="y2",
                hovertemplate="time=%{x:.1f}s<br>Price: %{y}<extra></extra>",
            ))

            fig_vol.add_hline(y=0, line_color=C["border"], line_width=1)
            fig_vol.update_layout(
                barmode="relative",
                yaxis2=dict(
                    overlaying="y", side="right", title="Price",
                    gridcolor=C["grid"], zeroline=False, showgrid=False,
                ),
            )
        else:
            fig_vol.add_annotation(text="No trades", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font=dict(color=C["dim"], size=14))

        fig_vol.update_layout(**BASE, xaxis_title="Time (s)", yaxis_title="Volume (Buy+ / Sell-)")

        # ═══════════════════════════════════════
        # SPREAD
        # ═══════════════════════════════════════
        fig_spread = go.Figure()
        mean_s = sprd.mean()
        fig_spread.add_trace(go.Scatter(
            x=t, y=sprd, mode="lines",
            line=dict(color=C["mid"], width=1.2),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.08)", name="Spread"))
        fig_spread.add_hline(y=mean_s, line_dash="dash", line_color=C["dim"],
            annotation_text=f"Avg: {mean_s:.1f}",
            annotation_font=dict(color=C["dim"], size=10))
        fig_spread.update_layout(**BASE, xaxis_title="Time (s)", yaxis_title="Spread")

        # ═══════════════════════════════════════
        # TOTAL / COMPARE PNL
        # ═══════════════════════════════════════
        fig_total = go.Figure()
        compare_list = [pr for pr in (compare_products or []) if pr in day_products]
        if not compare_list:
            compare_list = [product]
        for pr in compare_list:
            pr_df = day_all[day_all["product"] == pr]
            if pr_df.empty:
                continue
            pr_final = float(pr_df["profit_and_loss"].iloc[-1])
            line_color = C["mid"] if pr == product else (C["profit"] if pr_final >= 0 else C["loss"])
            fig_total.add_trace(go.Scatter(
                x=pr_df["time_s"],
                y=pr_df["profit_and_loss"],
                mode="lines",
                line=dict(color=line_color, width=2.5 if pr == product else 1.7),
                name=f"{pr} ({pr_final:+.1f})",
                hovertemplate=f"time=%{{x:.1f}}s<br>{pr} PnL=%{{y:+.1f}}<extra></extra>",
            ))
        if not graph_log.empty:
            gl = graph_log
            fig_total.add_trace(go.Scatter(
                x=gl["time_s"], y=gl["value"], mode="lines",
                line=dict(color=C["profit"], width=2.5),
                fill="tozeroy", fillcolor="rgba(34,197,94,0.1)", name="Total PnL (all)"))
            if total_profit is not None:
                fig_total.add_annotation(
                    text=f"Final: {total_profit:+.1f}",
                    xref="paper", yref="paper", x=0.97, y=0.9, showarrow=False,
                    font=dict(color=C["profit"] if total_profit >= 0 else C["loss"], size=14),
                    bgcolor=C["bg"], borderpad=5)
        else:
            all_pnl = activities.groupby("timestamp")["profit_and_loss"].sum().reset_index()
            all_pnl["time_s"] = all_pnl["timestamp"] / 100
            fig_total.add_trace(go.Scatter(
                x=all_pnl["time_s"], y=all_pnl["profit_and_loss"], mode="lines",
                line=dict(color=C["profit"], width=2.5),
                fill="tozeroy", fillcolor="rgba(34,197,94,0.1)", name="Total PnL (all)"))
        fig_total.add_hline(y=0, line_color=C["border"], line_width=1)
        fig_total.update_layout(**BASE, xaxis_title="Time (s)", yaxis_title="PnL")

        # ═══════════════════════════════════════
        # STATS
        # ═══════════════════════════════════════
        n_buys = len(our[our["side"] == "BUY"]) if not our.empty else 0
        n_sells = len(our[our["side"] == "SELL"]) if not our.empty else 0
        buy_vol = our[our["side"] == "BUY"]["quantity"].sum() if not our.empty else 0
        sell_vol = our[our["side"] == "SELL"]["quantity"].sum() if not our.empty else 0
        n_bot = len(mkt) if not mkt.empty else 0
        final_pos = pos_v.iloc[-1]
        pos_label = f"LONG {final_pos}" if final_pos > 0 else (f"SHORT {abs(final_pos)}" if final_pos < 0 else "FLAT")
        pos_c = C["your_buy"] if final_pos > 0 else (C["your_sell"] if final_pos < 0 else C["dim"])

        stats = [
            stat_badge("Final PnL", f"{final_pnl:+.1f}",
                       C["profit"] if final_pnl >= 0 else C["loss"]),
            stat_badge("Total Profit", f"{total_profit:+.1f}" if total_profit is not None else "—",
                       C["profit"] if (total_profit or 0) >= 0 else C["loss"]),
            stat_badge("Position", pos_label, pos_c),
            stat_badge("Inventory Used", f"{inv_used_pct.iloc[-1]:.1f}% / max {inv_used_pct.max():.1f}%",
                       C["your_sell"] if inv_used_pct.iloc[-1] > 80 else C["text"]),
            stat_badge("Your Buys", f"{n_buys} trades  |  {buy_vol} qty", C["your_buy"]),
            stat_badge("Your Sells", f"{n_sells} trades  |  {sell_vol} qty", C["your_sell"]),
            stat_badge("Bot Trades", f"{n_bot}", C["bot_buy"]),
            stat_badge("Price Range", f"{mid.min():.0f} – {mid.max():.0f}"),
            stat_badge("Avg Spread", f"{mean_s:.1f}"),
        ]

        return pnl_table, product_options, compare_options, fig_price, fig_pos, fig_pnl, fig_vol, fig_spread, fig_total, stats

    @app.callback(
        Output("product", "value"),
        [Input({"type": "ticker-btn", "index": ALL}, "n_clicks")],
        [State({"type": "ticker-btn", "index": ALL}, "id"), State("product", "value")],
        prevent_initial_call=True,
    )
    def select_product_from_table(_clicks, ids, current):
        if not ids:
            return no_update
        trig = ctx.triggered_id
        if isinstance(trig, dict) and trig.get("type") == "ticker-btn":
            return trig.get("index", current)
        return no_update

    return app


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prosperity Round 5 log visualizer")
    parser.add_argument("log_file", help="Path to a .log file or a submission logs folder")
    parser.add_argument("--port", "-p", type=int, default=8050)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--diag", action="store_true",
                        help="Print extra execution/inventory diagnostics to console")
    args = parser.parse_args()

    resolved = resolve_log_path(args.log_file)
    if not resolved:
        print(f"Not found or no .log file inside: {args.log_file}")
        sys.exit(1)

    log_name = os.path.basename(os.path.dirname(resolved))
    print(f"Parsing: {resolved}")
    parsed = parse_log(resolved)

    a = parsed["activities"]
    tr = parsed["trades"]
    ours = tr[tr["side"].isin(["BUY", "SELL"])] if not tr.empty else tr
    print(f"  Products: {', '.join(a['product'].unique())}")
    print(f"  Ticks: {len(a):,}  |  Your trades: {len(ours)}")
    if parsed["total_profit"] is not None:
        print(f"  Total profit: {parsed['total_profit']:+.1f}")
    if args.diag:
        print_log_diagnostics(parsed)

    app = create_app(parsed, log_name)

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    print(f"\n  http://localhost:{args.port}\n")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
