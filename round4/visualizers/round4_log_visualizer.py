"""
round4_log_visualizer.py
========================
Round 4 log/data visualizer adapted for:
  - 12 tradable algo products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, and
    10 VEV_xxxx voucher strikes (VEV_4000, VEV_4500, VEV_5000..VEV_5500,
    VEV_6000, VEV_6500).
  - Counterparty info now disclosed in trade data
    (buyer/seller = Mark 01, Mark 14, Mark 22, Mark 38, Mark 49, Mark 55,
     Mark 67, plus SUBMISSION).

Inputs (any combination):
  --data DIR            Round 4 dataset dir with prices_round_4_day_*.csv and
                        trades_round_4_day_*.csv (semicolon-delimited).
  --pnl FILE            Optional backtester pnl csv (pos_<P>, ppnl_<P> cols).
  --fills FILE          Optional backtester fills csv
                        (timestamp,product,side,price,quantity).
  --out-dir DIR         Output directory for HTML files (default ./viz_out).
  --products A,B,C      Restrict to subset of products.

Outputs:
  index.html                       Summary + links.
  <product>_view.html              Per-product chart (mid + spread + market
                                   trades colored by counterparty + optional
                                   backtester fills/position/pnl).
  counterparty_flow.html           Net buy minus sell volume per
                                   counterparty per product.
  voucher_surface.html             Mid price across strikes vs time
                                   (option smile context).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ALGO_PRODUCTS = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_PRODUCTS = [f"VEV_{k}" for k in VOUCHER_STRIKES]
ALL_PRODUCTS = ALGO_PRODUCTS + VOUCHER_PRODUCTS

KNOWN_MARKS = [
    "Mark 01", "Mark 14", "Mark 22", "Mark 38",
    "Mark 49", "Mark 55", "Mark 67", "SUBMISSION",
]
MARK_COLORS = {
    "Mark 01": "#1f77b4", "Mark 14": "#ff7f0e", "Mark 22": "#2ca02c",
    "Mark 38": "#d62728", "Mark 49": "#9467bd", "Mark 55": "#8c564b",
    "Mark 67": "#e377c2", "SUBMISSION": "#111111", "UNKNOWN": "#7f7f7f",
}


def _load_prices_dir(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "prices_*.csv")))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(p, delimiter=";") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "day" in df.columns:
        df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
        df["t_global"] = df["day"].astype(int) * 1_000_000 + df["timestamp"].astype(int)
    else:
        df["t_global"] = df["timestamp"].astype(int)
    return df


def _load_trades_dir(data_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, "trades_*.csv")))
    if not paths:
        return pd.DataFrame()
    frames = []
    for p in paths:
        df = pd.read_csv(p, delimiter=";")
        m = re.search(r"day_(-?\d+)", os.path.basename(p))
        df["day"] = int(m.group(1)) if m else 0
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    df["t_global"] = df["day"].astype(int) * 1_000_000 + df["timestamp"].astype(int)
    df["buyer"] = df["buyer"].fillna("UNKNOWN").astype(str).str.strip()
    df["seller"] = df["seller"].fillna("UNKNOWN").astype(str).str.strip()
    return df


def _signed_fill_qty(side: str, qty: float) -> float:
    if isinstance(side, str) and side.upper().startswith("BUY"):
        return abs(float(qty))
    return -abs(float(qty))


def _build_product_figure(
    product: str,
    prices: pd.DataFrame,
    trades: pd.DataFrame,
    pnl: Optional[pd.DataFrame],
    fills: Optional[pd.DataFrame],
) -> go.Figure:
    pdf = prices[prices["product"] == product].copy() if not prices.empty else pd.DataFrame()
    tdf = trades[trades["symbol"] == product].copy() if not trades.empty else pd.DataFrame()

    has_pnl = pnl is not None and f"pos_{product}" in pnl.columns
    rows = 3 if has_pnl else 2
    titles = [
        f"{product} - mid / bid / ask",
        f"{product} - market trades by counterparty",
    ]
    if has_pnl:
        titles.append(f"{product} - position & PnL (backtester)")

    specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
    if has_pnl:
        specs.append([{"secondary_y": True}])
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=titles, specs=specs,
    )

    if not pdf.empty:
        fig.add_trace(go.Scatter(x=pdf["t_global"], y=pdf["mid_price"], mode="lines",
                                 name="mid", line=dict(color="#0ea5e9", width=1.6)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=pdf["t_global"], y=pdf["bid_price_1"], mode="lines",
                                 name="bid1", line=dict(color="#16a34a", width=1, dash="dot"),
                                 opacity=0.6),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=pdf["t_global"], y=pdf["ask_price_1"], mode="lines",
                                 name="ask1", line=dict(color="#dc2626", width=1, dash="dot"),
                                 opacity=0.6),
                      row=1, col=1)

    if not tdf.empty:
        # Color each market trade by the non-SUBMISSION participant's role.
        # When neither side is SUBMISSION, color by buyer.
        for mark in KNOWN_MARKS + ["UNKNOWN"]:
            sub = tdf[(tdf["buyer"] == mark) | (tdf["seller"] == mark)]
            if sub.empty:
                continue
            sizes = np.clip(sub["quantity"].astype(float).abs() * 1.2 + 4, 4, 22)
            hover = [
                f"t={t}<br>price={p}<br>qty={q}<br>buyer={b}<br>seller={s}"
                for t, p, q, b, s in zip(sub["t_global"], sub["price"], sub["quantity"],
                                         sub["buyer"], sub["seller"])
            ]
            fig.add_trace(go.Scatter(
                x=sub["t_global"], y=sub["price"], mode="markers",
                name=mark, marker=dict(size=sizes, color=MARK_COLORS.get(mark, "#7f7f7f"),
                                       line=dict(width=0.4, color="#222")),
                hovertext=hover, hoverinfo="text",
                legendgroup=mark,
            ), row=2, col=1)

    if has_pnl:
        pos_col = f"pos_{product}"
        ppnl_col = f"ppnl_{product}"
        fig.add_trace(go.Scatter(x=pnl["t_global"], y=pnl[pos_col], mode="lines",
                                 name="position", line=dict(color="#f59e0b", width=1.8)),
                      row=3, col=1, secondary_y=False)
        if ppnl_col in pnl.columns:
            fig.add_trace(go.Scatter(x=pnl["t_global"], y=pnl[ppnl_col], mode="lines",
                                     name="asset PnL", line=dict(color="#7c3aed", width=1.8)),
                          row=3, col=1, secondary_y=True)
        if fills is not None and not fills.empty:
            af = fills[fills["product"] == product].copy()
            if not af.empty:
                af["sq"] = [_signed_fill_qty(s, q) for s, q in zip(af["side"], af["quantity"])]
                colors = ["#16a34a" if v > 0 else "#dc2626" for v in af["sq"]]
                fig.add_trace(go.Bar(x=af["t_global"], y=af["sq"], name="own fill qty",
                                     marker_color=colors, opacity=0.55),
                              row=3, col=1, secondary_y=False)

    fig.update_layout(
        title=f"Round 4 view - {product}",
        template="plotly_white", height=900,
        legend=dict(orientation="h", y=1.06),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    fig.update_xaxes(title_text="t_global (day*1e6 + timestamp)", row=rows, col=1)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="trade price", row=2, col=1)
    if has_pnl:
        fig.update_yaxes(title_text="position", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="PnL", row=3, col=1, secondary_y=True)
    return fig


def _build_counterparty_flow(trades: pd.DataFrame, products: List[str]) -> go.Figure:
    rows = []
    for prod in products:
        sub = trades[trades["symbol"] == prod]
        if sub.empty:
            continue
        for mark in KNOWN_MARKS:
            buy_qty = int(sub.loc[sub["buyer"] == mark, "quantity"].sum())
            sell_qty = int(sub.loc[sub["seller"] == mark, "quantity"].sum())
            net = buy_qty - sell_qty
            if buy_qty == 0 and sell_qty == 0:
                continue
            rows.append({"product": prod, "mark": mark, "buy": buy_qty,
                         "sell": sell_qty, "net": net})
    df = pd.DataFrame(rows)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No counterparty data", template="plotly_white")
        return fig
    pivot = df.pivot(index="mark", columns="product", values="net").fillna(0)
    pivot = pivot.reindex([m for m in KNOWN_MARKS if m in pivot.index])
    pivot = pivot[[c for c in products if c in pivot.columns]]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="RdBu", zmid=0,
        hovertemplate="mark=%{y}<br>product=%{x}<br>net=%{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Counterparty net flow (buy qty - sell qty) per product",
        template="plotly_white", height=480,
        margin=dict(l=80, r=40, t=70, b=80),
    )
    return fig


def _build_voucher_surface(prices: pd.DataFrame) -> go.Figure:
    sub = prices[prices["product"].isin(VOUCHER_PRODUCTS)].copy()
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(title="No voucher price data", template="plotly_white")
        return fig
    fig = make_subplots(rows=1, cols=1)
    for k in VOUCHER_STRIKES:
        prod = f"VEV_{k}"
        s = sub[sub["product"] == prod]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["t_global"], y=s["mid_price"], mode="lines",
            name=prod, line=dict(width=1.4),
        ))
    fig.update_layout(
        title="Voucher mid prices across strikes (log y)",
        template="plotly_white", height=560,
        legend=dict(orientation="h", y=1.06),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    fig.update_yaxes(type="log", title_text="mid (log)")
    fig.update_xaxes(title_text="t_global")
    return fig


def _summary_table(prices: pd.DataFrame, trades: pd.DataFrame, products: List[str]) -> pd.DataFrame:
    rows = []
    for prod in products:
        pdf = prices[prices["product"] == prod] if not prices.empty else pd.DataFrame()
        tdf = trades[trades["symbol"] == prod] if not trades.empty else pd.DataFrame()
        rows.append({
            "product": prod,
            "n_quotes": len(pdf),
            "mid_min": float(pdf["mid_price"].min()) if not pdf.empty else np.nan,
            "mid_max": float(pdf["mid_price"].max()) if not pdf.empty else np.nan,
            "mid_last": float(pdf["mid_price"].iloc[-1]) if not pdf.empty else np.nan,
            "n_trades": len(tdf),
            "trade_qty": int(tdf["quantity"].sum()) if not tdf.empty else 0,
            "n_counterparties": int(pd.unique(pd.concat([tdf["buyer"], tdf["seller"]])).size) if not tdf.empty else 0,
        })
    return pd.DataFrame(rows)


def _write_index(out_dir: str, products: List[str], summary: pd.DataFrame) -> None:
    rows_html = []
    for prod in products:
        rows_html.append(
            f'<li><a href="{prod}_view.html">{prod}</a></li>'
        )
    table_html = summary.to_html(index=False, float_format=lambda v: f"{v:.2f}")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Round 4 visualizer</title>
<style>body{{font-family:system-ui,Arial;margin:24px;color:#222}}
h1{{margin-top:0}} table{{border-collapse:collapse}}
th,td{{border:1px solid #ddd;padding:6px 10px;font-size:13px}}
th{{background:#f3f4f6}} a{{color:#0ea5e9}}</style></head>
<body>
<h1>Round 4 - log/data visualizer</h1>
<p>Counterparties disclosed: {", ".join(KNOWN_MARKS[:-1])}.</p>
<h2>Per-product views</h2>
<ul>{''.join(rows_html)}</ul>
<h2>Cross-product views</h2>
<ul>
  <li><a href="counterparty_flow.html">Counterparty net flow heatmap</a></li>
  <li><a href="voucher_surface.html">Voucher mid surface</a></li>
</ul>
<h2>Summary</h2>
{table_html}
</body></html>
"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Dataset/ROUND_4",
                    help="Dir with prices_*.csv and trades_*.csv (semicolon delimited)")
    ap.add_argument("--pnl", default=None, help="Optional backtester pnl csv")
    ap.add_argument("--fills", default=None, help="Optional backtester fills csv")
    ap.add_argument("--out-dir", default="viz_out")
    ap.add_argument("--products", default=None,
                    help="Comma-separated subset; default = all 12 products")
    args = ap.parse_args()

    products = ALL_PRODUCTS if not args.products else [p.strip() for p in args.products.split(",")]

    prices = _load_prices_dir(args.data) if args.data and os.path.isdir(args.data) else pd.DataFrame()
    trades = _load_trades_dir(args.data) if args.data and os.path.isdir(args.data) else pd.DataFrame()

    pnl = None
    if args.pnl and os.path.exists(args.pnl):
        pnl = pd.read_csv(args.pnl)
        if "day" in pnl.columns:
            pnl["t_global"] = pnl["day"].astype(int) * 1_000_000 + pnl["timestamp"].astype(int)
        else:
            pnl["t_global"] = pnl["timestamp"].astype(int)

    fills = None
    if args.fills and os.path.exists(args.fills):
        fills = pd.read_csv(args.fills)
        if "day" in fills.columns:
            fills["t_global"] = fills["day"].astype(int) * 1_000_000 + fills["timestamp"].astype(int)
        else:
            fills["t_global"] = fills["timestamp"].astype(int)

    os.makedirs(args.out_dir, exist_ok=True)

    for prod in products:
        fig = _build_product_figure(prod, prices, trades, pnl, fills)
        out = os.path.join(args.out_dir, f"{prod}_view.html")
        fig.write_html(out, include_plotlyjs="cdn")
        print(f"  wrote {out}")

    cp_fig = _build_counterparty_flow(trades, products)
    cp_fig.write_html(os.path.join(args.out_dir, "counterparty_flow.html"),
                      include_plotlyjs="cdn")
    vs_fig = _build_voucher_surface(prices)
    vs_fig.write_html(os.path.join(args.out_dir, "voucher_surface.html"),
                      include_plotlyjs="cdn")

    summary = _summary_table(prices, trades, products)
    _write_index(args.out_dir, products, summary)
    print(f"Index: {os.path.join(args.out_dir, 'index.html')}")


if __name__ == "__main__":
    main()
