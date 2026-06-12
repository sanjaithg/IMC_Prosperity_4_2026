import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import glob
import os

app = dash.Dash(__name__, title="Signal Analysis")

def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "..", "Datasets")
    if not os.path.exists(data_dir):
        for root, dirs, files in os.walk(os.path.join(base, "..")):
            if "prices_round_5_day_2.csv" in files:
                data_dir = root
                break

    price_frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "prices_round_*_day_*.csv"))):
        df = pd.read_csv(path, sep=";")
        price_frames.append(df)

    if not price_frames:
        return pd.DataFrame()

    prices = pd.concat(price_frames, ignore_index=True)
    prices = prices.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)

    # Imbalance
    bid_vols = prices["bid_volume_1"].fillna(0)
    ask_vols = prices["ask_volume_1"].fillna(0)
    denom = bid_vols + ask_vols
    prices["imbalance"] = np.where(denom > 0, (bid_vols - ask_vols) / denom, 0.0)

    # Book depth (relative to price)
    bid_cols = [c for c in prices.columns if c.startswith("bid_volume")]
    ask_cols = [c for c in prices.columns if c.startswith("ask_volume")]
    prices["book_depth"] = prices[bid_cols].fillna(0).sum(axis=1) + prices[ask_cols].fillna(0).sum(axis=1)
    prices["rel_depth"] = prices["book_depth"] / prices["mid_price"]

    # Returns & volatility
    prices["ret"] = prices.groupby(["day", "product"])["mid_price"].pct_change().fillna(0.0)

    # Continuous timestamp
    day_order = sorted(prices["day"].unique())
    unique_timestamps = np.sort(prices["timestamp"].unique())
    positive_steps = np.diff(unique_timestamps)
    positive_steps = positive_steps[positive_steps > 0]
    timestamp_step = int(np.median(positive_steps)) if len(positive_steps) else 1
    day_span = int(prices["timestamp"].max() + timestamp_step)
    day_offsets = {day: i * day_span for i, day in enumerate(day_order)}
    prices["continuous_timestamp"] = prices["timestamp"] + prices["day"].map(day_offsets).astype(int)

    return prices

print("Loading data...")
prices_df = load_data()
products = sorted(prices_df["product"].unique()) if not prices_df.empty else []
print(f"Data loaded. Found {len(products)} products.")

# --- Precompute lag correlations for all products ---
LAGS = [1, 3, 5, 10, 25, 50, 100]
CORR_FEATURES = ["imbalance", "rel_depth", "volatility_raw"]

print("Precomputing lag correlations...")
lag_corr_data = {}
for prod in products:
    df = prices_df[prices_df["product"] == prod].copy()
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)
    df["volatility_raw"] = df["ret"].rolling(50, min_periods=2).std(ddof=0).fillna(0.0)
    corrs = {}
    for lag in LAGS:
        delta = df["mid_price"].shift(-lag) - df["mid_price"]
        mask = ~delta.isna()
        for feat in CORR_FEATURES:
            x = df[feat].values
            y = delta.values
            valid = mask.values & ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() > 30:
                corrs[(feat, lag)] = np.corrcoef(x[valid], y[valid])[0, 1]
            else:
                corrs[(feat, lag)] = 0.0
    lag_corr_data[prod] = corrs
print("Done.")


def make_num_input(id_val, label, default_val, min_val=1):
    return html.Div([
        html.Label(label, style={"color": "#8ea0b8", "fontSize": "12px", "display": "block", "marginBottom": "4px"}),
        dcc.Input(
            id=id_val, type="number", value=default_val, min=min_val, step=1, debounce=True,
            style={"width": "60px", "backgroundColor": "#101722", "color": "#e5edf7",
                   "border": "1px solid #243246", "padding": "4px"}
        )
    ], style={"marginRight": "15px", "marginBottom": "10px"})


app.layout = html.Div([
    html.H1("Signal Predictor Analysis", style={"color": "#e5edf7", "fontFamily": "sans-serif", "marginBottom": "8px"}),

    # Row 1: Product selector
    html.Div([
        html.Label("Product:", style={"color": "#e5edf7", "marginRight": "10px", "fontFamily": "sans-serif"}),
        dcc.Dropdown(
            id="product-dropdown",
            options=[{"label": p, "value": p} for p in products],
            value=products[0] if products else None,
            style={"width": "300px", "color": "#000"}
        ),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),

    # Row 2: Feature windows
    html.Div([
        make_num_input("mid-filter-window", "Mid Filter Win", 10),
        make_num_input("imb-short-window", "Imb Short Win", 5),
        make_num_input("imb-med-window", "Imb Medium Win", 25),
        make_num_input("imb-long-window", "Imb Long Win", 100),
        make_num_input("depth-short-window", "Depth Short Win", 10),
        make_num_input("depth-med-window", "Depth Medium Win", 50),
        make_num_input("depth-long-window", "Depth Long Win", 100),
        make_num_input("vol-window", "Volatility Win", 50),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "10px", "fontFamily": "sans-serif"}),

    # Row 3: Prediction settings
    html.Div([
        make_num_input("future-shift", "Predict Future (steps)", 10, min_val=0),
        html.Div([
            html.Label("Auto-Fit OLS", style={"color": "#8ea0b8", "fontSize": "12px", "display": "block", "marginBottom": "4px"}),
            dcc.Checklist(
                id="auto-fit",
                options=[{"label": " Fit features → Δprice", "value": "fit"}],
                value=["fit"],
                labelStyle={"color": "#facc15", "fontSize": "13px", "cursor": "pointer"}
            )
        ], style={"marginRight": "20px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "marginBottom": "10px"}),

    # Fit stats bar
    html.Div(id="fit-stats", style={
        "fontFamily": "monospace", "fontSize": "12px", "color": "#94a3b8",
        "backgroundColor": "#101722", "border": "1px solid #1e3a5f",
        "borderRadius": "6px", "padding": "10px 16px", "marginBottom": "12px",
        "display": "flex", "flexWrap": "wrap", "gap": "18px"
    }),

    # Lag-correlation heatmap
    dcc.Graph(id="lag-corr-heatmap", style={"height": "220px", "marginBottom": "12px"}),

    # Main price + signal chart
    dcc.Graph(id="main-chart", style={"height": "500px", "marginBottom": "16px"}),

    # Component charts
    html.Div([
        dcc.Graph(id="imbalance-chart", style={"width": "50%", "height": "320px", "display": "inline-block"}),
        dcc.Graph(id="depth-chart", style={"width": "50%", "height": "320px", "display": "inline-block"}),
    ]),
    html.Div([
        dcc.Graph(id="volatility-chart", style={"width": "50%", "height": "320px", "display": "inline-block"}),
        dcc.Graph(id="delta-chart", style={"width": "50%", "height": "320px", "display": "inline-block"}),
    ]),

], style={"backgroundColor": "#090d14", "minHeight": "100vh", "padding": "20px"})


@app.callback(
    [Output("lag-corr-heatmap", "figure"), Output("main-chart", "figure"),
     Output("imbalance-chart", "figure"), Output("depth-chart", "figure"),
     Output("volatility-chart", "figure"), Output("delta-chart", "figure"),
     Output("fit-stats", "children")],
    [Input("product-dropdown", "value"),
     Input("mid-filter-window", "value"),
     Input("imb-short-window", "value"),
     Input("imb-med-window", "value"),
     Input("imb-long-window", "value"),
     Input("depth-short-window", "value"),
     Input("depth-med-window", "value"),
     Input("depth-long-window", "value"),
     Input("vol-window", "value"),
     Input("future-shift", "value"),
     Input("auto-fit", "value")]
)
def update_all(product, mid_win, imb_s, imb_m, imb_l, dep_s, dep_m, dep_l, vol_win,
               future_shift, auto_fit):
    empty = go.Figure()
    if not product or prices_df.empty:
        return empty, empty, empty, empty, empty, empty, []

    df = prices_df[prices_df["product"] == product].copy()
    df = df.sort_values("continuous_timestamp").reset_index(drop=True)

    mid_win = max(1, mid_win or 1)
    imb_s = max(1, imb_s or 1)
    imb_m = max(1, imb_m or 1)
    imb_l = max(1, imb_l or 1)
    dep_s = max(1, dep_s or 1)
    dep_m = max(1, dep_m or 1)
    dep_l = max(1, dep_l or 1)
    vol_win = max(2, vol_win or 2)
    k = max(0, future_shift or 0)

    # Smoothed features
    df["filtered_mid"] = df["mid_price"].rolling(mid_win, min_periods=1).mean().bfill()
    df["imb_short"]  = df["imbalance"].rolling(imb_s, min_periods=1).mean().bfill()
    df["imb_med"]    = df["imbalance"].rolling(imb_m, min_periods=1).mean().bfill()
    df["imb_long"]   = df["imbalance"].rolling(imb_l, min_periods=1).mean().bfill()
    df["dep_short"]  = df["rel_depth"].rolling(dep_s, min_periods=1).mean().bfill()
    df["dep_med"]    = df["rel_depth"].rolling(dep_m, min_periods=1).mean().bfill()
    df["dep_long"]   = df["rel_depth"].rolling(dep_l, min_periods=1).mean().bfill()
    df["vol_smooth"] = df["ret"].rolling(vol_win, min_periods=2).std(ddof=0).bfill() * 10000

    # --- Lag correlation heatmap for this product ---
    corrs = lag_corr_data.get(product, {})
    feat_labels = ["Imbalance", "Rel Depth", "Volatility"]
    z = []
    for feat in CORR_FEATURES:
        row = [corrs.get((feat, lag), 0.0) for lag in LAGS]
        z.append(row)

    heatmap_fig = go.Figure(go.Heatmap(
        z=z, x=[str(l) for l in LAGS], y=feat_labels,
        colorscale="RdYlGn", zmid=0, zmin=-0.15, zmax=0.15,
        text=[[f"{v:+.3f}" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=12),
        colorbar=dict(title="Corr", len=0.9)
    ))
    heatmap_fig.update_layout(
        title=f"{product} — Feature → Future Δprice Correlation by Lag",
        xaxis_title="Lag (steps ahead)",
        template="plotly_dark", paper_bgcolor="#101722", plot_bgcolor="#101722",
        margin=dict(l=100, r=40, t=40, b=40), font=dict(color="#e5edf7"),
        height=220
    )

    # --- OLS on Δprice ---
    fit_stats_children = []
    do_fit = auto_fit and "fit" in auto_fit

    feature_names = ["ImbShort", "ImbMed", "ImbLong", "DepShort", "DepMed", "DepLong", "Volatility"]
    feature_cols  = ["imb_short", "imb_med", "imb_long", "dep_short", "dep_med", "dep_long", "vol_smooth"]

    if k > 0:
        price_delta = df["mid_price"].shift(-k) - df["mid_price"]
    else:
        price_delta = df["mid_price"].diff().fillna(0.0)

    df["actual_delta"] = price_delta

    if do_fit:
        X = df[feature_cols].values
        X = np.hstack([np.ones((len(X), 1)), X])
        y = price_delta.values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)

        if mask.sum() > len(feature_cols) + 1:
            beta, _, _, _ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
            df["pred_delta"] = X @ beta

            y_pred = X[mask] @ beta
            y_true = y[mask]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            nonzero = y_true != 0
            dir_acc = np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero])) if nonzero.sum() > 0 else 0.5

            lag_label = f"+{k} steps" if k > 0 else "1-step"
            fit_stats_children = [
                html.Span(f"OLS Δprice  |  {lag_label}  |  R²= {r2:.4f}  |  Dir Acc= {dir_acc:.1%}",
                          style={"color": "#facc15", "fontWeight": "bold", "marginRight": "20px"}),
            ] + [
                html.Span(f"{name}: {coef:+.4f}",
                          style={"marginRight": "14px",
                                 "color": "#4ade80" if abs(coef) > 0.5 else "#94a3b8"})
                for name, coef in zip(feature_names, beta[1:])
            ]

    # --- Day boundaries ---
    day_starts = df.groupby("day")["continuous_timestamp"].min().sort_index()

    def add_day_lines(fig):
        for idx, (_, x_val) in enumerate(day_starts.items()):
            if idx > 0:
                fig.add_vline(x=x_val, line_dash="dash", line_color="gray", opacity=0.4)

    # --- Main chart: price + OLS delta comparison ---
    main_fig = make_subplots(specs=[[{"secondary_y": True}]])

    main_fig.add_trace(
        go.Scatter(x=df["continuous_timestamp"], y=df["mid_price"],
                   mode="lines", name="Mid Price",
                   line=dict(color="#38bdf8", width=1.5)),
        secondary_y=False)

    main_fig.add_trace(
        go.Scatter(x=df["continuous_timestamp"], y=df["filtered_mid"],
                   mode="lines", name=f"Filtered Mid ({mid_win})",
                   line=dict(color="#818cf8", width=2)),
        secondary_y=False)

    if do_fit and "pred_delta" in df.columns:
        ts = df["continuous_timestamp"].values
        main_fig.add_trace(
            go.Scatter(x=ts, y=df["pred_delta"],
                       mode="lines", name=f"Predicted Δ ({k}s)",
                       line=dict(color="#facc15", width=2)),
            secondary_y=True)

        main_fig.add_trace(
            go.Scatter(x=ts, y=df["actual_delta"],
                       mode="lines", name=f"Actual Δ ({k}s)",
                       line=dict(color="#fb923c", width=1), opacity=0.4),
            secondary_y=True)

    add_day_lines(main_fig)
    main_fig.update_layout(
        title=f"{product} — Mid Price + OLS Δprice Prediction",
        xaxis_title="Continuous Timestamp",
        template="plotly_dark", paper_bgcolor="#101722", plot_bgcolor="#101722",
        margin=dict(l=40, r=40, t=40, b=40), font=dict(color="#e5edf7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    main_fig.update_yaxes(title_text="Mid Price", secondary_y=False)
    main_fig.update_yaxes(title_text="Δ Price", secondary_y=True)

    # --- Component charts ---
    dark_layout = dict(template="plotly_dark", paper_bgcolor="#101722",
                       plot_bgcolor="#101722", margin=dict(l=40, r=40, t=40, b=40),
                       font=dict(color="#e5edf7"), showlegend=True,
                       legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"))

    # Imbalance
    imb_fig = go.Figure()
    imb_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["imbalance"],
                                 mode="lines", name="Raw", line=dict(color="#38bdf8", width=1), opacity=0.25))
    imb_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["imb_short"],
                                 mode="lines", name=f"Short ({imb_s})", line=dict(color="#4ade80", width=2)))
    imb_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["imb_med"],
                                 mode="lines", name=f"Med ({imb_m})", line=dict(color="#f59e0b", width=2)))
    imb_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["imb_long"],
                                 mode="lines", name=f"Long ({imb_l})", line=dict(color="#ef4444", width=2)))
    add_day_lines(imb_fig)
    imb_fig.update_layout(title="Imbalance (bid-ask volume ratio)", **dark_layout)

    # Depth
    dep_fig = go.Figure()
    dep_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["rel_depth"],
                                 mode="lines", name="Raw", line=dict(color="#38bdf8", width=1), opacity=0.25))
    dep_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["dep_short"],
                                 mode="lines", name=f"Short ({dep_s})", line=dict(color="#4ade80", width=2)))
    dep_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["dep_med"],
                                 mode="lines", name=f"Med ({dep_m})", line=dict(color="#f59e0b", width=2)))
    dep_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["dep_long"],
                                 mode="lines", name=f"Long ({dep_l})", line=dict(color="#ef4444", width=2)))
    add_day_lines(dep_fig)
    dep_fig.update_layout(title="Relative Book Depth (total volume / price)", **dark_layout)

    # Volatility
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["vol_smooth"],
                                 mode="lines", name=f"Volatility ({vol_win})",
                                 line=dict(color="#f59e0b", width=2)))
    add_day_lines(vol_fig)
    vol_fig.update_layout(title="Volatility (rolling std of returns × 10000)", **dark_layout)

    # Delta comparison
    delta_fig = go.Figure()
    if do_fit and "pred_delta" in df.columns:
        delta_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["actual_delta"],
                                       mode="lines", name="Actual Δ",
                                       line=dict(color="#fb923c", width=1), opacity=0.5))
        delta_fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df["pred_delta"],
                                       mode="lines", name="Predicted Δ",
                                       line=dict(color="#facc15", width=2)))
    add_day_lines(delta_fig)
    delta_fig.update_layout(title=f"Δprice: Predicted vs Actual (+{k} steps)", **dark_layout)

    return heatmap_fig, main_fig, imb_fig, dep_fig, vol_fig, delta_fig, fit_stats_children


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8061)
