import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import glob
import os

app = dash.Dash(__name__, title="Spread Signal Analysis")

# Find dataset
def load_data():
    data_dir = "Datasets"
    if not os.path.exists(data_dir):
        # Look for it generically
        for root, dirs, files in os.walk("."):
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
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    
    # Calculate imbalance
    bid_vols = prices["bid_volume_1"].fillna(0)
    ask_vols = prices["ask_volume_1"].fillna(0)
    denom = bid_vols + ask_vols
    prices["imbalance"] = np.where(denom > 0, (bid_vols - ask_vols) / denom, 0.0)
    
    # Calculate book depth
    bid_cols = [c for c in prices.columns if c.startswith("bid_volume")]
    ask_cols = [c for c in prices.columns if c.startswith("ask_volume")]
    prices["book_depth"] = prices[bid_cols].fillna(0).sum(axis=1) + prices[ask_cols].fillna(0).sum(axis=1)
    
    # Calculate returns (used for volatility)
    prices = prices.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)
    prices["ret"] = prices.groupby(["day", "product"])["mid_price"].pct_change().fillna(0.0)
    
    # Calculate continuous timestamp
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

def make_num_input(id_val, label, default_val):
    return html.Div([
        html.Label(label, style={"color": "#8ea0b8", "fontSize": "12px", "display": "block", "marginBottom": "4px"}),
        dcc.Input(
            id=id_val,
            type="number",
            value=default_val,
            min=1,
            step=1,
            debounce=True,
            style={"width": "60px", "backgroundColor": "#101722", "color": "#e5edf7", "border": "1px solid #243246", "padding": "4px"}
        )
    ], style={"marginRight": "15px", "marginBottom": "10px"})

def make_float_input(id_val, label, default_val):
    return html.Div([
        html.Label(label, style={"color": "#8ea0b8", "fontSize": "12px", "display": "block", "marginBottom": "4px"}),
        dcc.Input(
            id=id_val,
            type="number",
            value=default_val,
            step=0.1,
            debounce=True,
            style={"width": "70px", "backgroundColor": "#101722", "color": "#e5edf7", "border": "1px solid #243246", "padding": "4px"}
        )
    ], style={"marginRight": "15px", "marginBottom": "10px"})

def make_text_input(id_val, label, default_val, width="150px"):
    return html.Div([
        html.Label(label, style={"color": "#8ea0b8", "fontSize": "12px", "display": "block", "marginBottom": "4px"}),
        dcc.Input(
            id=id_val,
            type="text",
            value=default_val,
            debounce=True,
            style={"width": width, "backgroundColor": "#101722", "color": "#e5edf7", "border": "1px solid #243246", "padding": "4px"}
        )
    ], style={"marginRight": "15px", "marginBottom": "10px"})

app.layout = html.Div([
    html.H1("Spread Signal Analysis", style={"color": "#e5edf7", "fontFamily": "sans-serif"}),
    html.Div([
        html.Label("Select Product:", style={"color": "#e5edf7", "marginRight": "10px", "fontFamily": "sans-serif"}),
        dcc.Dropdown(
            id="product-dropdown",
            options=[{"label": p, "value": p} for p in products],
            value=products[0] if products else None,
            style={"width": "300px", "color": "#000"}
        ),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "marginBottom": "10px"}),
    
    html.Div([
        make_num_input("mid-filter-window", "Mid Filter Window", 10),
        make_text_input("multi-sma-windows", "Extra SMA Windows (csv)", "10, 20"),
        make_num_input("spread-window", "Spread Window", 50),
        make_num_input("imbalance-window", "Imbalance Window", 50),
        make_num_input("volatility-window", "Volatility Window", 50),
        make_num_input("depth-window", "Depth Window", 50),
    ], style={"display": "flex", "flexWrap": "wrap", "flexDirection": "row", "marginBottom": "10px", "fontFamily": "sans-serif"}),
    
    html.Div([
        make_text_input("multi-sma-weights", "Extra SMA Wts (csv)", "1.0, -0.5"),
        make_float_input("spread-weight", "Spread Wt (a)", 1.0),
        make_float_input("imbalance-weight", "Imbalance Wt (b)", 0.0),
        make_float_input("volatility-weight", "Vol Wt (c)", 0.0),
        make_float_input("depth-weight", "Depth Wt (d)", 0.0),
        html.Div([
            html.Label("Smoothing Type", style={"color": "#8ea0b8", "fontSize": "12px", "display": "block", "marginBottom": "4px"}),
            dcc.RadioItems(
                id="smoothing-type",
                options=[{"label": " SMA", "value": "SMA"}, {"label": " EMA", "value": "EMA"}],
                value="SMA",
                inline=True,
                labelStyle={"color": "#e5edf7", "marginRight": "10px", "cursor": "pointer"}
            )
        ], style={"marginRight": "15px", "marginLeft": "10px"}),
        make_float_input("ema-alpha", "EMA Alpha (>0)", 0.0),
        make_num_input("future-shift", "Predict Future (steps)", 0),
    ], style={"marginBottom": "20px", "display": "flex", "flexWrap": "wrap", "alignItems": "center"}),
    
    dcc.Graph(id="sma-dual-axis", style={"height": "500px", "marginBottom": "20px"}),

    dcc.Graph(id="spread-scatter", style={"height": "400px"}),
    
    html.Div([
        dcc.Graph(id="spread-component", style={"width": "50%", "height": "350px", "display": "inline-block"}),
        dcc.Graph(id="imbalance-component", style={"width": "50%", "height": "350px", "display": "inline-block"}),
    ]),
    html.Div([
        dcc.Graph(id="volatility-component", style={"width": "50%", "height": "350px", "display": "inline-block"}),
        dcc.Graph(id="depth-component", style={"width": "50%", "height": "350px", "display": "inline-block"}),
    ])
], style={"backgroundColor": "#090d14", "minHeight": "100vh", "padding": "20px"})

@app.callback(
    [Output("spread-scatter", "figure"), Output("sma-dual-axis", "figure"),
     Output("spread-component", "figure"), Output("imbalance-component", "figure"),
     Output("volatility-component", "figure"), Output("depth-component", "figure")],
    [Input("product-dropdown", "value"), 
     Input("mid-filter-window", "value"),
     Input("multi-sma-windows", "value"),
     Input("spread-window", "value"),
     Input("imbalance-window", "value"),
     Input("volatility-window", "value"),
     Input("depth-window", "value"),
     Input("multi-sma-weights", "value"),
     Input("spread-weight", "value"),
     Input("imbalance-weight", "value"),
     Input("volatility-weight", "value"),
     Input("depth-weight", "value"),
     Input("smoothing-type", "value"),
     Input("ema-alpha", "value"),
     Input("future-shift", "value")]
)
def update_charts(selected_product, mid_win, multi_sma_wins_str, spread_win, imb_win, vol_win, depth_win, 
                  multi_sma_wts_str, w_spread, w_imb, w_vol, w_depth, smooth_type, ema_alpha, future_shift):
    if not selected_product or prices_df.empty:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
        
    df = prices_df[prices_df["product"] == selected_product].copy()
    df = df.sort_values("continuous_timestamp")
    
    # Calculate independent SMAs or EMAs
    mid_win = max(1, mid_win or 1)
    try:
        sma_wins = [int(x.strip()) for x in str(multi_sma_wins_str).split(",") if x.strip()]
    except:
        sma_wins = []
    
    df["raw_volatility"] = df["ret"].rolling(window=max(2, vol_win or 2), min_periods=2).std(ddof=0).bfill() * 10000
    
    if smooth_type == "EMA":
        if ema_alpha and 0 < ema_alpha <= 1.0:
            df["filtered_mid_price"] = df["mid_price"].ewm(alpha=ema_alpha, adjust=False).mean()
            for i, win in enumerate(sma_wins):
                df[f"multi_sma_{i}_smooth"] = df["spread"].ewm(alpha=ema_alpha, adjust=False).mean()
            df["spread_smooth"] = df["spread"].ewm(alpha=ema_alpha, adjust=False).mean()
            df["imbalance_smooth"] = df["imbalance"].ewm(alpha=ema_alpha, adjust=False).mean()
            df["volatility_smooth"] = df["raw_volatility"].ewm(alpha=ema_alpha, adjust=False).mean()
            df["depth_smooth"] = df["book_depth"].ewm(alpha=ema_alpha, adjust=False).mean()
        else:
            df["filtered_mid_price"] = df["mid_price"].ewm(span=mid_win, adjust=False).mean()
            for i, win in enumerate(sma_wins):
                df[f"multi_sma_{i}_smooth"] = df["spread"].ewm(span=max(1, win), adjust=False).mean()
            df["spread_smooth"] = df["spread"].ewm(span=max(1, spread_win or 1), adjust=False).mean()
            df["imbalance_smooth"] = df["imbalance"].ewm(span=max(1, imb_win or 1), adjust=False).mean()
            df["volatility_smooth"] = df["raw_volatility"].ewm(span=max(1, vol_win or 1), adjust=False).mean()
            df["depth_smooth"] = df["book_depth"].ewm(span=max(1, depth_win or 1), adjust=False).mean()
    else:
        df["filtered_mid_price"] = df["mid_price"].rolling(window=mid_win, min_periods=1).mean().bfill()
        for i, win in enumerate(sma_wins):
            df[f"multi_sma_{i}_smooth"] = df["spread"].rolling(window=max(1, win), min_periods=1).mean().bfill()
        df["spread_smooth"] = df["spread"].rolling(window=max(1, spread_win or 1), min_periods=1).mean().bfill()
        df["imbalance_smooth"] = df["imbalance"].rolling(window=max(1, imb_win or 1), min_periods=1).mean().bfill()
        df["volatility_smooth"] = df["raw_volatility"].rolling(window=max(1, vol_win or 1), min_periods=1).mean().bfill()
        df["depth_smooth"] = df["book_depth"].rolling(window=max(1, depth_win or 1), min_periods=1).mean().bfill()
    
    # Shift filtered mid price backwards to represent future prediction
    shift_val = -(future_shift or 0)
    if shift_val != 0:
        df["target_mid_price"] = df["filtered_mid_price"].shift(shift_val)
    else:
        df["target_mid_price"] = df["filtered_mid_price"]
        
    try:
        sma_wts = [float(x.strip()) for x in str(multi_sma_wts_str).split(",") if x.strip()]
    except:
        sma_wts = []
        
    w_spread = w_spread or 0.0
    w_imb = w_imb or 0.0
    w_vol = w_vol or 0.0
    w_depth = w_depth or 0.0
    
    custom_signal = (
        df["spread_smooth"] * w_spread +
        df["imbalance_smooth"] * w_imb +
        df["volatility_smooth"] * w_vol +
        df["depth_smooth"] * w_depth
    )
    for i, wt in enumerate(sma_wts):
        if i < len(sma_wins):
            custom_signal += df[f"multi_sma_{i}_smooth"] * wt
            
    df["custom_signal"] = custom_signal
    
    # Scatter plot for Spread Distribution
    scatter_fig = go.Figure()
    scatter_fig.add_trace(go.Scatter(
        x=df["continuous_timestamp"],
        y=df["spread"],
        mode="markers",
        marker=dict(
            size=4,
            color=df["spread"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Spread Value")
        ),
        name="Spread"
    ))
    
    # Add vertical lines for day boundaries
    day_starts = df.groupby("day")["continuous_timestamp"].min().sort_index()
    for idx, (day_val, x_val) in enumerate(day_starts.items()):
        if idx > 0:
            scatter_fig.add_vline(x=x_val, line_dash="dash", line_color="gray", opacity=0.5)
            
    scatter_fig.update_layout(
        title=f"{selected_product} - Spread Signals Distribution",
        xaxis_title="Continuous Timestamp",
        yaxis_title="Spread",
        template="plotly_dark",
        paper_bgcolor="#101722",
        plot_bgcolor="#101722",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(color="#e5edf7")
    )
    
    # Dual Axis for Mid Price vs Signals
    dual_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Faint raw mid price
    dual_fig.add_trace(
        go.Scatter(x=df["continuous_timestamp"], y=df["mid_price"], mode="lines", name="Raw Mid Price", line=dict(color="#38bdf8", width=1, dash="dot"), opacity=0.4),
        secondary_y=False,
    )
    
    # Bright bid/ask prices
    dual_fig.add_trace(
        go.Scatter(x=df["continuous_timestamp"], y=df["bid_price_1"], mode="lines", name="Best Bid", line=dict(color="#10b981", width=1.5)),
        secondary_y=False,
    )
    dual_fig.add_trace(
        go.Scatter(x=df["continuous_timestamp"], y=df["ask_price_1"], mode="lines", name="Best Ask", line=dict(color="#ef4444", width=1.5)),
        secondary_y=False,
    )
    
    # Bold Target mid price
    target_name = f"Target Mid Price ({mid_win} {smooth_type}, +{future_shift or 0} steps)" if (future_shift or 0) > 0 else f"Filtered Mid Price ({mid_win} {smooth_type})"
    dual_fig.add_trace(
        go.Scatter(x=df["continuous_timestamp"], y=df["target_mid_price"], mode="lines", name=target_name, line=dict(color="#38bdf8", width=2)),
        secondary_y=False,
    )
    
    # Custom Signal on secondary axis
    dual_fig.add_trace(
        go.Scatter(x=df["continuous_timestamp"], y=df["custom_signal"], mode="lines", name=f"Custom Signal ({smooth_type})", line=dict(color="#a855f7", width=2)),
        secondary_y=True,
    )
    
    for idx, (day_val, x_val) in enumerate(day_starts.items()):
        if idx > 0:
            dual_fig.add_vline(x=x_val, line_dash="dash", line_color="gray", opacity=0.5)
            
    dual_fig.update_layout(
        title=f"{selected_product} - Filtered Mid Price vs Fitted Signal",
        xaxis_title="Continuous Timestamp",
        template="plotly_dark",
        paper_bgcolor="#101722",
        plot_bgcolor="#101722",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(color="#e5edf7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    dual_fig.update_yaxes(title_text="Mid Price", secondary_y=False)
    dual_fig.update_yaxes(title_text="Signal Component", secondary_y=True)
    
    def make_comp_fig(raw_col, smooth_col, title):
        fig = go.Figure()
        if raw_col:
            fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df[raw_col], mode="lines", name=f"Raw {title}", line=dict(color="#38bdf8", width=1), opacity=0.3))
        if smooth_col:
            fig.add_trace(go.Scatter(x=df["continuous_timestamp"], y=df[smooth_col], mode="lines", name=f"{smooth_type}", line=dict(color="#f59e0b", width=2)))
        for idx, (_, x_val) in enumerate(day_starts.items()):
            if idx > 0:
                fig.add_vline(x=x_val, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(title=title, template="plotly_dark", paper_bgcolor="#101722", plot_bgcolor="#101722", margin=dict(l=40, r=40, t=40, b=40), font=dict(color="#e5edf7"), showlegend=False)
        return fig
        
    spread_fig = make_comp_fig("spread", "spread_smooth", "Spread Signal")
    imb_fig = make_comp_fig("imbalance", "imbalance_smooth", "Imbalance Signal")
    vol_fig = make_comp_fig("raw_volatility", "volatility_smooth", "Volatility Signal")
    depth_fig = make_comp_fig("book_depth", "depth_smooth", "Depth Signal")
    
    return scatter_fig, dual_fig, spread_fig, imb_fig, vol_fig, depth_fig

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8061)
