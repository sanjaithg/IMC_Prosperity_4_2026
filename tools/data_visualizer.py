"""
Prosperity Round 5 market data explorer.

Designed for the Round 5 dataset:
  - 50 products at every timestamp
  - prices_round_5_day_*.csv files with 3 book levels
  - trades_round_5_day_*.csv files with blank buyer/seller fields
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import webbrowser
from threading import Timer
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


C = {
    "bg": "#090d14",
    "panel": "#101722",
    "panel2": "#141d2b",
    "border": "#243246",
    "text": "#e5edf7",
    "muted": "#8ea0b8",
    "dim": "#637086",
    "accent": "#38bdf8",
    "accent2": "#f59e0b",
    "good": "#22c55e",
    "bad": "#ef4444",
    "grid": "#223047",
}


ROUND5_GROUPS: Dict[str, List[str]] = {
    "Galaxy Sounds Recorders": [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "Vertical Sleeping Pods": [
        "SLEEP_POD_SUEDE",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_COTTON",
    ],
    "Organic Microchips": [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_TRIANGLE",
    ],
    "Purification Pebbles": [
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",
    ],
    "Domestic Robots": [
        "ROBOT_VACUUMING",
        "ROBOT_MOPPING",
        "ROBOT_DISHES",
        "ROBOT_LAUNDRY",
        "ROBOT_IRONING",
    ],
    "UV-Visors": [
        "UV_VISOR_YELLOW",
        "UV_VISOR_AMBER",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_MAGENTA",
    ],
    "Instant Translators": [
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ],
    "Construction Panels": [
        "PANEL_1X2",
        "PANEL_2X2",
        "PANEL_1X4",
        "PANEL_2X4",
        "PANEL_4X4",
    ],
    "Liquid Breath Oxygen Shakes": [
        "OXYGEN_SHAKE_MORNING_BREATH",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_GARLIC",
    ],
    "Protein Snack Packs": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ],
}

PRODUCT_TO_GROUP = {
    product: group for group, products in ROUND5_GROUPS.items() for product in products
}
ROUND5_PRODUCT_ORDER = [product for group_products in ROUND5_GROUPS.values() for product in group_products]
CATEGORY_RANK = {category: i for i, category in enumerate(ROUND5_GROUPS)}
PRODUCT_RANK = {product: i for i, product in enumerate(ROUND5_PRODUCT_ORDER)}
ROLLING_WINDOWS = (20, 50, 100)
_MAX_CHART_POINTS = 4_000


def _ds(series_or_array, n: int = _MAX_CHART_POINTS):
    """Uniform downsample a pandas Series or numpy array to at most n points."""
    if hasattr(series_or_array, "__len__") and len(series_or_array) > n:
        idx = np.linspace(0, len(series_or_array) - 1, n, dtype=int)
        if hasattr(series_or_array, "iloc"):
            return series_or_array.iloc[idx]
        return series_or_array[idx]
    return series_or_array


def _ds_df(df: pd.DataFrame, n: int = _MAX_CHART_POINTS) -> pd.DataFrame:
    """Uniform downsample a DataFrame to at most n rows."""
    if len(df) > n:
        idx = np.linspace(0, len(df) - 1, n, dtype=int)
        return df.iloc[idx].reset_index(drop=True)
    return df
MACD_PRESETS = {
    "8,21,5": (8, 21, 5),
    "12,26,9": (12, 26, 9),
    "20,50,9": (20, 50, 9),
}


def _product_sort_key(product: str) -> tuple[int, int, str]:
    category = _category(product)
    return (CATEGORY_RANK.get(category, 999), PRODUCT_RANK.get(product, 999), product)


def _ordered_products(items) -> List[str]:
    unique = dict.fromkeys([item for item in items if item])
    return sorted(unique, key=_product_sort_key)


def _macd_key(value: str | None) -> str:
    key = str(value or "12,26,9").replace(" ", "")
    return key if key in MACD_PRESETS else "12,26,9"


def _code_cache_version() -> str:
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    paths = [
        __file__,
        os.path.join(assets_dir, "dash_dropdown_dark.css"),
        os.path.join(assets_dir, "preserve_scroll.js"),
    ]
    mtimes = [os.path.getmtime(path) for path in paths if os.path.exists(path)]
    return str(max(mtimes) if mtimes else 0)


DROPDOWN_CSS = """
*{box-sizing:border-box}
html,body{max-width:100%;overflow-x:hidden}
body{margin:0;background:#090d14;color:#e5edf7}
.dash-graph,.js-plotly-plot{max-width:100%}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:#090d14}
::-webkit-scrollbar-thumb{background:#243246;border-radius:4px}
.dark-dropdown {
  font-size:8px!important;
  width:100%;
  min-width:0;
  height:24px;
}
.dark-dropdown,
.dark-dropdown * {
  font-size:8px!important;
}
.dark-dropdown .Select,
.dark-dropdown .Select-control,
.dark-dropdown .Select-multi-value-wrapper {
  width:100%!important;
}
.dark-dropdown .Select-control,
.dark-dropdown .Select-menu-outer,
.dark-dropdown .Select-menu,
.dark-dropdown .Select-option,
.dark-dropdown .Select-value,
.dark-dropdown .Select-placeholder,
.dark-dropdown .Select-input,
.dark-dropdown .Select-input input {
  background:#101722!important;
  color:#e5edf7!important;
}
.dark-dropdown .Select-control {
  border:1px solid #243246!important;
  box-shadow:none!important;
  min-height:24px;
  height:24px;
  max-height:24px;
  overflow:hidden;
}
.dark-dropdown .Select-placeholder,
.dark-dropdown .Select--single > .Select-control .Select-value,
.dark-dropdown .Select.has-value.Select--single > .Select-control .Select-value {
  line-height:22px!important;
  height:22px!important;
  padding-left:8px!important;
  padding-right:22px!important;
}
.dark-dropdown .Select-input {
  height:22px!important;
  line-height:22px!important;
  padding-left:8px!important;
}
.dark-dropdown .Select-input input {
  padding:2px 0!important;
  line-height:18px!important;
}
.dark-dropdown .Select-multi-value-wrapper {
  height:22px;
  max-height:22px;
  overflow:hidden;
  white-space:nowrap;
}
.dark-dropdown .Select-menu-outer {
  border:1px solid #243246!important;
  box-shadow:0 14px 32px rgba(0,0,0,.38)!important;
  z-index:9999!important;
}
.dark-dropdown .Select-option { color:#e5edf7!important; }
.dark-dropdown .Select-option,
.dark-dropdown .VirtualizedSelectOption,
.dark-dropdown .Select-noresults {
  font-size:8px!important;
  line-height:18px!important;
  min-height:20px!important;
  padding:4px 8px!important;
}
.dark-dropdown .Select-option.is-focused,
.dark-dropdown .Select-option.is-selected {
  background:#1e293b!important;
  color:#ffffff!important;
}
.dark-dropdown .Select-value-label,
.dark-dropdown .Select--single > .Select-control .Select-value,
.dark-dropdown .Select-placeholder {
  color:#e5edf7!important;
  font-size:8px!important;
}
.dark-dropdown .Select--multi .Select-value {
  background:#1e293b!important;
  border-color:#334155!important;
  color:#e5edf7!important;
  margin-top:2px!important;
  margin-bottom:0!important;
  max-width:92px;
  height:18px!important;
  line-height:16px!important;
  overflow:hidden;
  font-size:7px!important;
}
.dark-dropdown .Select--multi .Select-value-label {
  color:#e5edf7!important;
  overflow:hidden;
  text-overflow:ellipsis;
  line-height:16px!important;
  padding:1px 5px!important;
  font-size:7px!important;
}
.dark-dropdown .Select--multi .Select-value-icon {
  line-height:16px!important;
  padding:1px 4px!important;
  font-size:7px!important;
}
.dark-dropdown .Select-arrow-zone,
.dark-dropdown .Select-clear-zone {
  width:22px!important;
  padding-right:4px!important;
}
.dark-dropdown .Select-arrow-zone .Select-arrow {
  border-top-color:#8ea0b8!important;
}
.dark-dropdown .Select-clear-zone { color:#8ea0b8!important; }
.modebar-btn path{fill:#8ea0b8!important}
.modebar-btn:hover path{fill:#38bdf8!important}
.modebar-btn.active path{fill:#38bdf8!important}
"""


def find_data_folder() -> str | None:
    candidates = [
        "Datasets",
        "Dataset",
        "Round_5/Dataset",
        "Dataset/ROUND_5",
        "Datasets/ROUND_5",
        "Data",
        "data",
        ".",
    ]
    for path in candidates:
        if os.path.isdir(path) and glob.glob(os.path.join(path, "prices_round_*_day_*.csv")):
            return path
    for path in glob.glob("**/prices_round_*_day_*.csv", recursive=True):
        if "venv" not in path.split(os.sep):
            return os.path.dirname(path)
    return None


def _day_from_filename(path: str) -> int:
    match = re.search(r"day_(-?\d+)", os.path.basename(path))
    return int(match.group(1)) if match else 0


def _category(product: str) -> str:
    return PRODUCT_TO_GROUP.get(product, "Other")


def load_data(data_dir: str, days: List[int] | None = None, timestamp_stride: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wanted_days = set(days or [])
    timestamp_stride = max(1, int(timestamp_stride or 1))
    bid_price_cols = ["bid_price_1", "bid_price_2", "bid_price_3"]
    ask_price_cols = ["ask_price_1", "ask_price_2", "ask_price_3"]
    bid_vol_cols = ["bid_volume_1", "bid_volume_2", "bid_volume_3"]
    ask_vol_cols = ["ask_volume_1", "ask_volume_2", "ask_volume_3"]
    price_cols = [
        "day",
        "timestamp",
        "product",
        *bid_price_cols,
        *bid_vol_cols,
        *ask_price_cols,
        *ask_vol_cols,
        "mid_price",
    ]
    price_dtypes = {
        "day": "int16",
        "timestamp": "int32",
        "product": "category",
        **{col: "float32" for col in [*bid_price_cols, *ask_price_cols, "mid_price"]},
        **{col: "float32" for col in [*bid_vol_cols, *ask_vol_cols]},
    }
    price_frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "prices_round_*_day_*.csv"))):
        file_day = _day_from_filename(path)
        if wanted_days and file_day not in wanted_days:
            continue
        skiprows = None
        if timestamp_stride > 1:
            products_per_timestamp = len(ROUND5_PRODUCT_ORDER)
            skiprows = lambda i, stride=timestamp_stride, n=products_per_timestamp: i > 0 and ((i - 1) // n) % stride != 0
        df = pd.read_csv(path, sep=";", usecols=price_cols, dtype=price_dtypes, skiprows=skiprows)
        price_frames.append(df)

    if not price_frames:
        raise FileNotFoundError(f"No prices_round_*_day_*.csv files found in {data_dir}")

    prices = pd.concat(price_frames, ignore_index=True)
    prices = prices.dropna(subset=["day", "timestamp", "product"]).copy()
    prices["day"] = prices["day"].astype("int16")
    prices["timestamp"] = prices["timestamp"].astype("int32")
    prices["product"] = pd.Categorical(prices["product"].astype(str), categories=ROUND5_PRODUCT_ORDER)
    prices["category"] = pd.Categorical(prices["product"].astype(str).map(_category), categories=list(ROUND5_GROUPS))
    prices["spread"] = (prices["ask_price_1"] - prices["bid_price_1"]).astype("float32")
    bid_vols = prices[bid_vol_cols].fillna(0).abs()
    ask_vols = prices[ask_vol_cols].fillna(0).abs()
    prices["bid_level_count"] = ((prices[bid_price_cols].notna().to_numpy()) & (bid_vols.to_numpy() > 0)).sum(axis=1).astype("int8")
    prices["ask_level_count"] = ((prices[ask_price_cols].notna().to_numpy()) & (ask_vols.to_numpy() > 0)).sum(axis=1).astype("int8")
    prices["bid_total_vol"] = bid_vols.sum(axis=1).astype("float32")
    prices["ask_total_vol"] = ask_vols.sum(axis=1).astype("float32")
    denom = prices["bid_total_vol"] + prices["ask_total_vol"]
    prices["imbalance"] = np.where(denom > 0, (prices["bid_total_vol"] - prices["ask_total_vol"]) / denom, 0.0).astype("float32")
    prices["book_depth"] = denom.astype("float32")
    bid_notional = (prices[bid_price_cols].fillna(0).to_numpy() * bid_vols.to_numpy()).sum(axis=1)
    ask_notional = (prices[ask_price_cols].fillna(0).to_numpy() * ask_vols.to_numpy()).sum(axis=1)
    weighted_bid = np.divide(
        bid_notional,
        prices["bid_total_vol"],
        out=np.full(len(prices), np.nan, dtype="float32"),
        where=prices["bid_total_vol"].to_numpy() > 0,
    )
    weighted_ask = np.divide(
        ask_notional,
        prices["ask_total_vol"],
        out=np.full(len(prices), np.nan, dtype="float32"),
        where=prices["ask_total_vol"].to_numpy() > 0,
    )
    prices["weighted_mid_price"] = np.where(
        denom > 0,
        (weighted_ask * prices["bid_total_vol"] + weighted_bid * prices["ask_total_vol"]) / denom,
        prices["mid_price"],
    )
    prices["weighted_mid_price"] = prices["weighted_mid_price"].fillna(prices["mid_price"]).astype("float32")
    bid_v1 = prices["bid_volume_1"].fillna(0).abs()
    ask_v1 = prices["ask_volume_1"].fillna(0).abs()
    top_denom = bid_v1 + ask_v1
    prices["microprice"] = np.where(
        top_denom > 0,
        (prices["ask_price_1"] * bid_v1 + prices["bid_price_1"] * ask_v1) / top_denom,
        prices["mid_price"],
    ).astype("float32")
    prices = prices.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)
    day_order = sorted(prices["day"].unique())
    unique_timestamps = np.sort(prices["timestamp"].unique())
    positive_steps = np.diff(unique_timestamps)
    positive_steps = positive_steps[positive_steps > 0]
    timestamp_step = int(np.median(positive_steps)) if len(positive_steps) else 1
    day_span = int(prices["timestamp"].max() + timestamp_step)
    day_offsets = {day: i * day_span for i, day in enumerate(day_order)}
    prices["continuous_timestamp"] = (prices["timestamp"] + prices["day"].map(day_offsets).astype("int32")).astype("int32")
    prices["ret"] = prices.groupby(["day", "product"], observed=True)["mid_price"].pct_change().fillna(0.0).astype("float32")
    kept_timestamps = {
        int(day): set(group["timestamp"].unique().tolist())
        for day, group in prices.groupby("day", observed=True)
    }

    trade_frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "trades_round_*_day_*.csv"))):
        file_day = _day_from_filename(path)
        if wanted_days and file_day not in wanted_days:
            continue
        df = pd.read_csv(
            path,
            sep=";",
            usecols=["timestamp", "symbol", "price", "quantity"],
            dtype={"timestamp": "int32", "symbol": "category", "price": "float32", "quantity": "int16"},
        )
        df["day"] = np.int16(file_day)
        if timestamp_stride > 1:
            df = df[df["timestamp"].isin(kept_timestamps.get(file_day, set()))]
        trade_frames.append(df)

    if trade_frames:
        trades = pd.concat(trade_frames, ignore_index=True)
        trades["day"] = trades["day"].astype("int16")
        trades["timestamp"] = trades["timestamp"].astype("int32")
        trades["symbol"] = pd.Categorical(trades["symbol"].astype(str), categories=ROUND5_PRODUCT_ORDER)
        trades["category"] = pd.Categorical(trades["symbol"].astype(str).map(_category), categories=list(ROUND5_GROUPS))
        trades["continuous_timestamp"] = (trades["timestamp"] + trades["day"].map(day_offsets).fillna(0).astype("int32")).astype("int32")
    else:
        trades = pd.DataFrame(columns=["timestamp", "symbol", "price", "quantity", "day", "category", "continuous_timestamp"])

    return prices, trades


def theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        font=dict(color=C["text"], family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=10),
        margin=dict(l=44, r=24, t=50, b=72),
        title=dict(text=title, x=0.015, y=0.98, font=dict(size=13)) if title else None,
        legend=dict(
            orientation="h",
            x=0,
            y=-0.22,
            xanchor="left",
            yanchor="top",
            font=dict(size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor=C["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], zeroline=False)
    return fig


def empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=13))
    return theme(fig)


def panel(children, style=None):
    base = {
        "background": C["panel"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "8px",
        "padding": "12px",
    }
    if style:
        base.update(style)
    return html.Div(children=children, style=base)


def stat(label: str, value: str, color: str = C["text"]):
    return panel([
        html.Div(label, style={
            "fontSize": "7px",
            "color": C["muted"],
            "textTransform": "uppercase",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        }),
        html.Div(value, style={
            "fontSize": "11px",
            "fontWeight": 800,
            "color": color,
            "marginTop": "1px",
            "whiteSpace": "nowrap",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        }),
    ], style={"padding": "5px 6px", "borderRadius": "6px"})


def is_all_days(day) -> bool:
    return str(day) == "all"


def add_day_boundaries(fig: go.Figure, frame: pd.DataFrame, x_col: str, day) -> go.Figure:
    if not is_all_days(day) or frame.empty or "day" not in frame.columns or x_col not in frame.columns:
        return fig
    starts = frame.groupby("day")[x_col].min().sort_index()
    for idx, (day_value, x_value) in enumerate(starts.items()):
        if idx == 0:
            continue
        fig.add_shape(
            type="line",
            x0=float(x_value),
            x1=float(x_value),
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=C["muted"], width=1.4, dash="dash"),
            opacity=0.85,
        )
        fig.add_annotation(
            x=float(x_value),
            y=1,
            xref="x",
            yref="paper",
            text=f"Day {int(day_value)}",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=9, color=C["muted"]),
            bgcolor=C["panel"],
            bordercolor=C["border"],
            borderpad=2,
        )
    return fig


def day_summary(day_prices: pd.DataFrame, day_trades: pd.DataFrame) -> pd.DataFrame:
    summary = (
        day_prices.sort_values(["product", "day", "timestamp"])
        .groupby(["product", "category"], as_index=False, observed=True)
        .agg(
            start=("mid_price", "first"),
            end=("mid_price", "last"),
            high=("mid_price", "max"),
            low=("mid_price", "min"),
            avg_spread=("spread", "mean"),
            med_spread=("spread", "median"),
            avg_depth=("bid_total_vol", "mean"),
            avg_ask_depth=("ask_total_vol", "mean"),
            avg_abs_imb=("imbalance", lambda s: float(np.abs(s).mean())),
            vol_bps=("ret", lambda s: float(s.std(ddof=0) * 10_000)),
        )
    )
    summary["return_bps"] = (summary["end"] / summary["start"] - 1.0) * 10_000
    summary["range_bps"] = (summary["high"] / summary["low"] - 1.0) * 10_000
    summary["avg_book_depth"] = summary["avg_depth"] + summary["avg_ask_depth"]

    if day_trades.empty:
        summary["n_trades"] = 0
        summary["trade_qty"] = 0
        summary["avg_trade_qty"] = 0.0
    else:
        trade_counts = (
            day_trades.groupby("symbol", observed=True)["quantity"]
            .agg(n_trades="count", trade_qty="sum", avg_trade_qty="mean")
            .reset_index()
            .rename(columns={"symbol": "product"})
        )
        summary = summary.merge(trade_counts, on="product", how="left")
        summary[["n_trades", "trade_qty", "avg_trade_qty"]] = summary[
            ["n_trades", "trade_qty", "avg_trade_qty"]
        ].fillna(0)

    summary["efficiency_score"] = (
        summary["return_bps"].abs()
        + summary["vol_bps"] * 4
        + summary["range_bps"] * 0.2
        + summary["n_trades"] * 0.2
        - summary["avg_spread"] * 2
    )
    summary["category_rank"] = summary["category"].map(CATEGORY_RANK).fillna(999).astype(int)
    summary["product_rank"] = summary["product"].map(PRODUCT_RANK).fillna(999).astype(int)
    return summary.sort_values(["category_rank", "product_rank", "product"]).reset_index(drop=True)


def _add_cached_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["product", "day", "timestamp"]).reset_index(drop=True).copy()

    for window in ROLLING_WINDOWS:
        mid_ma = frame["mid_price"].rolling(window, min_periods=5).mean()
        mid_std = frame["mid_price"].rolling(window, min_periods=5).std(ddof=0)
        spread_ma = frame["spread"].rolling(window, min_periods=1).mean()
        spread_ma_mean = spread_ma.rolling(window, min_periods=5).mean()
        spread_ma_std = spread_ma.rolling(window, min_periods=5).std(ddof=0)

        frame[f"mid_ma_{window}"] = mid_ma.astype("float32")
        frame[f"mid_std_{window}"] = mid_std.astype("float32")
        frame[f"spread_ma_{window}"] = spread_ma.astype("float32")
        frame[f"spread_price_corr_{window}"] = spread_ma.rolling(window, min_periods=5).corr(frame["mid_price"]).astype("float32")
        frame[f"imbalance_ma_{window}"] = frame["imbalance"].rolling(window, min_periods=1).mean().astype("float32")
        frame[f"book_depth_ma_{window}"] = frame["book_depth"].rolling(window, min_periods=1).mean().astype("float32")
        frame[f"book_vol_bps_{window}"] = (frame["ret"].rolling(window, min_periods=5).std(ddof=0) * 10_000).astype("float32")
        frame[f"zscore_{window}"] = np.where(
            mid_std > 0,
            (frame["mid_price"] - mid_ma) / mid_std,
            0.0,
        ).astype("float32")
        frame[f"spread_ma_zscore_{window}"] = np.where(
            spread_ma_std > 0,
            (spread_ma - spread_ma_mean) / spread_ma_std,
            0.0,
        ).astype("float32")

    for key, (fast, slow, signal) in MACD_PRESETS.items():
        suffix = key.replace(",", "_")
        ema_fast = frame["mid_price"].ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = frame["mid_price"].ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
        frame[f"macd_{suffix}"] = macd.astype("float32")
        frame[f"macd_signal_{suffix}"] = macd_signal.astype("float32")
        frame[f"macd_hist_{suffix}"] = (macd - macd_signal).astype("float32")
        frame[f"bullish_cross_{suffix}"] = (
            (macd > macd_signal)
            & (macd.shift(1) <= macd_signal.shift(1))
        )
        frame[f"bearish_cross_{suffix}"] = (
            (macd < macd_signal)
            & (macd.shift(1) >= macd_signal.shift(1))
        )

    return frame


_PRODUCT_FRAME_CACHE_MAX = 12
_RETURN_SERIES_CACHE_MAX = 30


def _evict_cache(d: dict, max_size: int) -> None:
    """Remove oldest entries when the dict exceeds max_size."""
    while len(d) > max_size:
        d.pop(next(iter(d)))


def build_visualizer_cache(prices: pd.DataFrame, trades: pd.DataFrame, products: List[str]) -> Dict[str, object]:
    """Small lazy caches; heavy per-product features are built on first use."""
    return {
        "version": _code_cache_version(),
        "summaries": {},
        "product_frames": {},
        "product_trades": {},
        "return_series": {},
        "corr_matrices": {},
    }


def dark_table(summary: pd.DataFrame, selected_products: List[str] | None = None):
    selected = set(selected_products or [])
    by_product = summary.set_index("product")
    rows = []
    for category, category_products in ROUND5_GROUPS.items():
        product_cells = []
        for product in category_products:
            if product not in by_product.index:
                product_cells.append(html.Div("", style={"minHeight": "64px"}))
                continue
            row = by_product.loc[product]
            ret = float(row["return_bps"])
            is_selected = product in selected
            product_cells.append(html.Div(
                style={
                    "background": C["panel2"] if is_selected else "#0d1420",
                    "border": f"1px solid {C['accent'] if is_selected else C['border']}",
                    "borderRadius": "6px",
                    "padding": "7px 8px",
                    "minHeight": "64px",
                    "boxShadow": "0 0 0 1px rgba(56,189,248,.25)" if is_selected else "none",
                },
                children=[
                    html.Div(product, style={
                        "fontSize": "9px",
                        "fontWeight": 700,
                        "color": C["text"],
                        "lineHeight": "1.15",
                        "wordBreak": "break-word",
                    }),
                    html.Div(
                        f"ret {ret:+.1f} bps | vol {float(row['vol_bps']):.2f} | spr {float(row['avg_spread']):.2f}",
                        style={"fontSize": "9px", "color": C["good"] if ret >= 0 else C["bad"], "marginTop": "5px"},
                    ),
                    html.Div(
                        f"depth {float(row['avg_book_depth']):.1f} | imb {float(row['avg_abs_imb']):.3f} | score {float(row['efficiency_score']):.1f}",
                        style={"fontSize": "9px", "color": C["muted"], "marginTop": "3px"},
                    ),
                ],
            ))

        rows.append(html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "190px repeat(5, minmax(150px, 1fr))",
                "gap": "8px",
                "alignItems": "stretch",
                "marginBottom": "8px",
            },
            children=[
                html.Div(
                    category,
                    style={
                        "background": "#0d1420",
                        "border": f"1px solid {C['border']}",
                        "borderRadius": "6px",
                        "padding": "8px",
                        "fontSize": "10px",
                        "fontWeight": 800,
                        "color": C["accent"],
                        "display": "flex",
                        "alignItems": "center",
                    },
                ),
                *product_cells,
            ],
        ))
    return html.Div(rows, style={"overflowX": "auto"})


def create_app(prices: pd.DataFrame, trades: pd.DataFrame) -> Dash:
    products = _ordered_products(prices["product"].unique())
    categories = [g for g in ROUND5_GROUPS if any(p in products for p in ROUND5_GROUPS[g])]
    days = sorted(prices["day"].unique())
    day_options = [{"label": f"Day {int(d)}", "value": int(d)} for d in days]
    first_category = categories[0] if categories else "Other"
    first_products = [p for p in ROUND5_GROUPS.get(first_category, products) if p in products]
    default_product = first_products[0] if first_products else products[0]

    app = Dash(__name__, title="Round 5 Market Explorer", suppress_callback_exceptions=True)
    print("Initializing lazy in-memory visualizer cache...")
    cache = build_visualizer_cache(prices, trades, products)
    print(f"  Cache initialized  |  version={cache['version']}")
    cache_holder = {"cache": cache}

    def get_cache() -> Dict[str, object]:
        active_cache = cache_holder["cache"]
        version = _code_cache_version()
        if active_cache["version"] != version:
            print("Source/style changed; resetting lazy visualizer cache...")
            active_cache = build_visualizer_cache(prices, trades, products)
            cache_holder["cache"] = active_cache
            print(f"  Cache initialized  |  version={active_cache['version']}")
        return active_cache

    app.layout = html.Div(
        style={
            "background": C["bg"],
            "color": C["text"],
            "minHeight": "100vh",
            "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            "padding": "18px",
        },
        children=[
            html.Div(
                [
                    html.Div("Prosperity Round 5", style={"fontSize": "9px", "color": C["accent"], "fontWeight": 700}),
                    html.H1("Market Data Explorer", style={"margin": "0", "fontSize": "15px", "lineHeight": "1.05"}),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "18px",
                    "right": "18px",
                    "zIndex": 1000,
                    "background": C["bg"],
                    "borderBottom": f"1px solid {C['border']}",
                    "boxShadow": "0 8px 18px rgba(0,0,0,.24)",
                    "padding": "5px 0",
                    "maxWidth": "100%",
                },
                children=[
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "minmax(120px, 1fr) 64px 68px 66px 72px 82px minmax(105px, .8fr) minmax(160px, 1.3fr)",
                            "gap": "5px",
                            "alignItems": "end",
                            "width": "100%",
                            "maxWidth": "100%",
                        },
                        children=[
                            html.Div([
                                html.Label("Product", style={"fontSize": "7px", "color": C["muted"]}),
                                dcc.Dropdown(
                                    id="product",
                                    value=default_product,
                                    clearable=False,
                                    className="dark-dropdown",
                                    style={"width": "100%"},
                                ),
                            ], style={"minWidth": 0}),
                            html.Div([
                                html.Label("Day", style={"fontSize": "7px", "color": C["muted"]}),
                                dcc.Dropdown(
                                    id="day",
                                    options=day_options,
                                    value=int(days[0]),
                                    clearable=False,
                                    className="dark-dropdown",
                                    style={"width": "100%"},
                                ),
                            ], style={"minWidth": 0}),
                            html.Div([
                                html.Label("Scope", style={"fontSize": "7px", "color": C["muted"]}),
                                dcc.Checklist(
                                    id="show-all-days",
                                    options=[{"label": "All", "value": "all"}],
                                    value=["all"],
                                    inputStyle={"marginRight": "6px"},
                                    labelStyle={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "height": "24px",
                                        "fontSize": "9px",
                                        "color": C["text"],
                                        "whiteSpace": "nowrap",
                                    },
                                    style={
                                        "background": C["panel"],
                                        "border": f"1px solid {C['border']}",
                                        "borderRadius": "6px",
                                        "padding": "0 5px",
                                        "height": "24px",
                                        "overflow": "hidden",
                                    },
                                ),
                            ], style={"minWidth": 0}),
                            html.Div([
                                html.Label("Rolling", style={"fontSize": "7px", "color": C["muted"]}),
                                dcc.Dropdown(
                                    id="rolling",
                                    options=[
                                        {"label": "20", "value": 20},
                                        {"label": "50", "value": 50},
                                        {"label": "100", "value": 100},
                                    ],
                                    value=50,
                                    clearable=False,
                                    className="dark-dropdown",
                                    style={"width": "100%"},
                                ),
                            ], style={"minWidth": 0}),
                            html.Div([
                                html.Label("Band σ", style={"fontSize": "7px", "color": C["muted"]}),
                                dcc.Dropdown(
                                    id="bb-sigma",
                                    options=[
                                        {"label": "1.5σ", "value": 1.5},
                                        {"label": "2.0σ", "value": 2.0},
                                        {"label": "2.5σ", "value": 2.5},
                                        {"label": "3.0σ", "value": 3.0},
                                    ],
                                    value=2.0,
                                    clearable=False,
                                    className="dark-dropdown",
                                    style={"width": "100%"},
                                ),
                            ], style={"minWidth": 0}),
                            html.Div([
                                html.Label("MACD", style={"fontSize": "7px", "color": C["muted"]}),
                                dcc.Dropdown(
                                    id="macd-preset",
                                    options=[
                                        {"label": "8 / 21 / 5", "value": "8,21,5"},
                                        {"label": "12 / 26 / 9", "value": "12,26,9"},
                                        {"label": "20 / 50 / 9", "value": "20,50,9"},
                                    ],
                                    value="12,26,9",
                                    clearable=False,
                                    className="dark-dropdown",
                                    style={"width": "100%"},
                                ),
                            ], style={"minWidth": 0}),
                            html.Div([
                                html.Label("Category", style={"fontSize": "7px", "color": C["muted"]}),
                                dcc.Dropdown(
                                    id="category",
                                    options=[{"label": c, "value": c} for c in categories],
                                    value=first_category,
                                    clearable=False,
                                    className="dark-dropdown",
                                    style={"width": "100%"},
                                ),
                            ], style={"minWidth": 0}),
                            html.Div([
                                html.Label("Analysis", style={"fontSize": "7px", "color": C["muted"]}),
                                dcc.Dropdown(
                                    id="compare-products",
                                    options=[{"label": p, "value": p} for p in products],
                                    multi=True,
                                    value=first_products[:5],
                                    placeholder="Search and select any products",
                                    className="dark-dropdown",
                                    style={"width": "100%"},
                                ),
                            ], style={"minWidth": 0}),
                        ],
                    ),
                ],
            ),
            html.Div(style={"height": "42px"}),
            html.Div(id="stats", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(84px, 1fr))",
                "gap": "5px",
                "marginBottom": "8px",
                "maxWidth": "100%",
                "overflow": "hidden",
            }),
            html.Div(
                style={"display": "flex", "flexDirection": "column", "gap": "12px", "marginBottom": "12px"},
                children=[
                    panel(dcc.Graph(id="price-chart", config={"displaylogo": False}, style={"height": "390px"})),
                    panel(dcc.Graph(id="spread-depth-chart", config={"displaylogo": False}, style={"height": "390px"})),
                    panel(dcc.Graph(id="book-price-chart", config={"displaylogo": False}, style={"height": "300px"})),
                    panel(dcc.Graph(id="book-vol-chart", config={"displaylogo": False}, style={"height": "300px"})),
                    panel(dcc.Graph(id="book-depth-chart", config={"displaylogo": False}, style={"height": "330px"})),
                    panel(dcc.Graph(id="book-imbalance-chart", config={"displaylogo": False}, style={"height": "300px"})),
                    panel(dcc.Graph(id="rolling-chart", config={"displaylogo": False}, style={"height": "370px"})),
                    panel(dcc.Graph(id="macd-chart", config={"displaylogo": False}, style={"height": "360px"})),
                ],
            ),
            html.Div(
                style={"display": "flex", "flexDirection": "column", "gap": "12px", "marginBottom": "12px"},
                children=[
                    panel(dcc.Graph(
                        id="compare-chart",
                        config={"displaylogo": False},
                        style={"height": "52vh", "minHeight": "420px", "maxHeight": "620px"},
                    )),
                    panel(dcc.Graph(id="category-chart", config={"displaylogo": False}, style={"height": "370px"})),
                    panel(dcc.Graph(id="market-map-chart", config={"displaylogo": False}, style={"height": "390px"})),
                    panel(dcc.Graph(id="correlation-chart", config={"displaylogo": False}, style={"height": "680px"})),
                ],
            ),
            panel([
                html.Div("Product Screener - Category Order", style={"fontSize": "10px", "color": C["muted"], "marginBottom": "8px"}),
                html.Div(id="screener"),
            ], style={"marginBottom": "12px"}),
        ],
    )

    app.index_string = f"""<!DOCTYPE html><html><head>
    {{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
    <style>{DROPDOWN_CSS}</style></head><body>
    {{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
    </body></html>"""

    @app.callback(
        [Output("product", "options"), Output("product", "value")],
        [Input("category", "value")],
    )
    def update_product_options(category: str):
        category_products = [p for p in ROUND5_GROUPS.get(category, products) if p in products]
        if not category_products:
            category_products = products
        opts = [{"label": p, "value": p} for p in category_products]
        product_value = category_products[0]
        return opts, product_value

    @app.callback(
        [
            Output("stats", "children"),
            Output("screener", "children"),
            Output("price-chart", "figure"),
            Output("spread-depth-chart", "figure"),
            Output("book-price-chart", "figure"),
            Output("book-vol-chart", "figure"),
            Output("book-depth-chart", "figure"),
            Output("book-imbalance-chart", "figure"),
            Output("rolling-chart", "figure"),
            Output("macd-chart", "figure"),
            Output("compare-chart", "figure"),
            Output("category-chart", "figure"),
            Output("market-map-chart", "figure"),
            Output("correlation-chart", "figure"),
        ],
        [
            Input("product", "value"),
            Input("day", "value"),
            Input("show-all-days", "value"),
            Input("compare-products", "value"),
            Input("rolling", "value"),
            Input("bb-sigma", "value"),
            Input("macd-preset", "value"),
        ],
    )
    def update(product: str, day: int, show_all_days: List[str], compare_products: List[str], rolling: int, bb_sigma: float, macd_preset: str):
        active_cache = get_cache()
        all_days = "all" in (show_all_days or [])
        scope_key: str | int = "all" if all_days else int(day)
        x_col = "continuous_timestamp" if all_days else "timestamp"
        x_title = "Continuous Timestamp" if all_days else "Timestamp"
        day_label = "All days" if all_days else f"Day {int(day)}"
        day_prices = prices if all_days else prices[prices["day"] == int(day)]
        day_trades = trades if all_days else (trades[trades["day"] == int(day)] if not trades.empty else trades)

        product_cache_key = (scope_key, product)
        product_base = active_cache["product_frames"].get(product_cache_key)
        if product_base is None:
            product_prices = day_prices[day_prices["product"] == product].copy()
            product_base = _add_cached_features(product_prices) if not product_prices.empty else product_prices
            active_cache["product_frames"][product_cache_key] = product_base
            _evict_cache(active_cache["product_frames"], _PRODUCT_FRAME_CACHE_MAX)
            _evict_cache(active_cache["product_trades"], _PRODUCT_FRAME_CACHE_MAX)
        if product_base is None or product_base.empty:
            empty = empty_fig("No price data")
            return [], html.Div(), empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty, empty

        category = _category(product)
        category_products = [p for p in ROUND5_GROUPS.get(category, []) if p in products]
        selected_products = _ordered_products([p for p in (compare_products or []) if p in products])
        if not selected_products:
            selected_products = category_products[:5]
        if product not in selected_products:
            selected_products = _ordered_products([product] + selected_products)
        product_trades = active_cache["product_trades"].get(product_cache_key)
        if product_trades is None:
            raw_trades = day_trades[day_trades["symbol"] == product].copy() if not day_trades.empty else pd.DataFrame()
            if raw_trades.empty:
                product_trades = raw_trades
            else:
                book = product_base[["day", "timestamp", "bid_price_1", "ask_price_1", "mid_price"]]
                product_trades = raw_trades.merge(book, on=["day", "timestamp"], how="left")
                product_trades["bot_class"] = np.select(
                    [
                        product_trades["price"] >= product_trades["ask_price_1"],
                        product_trades["price"] <= product_trades["bid_price_1"],
                        product_trades["price"] > product_trades["mid_price"],
                        product_trades["price"] < product_trades["mid_price"],
                    ],
                    [
                        "Bot buy at ask",
                        "Bot sell at bid",
                        "Bot buy inside",
                        "Bot sell inside",
                    ],
                    default="Inside / mid",
                )
                product_trades = product_trades.sort_values(["day", "timestamp"]).reset_index(drop=True)
                product_trades["trade_vwap"] = (
                    (product_trades["price"] * product_trades["quantity"]).cumsum()
                    / product_trades["quantity"].cumsum()
                ).astype("float32")
            active_cache["product_trades"][product_cache_key] = product_trades
        summary = active_cache["summaries"].get(scope_key)
        if summary is None:
            summary = day_summary(day_prices, day_trades)
            active_cache["summaries"][scope_key] = summary
        selected_row = summary[summary["product"] == product].iloc[0]

        window = int(rolling or 50)
        sigma = float(bb_sigma or 2.0)
        if window not in ROLLING_WINDOWS:
            window = 50
        macd_key = _macd_key(macd_preset)
        macd_fast, macd_slow, macd_signal_span = MACD_PRESETS[macd_key]
        macd_suffix = macd_key.replace(",", "_")

        product_df = _ds_df(product_base.copy(deep=False))
        product_df["mid_ma"] = product_df[f"mid_ma_{window}"]
        product_df["mid_std"] = product_df[f"mid_std_{window}"]
        product_df["bb_upper"] = product_df["mid_ma"] + sigma * product_df["mid_std"]
        product_df["bb_lower"] = product_df["mid_ma"] - sigma * product_df["mid_std"]
        product_df["spread_ma"] = product_df[f"spread_ma_{window}"]
        product_df["spread_price_corr"] = product_df[f"spread_price_corr_{window}"]
        product_df["imbalance_ma"] = product_df[f"imbalance_ma_{window}"]
        product_df["book_depth_ma"] = product_df[f"book_depth_ma_{window}"]
        product_df["book_vol_bps"] = product_df[f"book_vol_bps_{window}"]
        product_df["zscore"] = product_df[f"zscore_{window}"]
        product_df["spread_ma_zscore"] = product_df[f"spread_ma_zscore_{window}"]
        product_df["macd"] = product_df[f"macd_{macd_suffix}"]
        product_df["macd_signal"] = product_df[f"macd_signal_{macd_suffix}"]
        product_df["macd_hist"] = product_df[f"macd_hist_{macd_suffix}"]
        product_df["bullish_cross"] = product_df[f"bullish_cross_{macd_suffix}"]
        product_df["bearish_cross"] = product_df[f"bearish_cross_{macd_suffix}"]
        spread_price_corr = product_df["spread_ma"].corr(product_df["mid_price"])
        if pd.isna(spread_price_corr):
            spread_price_corr = 0.0
        latest_macd = product_df[["macd", "macd_signal"]].dropna().tail(1)
        if latest_macd.empty:
            macd_state = "Warming"
            macd_color = C["muted"]
        else:
            macd_state = "Bullish" if latest_macd["macd"].iloc[0] >= latest_macd["macd_signal"].iloc[0] else "Bearish"
            macd_color = C["good"] if macd_state == "Bullish" else C["bad"]
        bot_buy_count = int(product_trades["bot_class"].str.contains("buy", case=False).sum()) if not product_trades.empty else 0
        bot_sell_count = int(product_trades["bot_class"].str.contains("sell", case=False).sum()) if not product_trades.empty else 0

        stats = [
            stat("Product", product, C["accent"]),
            stat("Day Scope", day_label, C["accent2"]),
            stat("Category", category),
            stat("Analysis Basket", f"{len(selected_products)} products", C["accent2"]),
            stat("Mid Start", f"{product_df['mid_price'].iloc[0]:,.1f}"),
            stat("Mid End", f"{product_df['mid_price'].iloc[-1]:,.1f}", C["good"] if product_df["mid_price"].iloc[-1] >= product_df["mid_price"].iloc[0] else C["bad"]),
            stat("Avg Spread", f"{product_df['spread'].mean():.2f}"),
            stat("Avg Book Depth", f"{product_df['book_depth'].mean():.1f}"),
            stat("Imbalance MA", f"{product_df['imbalance_ma'].iloc[-1]:+.3f}", C["good"] if product_df["imbalance_ma"].iloc[-1] >= 0 else C["bad"]),
            stat("Trades", f"{len(product_trades):,}"),
            stat("Bot Buy / Sell", f"{bot_buy_count} / {bot_sell_count}"),
            stat("Vol bps", f"{float(selected_row['vol_bps']):.2f}"),
            stat("Range bps", f"{float(selected_row['range_bps']):.1f}"),
            stat("Spread/Price Corr", f"{spread_price_corr:.3f}", C["good"] if spread_price_corr >= 0 else C["bad"]),
            stat("MACD State", macd_state, macd_color),
        ]
        screener = dark_table(summary, selected_products)

        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["bb_lower"],
            mode="lines",
            name=f"lower band {sigma:g}σ",
            line=dict(color="rgba(148,163,184,0.35)", width=1),
            hovertemplate="timestamp=%{x}<br>lower band %{y:.2f}<extra></extra>",
        ))
        price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["bb_upper"],
            mode="lines",
            name=f"upper band {sigma:g}σ",
            line=dict(color="rgba(148,163,184,0.35)", width=1),
            fill="tonexty",
            fillcolor="rgba(148,163,184,0.10)",
            hovertemplate="timestamp=%{x}<br>upper band %{y:.2f}<extra></extra>",
        ))
        price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["mid_price"],
            mode="lines",
            name="mid",
            line=dict(color=C["accent"], width=2.2),
        ))
        price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["mid_ma"],
            mode="lines",
            name=f"SMA {window}",
            line=dict(color=C["accent2"], width=1.8),
        ))
        price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["microprice"],
            mode="lines",
            name="microprice",
            line=dict(color="#facc15", width=1.2),
            opacity=0.75,
        ))
        price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["bid_price_1"],
            mode="lines",
            name="best bid",
            line=dict(color=C["good"], width=1),
            opacity=0.65,
        ))
        price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["ask_price_1"],
            mode="lines",
            name="best ask",
            line=dict(color=C["bad"], width=1),
            opacity=0.65,
        ))
        if not product_trades.empty:
            price_fig.add_trace(go.Scatter(
                x=product_trades[x_col],
                y=product_trades["trade_vwap"],
                mode="lines",
                name="trade VWAP",
                line=dict(color="#f97316", width=1.6, dash="dash"),
                opacity=0.9,
                hovertemplate="timestamp=%{x}<br>VWAP %{y:.2f}<extra></extra>",
            ))
            for trade_class, color, symbol in [
                ("Bot buy at ask", C["good"], "arrow-up"),
                ("Bot buy inside", "#86efac", "arrow-up-open"),
                ("Bot sell at bid", C["bad"], "arrow-down"),
                ("Bot sell inside", "#fca5a5", "arrow-down-open"),
                ("Inside / mid", C["muted"], "diamond"),
            ]:
                classified = product_trades[product_trades["bot_class"] == trade_class]
                if classified.empty:
                    continue
                price_fig.add_trace(go.Scatter(
                    x=classified[x_col],
                    y=classified["price"],
                    mode="markers",
                    name=trade_class,
                    marker=dict(
                        size=np.clip(classified["quantity"] * 3.0, 8, 16),
                        color=color,
                        symbol=symbol,
                        opacity=0.78,
                        line=dict(color="#ffffff", width=0.7),
                    ),
                    customdata=np.stack([classified["quantity"], classified["bid_price_1"], classified["ask_price_1"], classified["mid_price"]], axis=-1),
                    hovertemplate=(
                        "timestamp=%{x}<br>"
                        + trade_class
                        + "<br>price=%{y:.1f}<br>qty=%{customdata[0]}"
                        + "<br>bid=%{customdata[1]:.1f}<br>ask=%{customdata[2]:.1f}<br>mid=%{customdata[3]:.1f}<extra></extra>"
                    ),
                ))
        bullish_df = product_df[product_df["bullish_cross"]].copy()
        bearish_df = product_df[product_df["bearish_cross"]].copy()
        theme(price_fig, f"{product} price, SMA, Bollinger bands, and bot trades")
        price_fig.update_xaxes(title=x_title)
        price_fig.update_yaxes(title="Price")
        add_day_boundaries(price_fig, product_df, x_col, "all" if all_days else day)

        sd_fig = go.Figure()
        sd_fig.add_trace(go.Scatter(
            x=product_df[x_col], y=product_df["spread"], mode="lines", name="spread",
            line=dict(color=C["accent"], width=1.2), opacity=0.45,
        ))
        sd_fig.add_trace(go.Scatter(
            x=product_df[x_col], y=product_df["spread_ma"], mode="lines", name=f"spread MA {window}",
            line=dict(color=C["accent2"], width=2.2),
        ))
        sd_fig.add_trace(go.Scatter(
            x=product_df[x_col], y=product_df["mid_price"], mode="lines", name="mid price",
            line=dict(color="#a78bfa", width=1.7), yaxis="y2",
        ))
        theme(sd_fig, f"{product} spread MA vs mid price")
        sd_fig.update_xaxes(title=x_title)
        sd_fig.update_yaxes(title="Spread")
        sd_fig.update_layout(yaxis2=dict(title="Mid price", overlaying="y", side="right", showgrid=False, zeroline=False))
        add_day_boundaries(sd_fig, product_df, x_col, "all" if all_days else day)

        book_price_fig = go.Figure()
        book_price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["mid_price"],
            mode="lines",
            name="mid price",
            line=dict(color=C["accent"], width=1.9),
            hovertemplate="timestamp=%{x}<br>mid=%{y:.2f}<extra></extra>",
        ))
        book_price_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["weighted_mid_price"],
            mode="lines",
            name="weighted mid price",
            line=dict(color=C["accent2"], width=1.8),
            hovertemplate="timestamp=%{x}<br>weighted mid=%{y:.2f}<extra></extra>",
        ))
        theme(book_price_fig, f"{product} mid price vs weighted mid price")
        book_price_fig.update_xaxes(title=x_title)
        book_price_fig.update_yaxes(title="Price")
        add_day_boundaries(book_price_fig, product_df, x_col, "all" if all_days else day)

        book_vol_fig = go.Figure()
        book_vol_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["spread"],
            mode="lines",
            name="spread",
            line=dict(color=C["accent"], width=1.3),
            opacity=0.6,
            hovertemplate="timestamp=%{x}<br>spread=%{y:.2f}<extra></extra>",
        ))
        book_vol_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["book_vol_bps"],
            mode="lines",
            name=f"book volatility {window}",
            line=dict(color=C["bad"], width=1.7),
            yaxis="y2",
            hovertemplate="timestamp=%{x}<br>vol=%{y:.3f} bps<extra></extra>",
        ))
        theme(book_vol_fig, f"{product} book volatility and spread")
        book_vol_fig.update_xaxes(title=x_title)
        book_vol_fig.update_yaxes(title="Spread")
        book_vol_fig.update_layout(yaxis2=dict(title="Vol bps", overlaying="y", side="right", showgrid=False, zeroline=False))
        add_day_boundaries(book_vol_fig, product_df, x_col, "all" if all_days else day)

        book_depth_fig = go.Figure()
        book_depth_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["bid_total_vol"],
            mode="lines",
            name="bid depth",
            line=dict(color=C["good"], width=1.4),
            hovertemplate="timestamp=%{x}<br>bid depth=%{y:.0f}<extra></extra>",
        ))
        book_depth_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["ask_total_vol"],
            mode="lines",
            name="ask depth",
            line=dict(color=C["bad"], width=1.4),
            hovertemplate="timestamp=%{x}<br>ask depth=%{y:.0f}<extra></extra>",
        ))
        book_depth_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["book_depth_ma"],
            mode="lines",
            name=f"total depth MA {window}",
            line=dict(color=C["accent2"], width=2),
            hovertemplate="timestamp=%{x}<br>depth MA=%{y:.1f}<extra></extra>",
        ))
        book_depth_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["bid_level_count"],
            mode="lines",
            name="bid levels 1-3",
            line=dict(color="#86efac", width=1, dash="dot"),
            yaxis="y2",
            hovertemplate="timestamp=%{x}<br>bid levels=%{y:.0f}<extra></extra>",
        ))
        book_depth_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["ask_level_count"],
            mode="lines",
            name="ask levels 1-3",
            line=dict(color="#fca5a5", width=1, dash="dot"),
            yaxis="y2",
            hovertemplate="timestamp=%{x}<br>ask levels=%{y:.0f}<extra></extra>",
        ))
        theme(book_depth_fig, f"{product} book depth and active levels")
        book_depth_fig.update_xaxes(title=x_title)
        book_depth_fig.update_yaxes(title="Volume")
        book_depth_fig.update_layout(yaxis2=dict(title="Levels", overlaying="y", side="right", range=[0, 3.2], showgrid=False, zeroline=False))
        add_day_boundaries(book_depth_fig, product_df, x_col, "all" if all_days else day)

        book_imbalance_fig = go.Figure()
        imbalance_colors = np.where(product_df["imbalance"] >= 0, C["good"], C["bad"])
        book_imbalance_fig.add_trace(go.Bar(
            x=product_df[x_col],
            y=product_df["imbalance"],
            name="book imbalance",
            marker_color=imbalance_colors,
            opacity=0.35,
            hovertemplate="timestamp=%{x}<br>imbalance=%{y:+.3f}<extra></extra>",
        ))
        book_imbalance_fig.add_trace(go.Scatter(
            x=product_df[x_col],
            y=product_df["imbalance_ma"],
            mode="lines",
            name=f"imbalance MA {window}",
            line=dict(color=C["accent2"], width=2.2),
            hovertemplate="timestamp=%{x}<br>imbalance MA=%{y:+.3f}<extra></extra>",
        ))
        book_imbalance_fig.add_hline(y=0, line=dict(color=C["grid"], dash="dot", width=1))
        theme(book_imbalance_fig, f"{product} order book imbalance signal")
        book_imbalance_fig.update_xaxes(title=x_title)
        book_imbalance_fig.update_yaxes(title="Imbalance", range=[-1, 1])
        add_day_boundaries(book_imbalance_fig, product_df, x_col, "all" if all_days else day)

        rolling_fig = go.Figure()
        rolling_fig.add_trace(go.Scatter(
            x=product_df[x_col], y=product_df["zscore"], mode="lines",
            name=f"mid z-score {window}", line=dict(color=C["accent"], width=1.8),
        ))
        rolling_fig.add_trace(go.Scatter(
            x=product_df[x_col], y=product_df["spread_ma_zscore"], mode="lines",
            name=f"spread MA z-score {window}", line=dict(color=C["accent2"], width=1.6),
        ))
        rolling_fig.add_trace(go.Scatter(
            x=product_df[x_col], y=product_df["spread_price_corr"], mode="lines",
            name=f"rolling spread/price corr {window}", line=dict(color="#a78bfa", width=1.9), yaxis="y2",
        ))
        rolling_fig.add_hline(y=2, line=dict(color=C["bad"], dash="dot", width=1))
        rolling_fig.add_hline(y=-2, line=dict(color=C["good"], dash="dot", width=1))
        theme(rolling_fig, f"{product} rolling signal and spread/price correlation")
        rolling_fig.update_xaxes(title=x_title)
        rolling_fig.update_yaxes(title="Z-score")
        rolling_fig.update_layout(yaxis2=dict(title="Correlation", overlaying="y", side="right", range=[-1, 1], showgrid=False, zeroline=False))
        add_day_boundaries(rolling_fig, product_df, x_col, "all" if all_days else day)

        macd_fig = go.Figure()
        macd_df = product_df.dropna(subset=["macd", "macd_signal"]).copy()
        if not macd_df.empty:
            hist_colors = np.where(macd_df["macd_hist"] >= 0, C["good"], C["bad"])
            macd_fig.add_trace(go.Bar(
                x=macd_df[x_col],
                y=macd_df["macd_hist"],
                name="MACD histogram",
                marker_color=hist_colors,
                opacity=0.42,
                hovertemplate="timestamp=%{x}<br>hist=%{y:.4f}<extra></extra>",
            ))
            macd_fig.add_trace(go.Scatter(
                x=macd_df[x_col],
                y=macd_df["macd"],
                mode="lines",
                name=f"MACD {macd_fast}-{macd_slow}",
                line=dict(color=C["accent"], width=2),
                hovertemplate="timestamp=%{x}<br>MACD=%{y:.4f}<extra></extra>",
            ))
            macd_fig.add_trace(go.Scatter(
                x=macd_df[x_col],
                y=macd_df["macd_signal"],
                mode="lines",
                name=f"signal {macd_signal_span}",
                line=dict(color=C["accent2"], width=2),
                hovertemplate="timestamp=%{x}<br>signal=%{y:.4f}<extra></extra>",
            ))
        if not bullish_df.empty:
            macd_fig.add_trace(go.Scatter(
                x=bullish_df[x_col],
                y=bullish_df["macd"],
                mode="markers",
                name="bullish signal",
                marker=dict(symbol="triangle-up", size=11, color=C["good"], line=dict(color="#ffffff", width=1)),
                customdata=np.stack([bullish_df["mid_price"], bullish_df["macd_signal"]], axis=-1),
                hovertemplate="timestamp=%{x}<br>bullish signal<br>MACD=%{y:.4f}<br>signal=%{customdata[1]:.4f}<br>mid=%{customdata[0]:.2f}<extra></extra>",
            ))
        if not bearish_df.empty:
            macd_fig.add_trace(go.Scatter(
                x=bearish_df[x_col],
                y=bearish_df["macd"],
                mode="markers",
                name="bearish signal",
                marker=dict(symbol="triangle-down", size=11, color=C["bad"], line=dict(color="#ffffff", width=1)),
                customdata=np.stack([bearish_df["mid_price"], bearish_df["macd_signal"]], axis=-1),
                hovertemplate="timestamp=%{x}<br>bearish signal<br>MACD=%{y:.4f}<br>signal=%{customdata[1]:.4f}<br>mid=%{customdata[0]:.2f}<extra></extra>",
            ))
        macd_fig.add_hline(y=0, line=dict(color=C["grid"], dash="dot", width=1))
        theme(macd_fig, f"{product} MACD crossover signals")
        macd_fig.update_xaxes(title=x_title)
        macd_fig.update_yaxes(title="MACD / histogram")
        add_day_boundaries(macd_fig, product_df, x_col, "all" if all_days else day)

        compare_fig = go.Figure()
        for item in selected_products:
            series_key = (scope_key, item)
            comp = active_cache["return_series"].get(series_key)
            if comp is None:
                item_prices = day_prices[day_prices["product"] == item].sort_values(["day", "timestamp"])
                if item_prices.empty:
                    comp = pd.DataFrame()
                else:
                    base = item_prices["mid_price"].iloc[0]
                    norm = (item_prices["mid_price"] / base - 1.0) * 10_000 if base else item_prices["mid_price"] * 0
                    comp = pd.DataFrame({
                        "timestamp": item_prices["timestamp"].to_numpy(),
                        "continuous_timestamp": item_prices["continuous_timestamp"].to_numpy(),
                        "return_bps": norm.astype("float32").to_numpy(),
                    })
                active_cache["return_series"][series_key] = comp
                _evict_cache(active_cache["return_series"], _RETURN_SERIES_CACHE_MAX)
            if comp.empty:
                continue
            comp_plot = _ds_df(comp)
            compare_fig.add_trace(go.Scatter(x=comp_plot[x_col], y=comp_plot["return_bps"], mode="lines", name=item, line=dict(width=1.7)))
        theme(compare_fig, "Selected products normalized mid-price comparison")
        compare_fig.update_layout(
            autosize=True,
            legend=dict(
                orientation="v",
                x=1.01,
                y=1,
                xanchor="left",
                yanchor="top",
                font=dict(size=7),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=44, r=285, t=50, b=62),
        )
        compare_fig.update_xaxes(title=x_title)
        compare_fig.update_yaxes(title="Return from open (bps)")
        add_day_boundaries(compare_fig, day_prices, x_col, "all" if all_days else day)

        cat_fig = go.Figure()
        selected_summary = summary[summary["product"].isin(selected_products)].copy()
        if not selected_summary.empty:
            cat_fig.add_trace(go.Bar(
                x=selected_summary["product"],
                y=selected_summary["return_bps"],
                marker_color=np.where(selected_summary["return_bps"] >= 0, C["good"], C["bad"]),
                customdata=np.stack([selected_summary["category"], selected_summary["avg_spread"], selected_summary["n_trades"], selected_summary["trade_qty"]], axis=-1),
                hovertemplate="%{x}<br>%{customdata[0]}<br>return=%{y:.1f} bps<br>avg spread=%{customdata[1]:.2f}<br>trades=%{customdata[2]:.0f}<br>trade qty=%{customdata[3]:.0f}<extra></extra>",
            ))
        theme(cat_fig, "Selected products cross-section")
        cat_fig.update_xaxes(title="", tickangle=25)
        cat_fig.update_yaxes(title="Return from open (bps)")

        map_fig = go.Figure()
        map_summary = summary.copy()
        map_summary["is_selected"] = map_summary["product"].isin(selected_products)
        map_fig.add_trace(go.Scatter(
            x=map_summary["avg_spread"],
            y=map_summary["vol_bps"],
            mode="markers",
            text=map_summary["product"],
            marker=dict(
                size=np.clip(map_summary["trade_qty"].astype(float) / max(1.0, map_summary["trade_qty"].max()) * 28 + 6, 6, 34),
                color=map_summary["return_bps"],
                colorscale="RdYlGn",
                colorbar=dict(title="ret bps"),
                line=dict(color=np.where(map_summary["is_selected"], C["accent"], C["border"]), width=np.where(map_summary["is_selected"], 2.6, 1)),
                opacity=0.88,
            ),
            customdata=np.stack([map_summary["category"], map_summary["return_bps"], map_summary["n_trades"], map_summary["range_bps"], map_summary["is_selected"]], axis=-1),
            hovertemplate="%{text}<br>%{customdata[0]}<br>selected=%{customdata[4]}<br>spread=%{x:.2f}<br>vol=%{y:.2f} bps<br>return=%{customdata[1]:.1f} bps<br>trades=%{customdata[2]:.0f}<br>range=%{customdata[3]:.1f} bps<extra></extra>",
        ))
        theme(map_fig, "All-product opportunity map with selected basket highlighted")
        map_fig.update_xaxes(title="Average spread")
        map_fig.update_yaxes(title="Realized volatility (bps/tick)")

        corr_products = selected_products
        if len(corr_products) < 2:
            corr_products = category_products
        corr_fig = go.Figure()
        full_corr = active_cache["corr_matrices"].get(scope_key)
        if full_corr is None:
            pivot = day_prices.pivot(index=x_col, columns="product", values="ret").fillna(0.0)
            pivot = pivot.reindex(columns=[p for p in products if p in pivot.columns])
            full_corr = pivot.corr().fillna(0.0) if pivot.shape[1] >= 2 else pd.DataFrame()
            active_cache["corr_matrices"][scope_key] = full_corr
        corr_labels = [p for p in corr_products if p in full_corr.index and p in full_corr.columns]
        if len(corr_labels) >= 2:
            corr = full_corr.loc[corr_labels, corr_labels].fillna(0.0)
            corr_labels = corr.columns.tolist()
            corr_fig.add_trace(go.Heatmap(
                z=corr.values,
                x=corr_labels,
                y=corr_labels,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="corr", len=0.85),
                hovertemplate="%{y} vs %{x}<br>corr=%{z:.3f}<extra></extra>",
            ))
            corr_fig.update_xaxes(
                tickangle=45,
                tickmode="array",
                tickvals=corr_labels,
                ticktext=corr_labels,
                tickfont=dict(size=8),
                automargin=True,
            )
            corr_fig.update_yaxes(
                tickmode="array",
                tickvals=corr_labels,
                ticktext=corr_labels,
                tickfont=dict(size=8),
                automargin=True,
            )
        theme(corr_fig, "Selected products return correlation heatmap")
        corr_fig.update_layout(margin=dict(l=190, r=50, t=50, b=170))

        return (
            stats,
            screener,
            price_fig,
            sd_fig,
            book_price_fig,
            book_vol_fig,
            book_depth_fig,
            book_imbalance_fig,
            rolling_fig,
            macd_fig,
            compare_fig,
            cat_fig,
            map_fig,
            corr_fig,
        )

    return app


def print_diagnostics(prices: pd.DataFrame, trades: pd.DataFrame) -> None:
    print(f"  Price rows: {len(prices):,}")
    print(f"  Trade rows: {len(trades):,}")
    print(f"  Days: {sorted(prices['day'].unique().tolist())}")
    print(f"  Products: {prices['product'].nunique()}")
    print("  Category product counts:")
    for category, count in prices.groupby("category")["product"].nunique().sort_index().items():
        print(f"    {category:<32s} {int(count):>2d}")
    print("  Trade rows by day:")
    if trades.empty:
        print("    none")
    else:
        for day, count in trades.groupby("day").size().items():
            print(f"    Day {int(day)}: {int(count):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prosperity Round 5 market data explorer")
    parser.add_argument("--data", "-d", default=None, help="Dataset directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", "-p", type=int, default=8050)
    parser.add_argument("--days", nargs="+", type=int, default=None, help="Load only these dataset days, e.g. --days 2")
    parser.add_argument("--timestamp-stride", type=int, default=1, help="Load every Nth timestamp group to reduce hosted memory")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--diag", action="store_true")
    args = parser.parse_args()

    data_dir = args.data or find_data_folder()
    if not data_dir:
        raise SystemExit("No data folder found. Pass --data Datasets.")

    if args.host == "0.0.0.0" and args.days is None:
        args.days = [2]
        args.timestamp_stride = max(args.timestamp_stride, 50)
        print("Hosted mode detected: defaulting to --days 2 --timestamp-stride 50 to stay under memory limits.")

    print(f"Loading: {data_dir}")
    prices, trades = load_data(data_dir, args.days, args.timestamp_stride)
    if args.diag:
        print_diagnostics(prices, trades)

    app = create_app(prices, trades)
    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    print(f"\n  http://{args.host}:{args.port}\n")
    app.run(debug=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
