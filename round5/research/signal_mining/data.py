from __future__ import annotations

import glob
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def discover_data_dir(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            ROOT / "Datasets",
            ROOT / "Round_5" / "Dataset",
            ROOT / "Dataset",
            ROOT / "Data",
            ROOT / "data",
        ]
    )
    for path in candidates:
        if path.is_dir() and list(path.glob("prices_round_*_day_*.csv")):
            return path.resolve()
    raise FileNotFoundError("No data directory with prices_round_*_day_*.csv found.")


def day_from_path(path: Path) -> int:
    m = re.search(r"day_(-?\d+)\.csv$", path.name)
    if not m:
        raise ValueError(f"Cannot infer day from {path}")
    return int(m.group(1))


def discover_days(data_dir: Path) -> List[int]:
    return sorted({day_from_path(path) for path in data_dir.glob("prices_round_*_day_*.csv")})


def data_file_hashes(data_dir: Path) -> Dict[str, str]:
    hashes = {}
    for path in sorted(data_dir.glob("*round_*_day_*.csv")):
        hashes[path.name] = file_hash(path)
    return hashes


def _detect_delimiter(path: Path) -> str:
    head = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    return ";" if ";" in head else ","


def _skiprows_for_stride(stride: int, products_per_timestamp: int = 50):
    if stride <= 1:
        return None
    return lambda i: i > 0 and ((i - 1) // products_per_timestamp) % stride != 0


def load_prices(data_dir: Path, days: Iterable[int], timestamp_stride: int = 1) -> pd.DataFrame:
    frames = []
    for day in days:
        matches = sorted(data_dir.glob(f"prices_round_*_day_{day}.csv"))
        if not matches:
            continue
        path = matches[-1]
        frame = pd.read_csv(path, sep=_detect_delimiter(path), skiprows=_skiprows_for_stride(timestamp_stride))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No price files loaded for days {list(days)} from {data_dir}")
    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)
    prices["product"] = prices["product"].astype(str)
    prices["day"] = prices["day"].astype(int)
    prices["timestamp"] = prices["timestamp"].astype(int)
    return prices


def load_trades(data_dir: Path, days: Iterable[int]) -> pd.DataFrame:
    frames = []
    for day in days:
        matches = sorted(data_dir.glob(f"trades_round_*_day_{day}.csv"))
        if not matches:
            continue
        path = matches[-1]
        frame = pd.read_csv(path, sep=_detect_delimiter(path))
        if "symbol" in frame.columns and "product" not in frame.columns:
            frame = frame.rename(columns={"symbol": "product"})
        frame["day"] = day
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["day", "timestamp", "product", "price", "quantity"])
    trades = pd.concat(frames, ignore_index=True)
    trades["product"] = trades["product"].astype(str)
    trades["day"] = trades["day"].astype(int)
    trades["timestamp"] = trades["timestamp"].astype(int)
    return trades


def available_products(prices: pd.DataFrame) -> List[str]:
    return sorted(prices["product"].dropna().astype(str).unique())

