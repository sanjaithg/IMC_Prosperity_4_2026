"""
IMC Prosperity — Data Model
Mirrors the classes available on the IMC platform.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Listing:
    symbol: str
    product: str
    denomination: str


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)   # price -> qty (positive)
    sell_orders: Dict[int, int] = field(default_factory=dict)  # price -> qty (negative)


@dataclass
class Trade:
    symbol: str
    price: float
    quantity: int
    buyer: str = ""
    seller: str = ""
    timestamp: int = 0


@dataclass
class Order:
    symbol: str
    price: int
    quantity: int  # positive = buy, negative = sell


@dataclass
class ConversionObservation:
    bidPrice: float = 0
    askPrice: float = 0
    transportFees: float = 0
    exportTariff: float = 0
    importTariff: float = 0
    sunlight: float = 0
    humidity: float = 0


@dataclass
class Observation:
    plainValueObservations: Dict[str, float] = field(default_factory=dict)
    conversionObservations: Dict[str, ConversionObservation] = field(default_factory=dict)


@dataclass
class TradingState:
    traderData: str = ""
    timestamp: int = 0
    listings: Dict[str, Listing] = field(default_factory=dict)
    order_depths: Dict[str, OrderDepth] = field(default_factory=dict)
    own_trades: Dict[str, List[Trade]] = field(default_factory=dict)
    market_trades: Dict[str, List[Trade]] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)
    observations: Optional[Observation] = None
