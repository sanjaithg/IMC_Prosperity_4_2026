import json
from typing import Dict, List
from json import JSONEncoder
import jsonpickle

# Type Aliases
Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int

class Listing:
    """Represents a product listed on the exchange."""
    def __init__(self, symbol: Symbol, product: Product, denomination: Product):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination
        
                 
class ConversionObservation:
    """Observation data relevant for conversion requests."""
    def __init__(self, bidPrice: float, askPrice: float, transportFees: float, exportTariff: float, importTariff: float, sunlight: float, humidity: float):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        # These were mentioned in the text but not explicitly in the constructor above
        # adding them for completeness as they appear in the description
        self.sugarPrice = 0.0 # Placeholder if needed
        self.sunlightIndex = 0.0 # Placeholder if needed
        

class Observation:
    """Container for all observations delivered in a TradingState."""
    def __init__(self, plainValueObservations: Dict[Product, ObservationValue], conversionObservations: Dict[Product, ConversionObservation]) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations
        
    def __str__(self) -> str:
        return "(plainValueObservations: " + jsonpickle.encode(self.plainValueObservations) + ", conversionObservations: " + jsonpickle.encode(self.conversionObservations) + ")"
     

class Order:
    """An order to be sent to the exchange."""
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity # Positive for BUY, Negative for SELL

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"
    

class OrderDepth:
    """The collection of all outstanding buy and sell orders for a symbol."""
    def __init__(self):
        self.buy_orders: Dict[int, int] = {} # Price -> Quantity (Positive)
        self.sell_orders: Dict[int, int] = {} # Price -> Quantity (Negative)

class Trade:
    """A trade that has occurred on the exchange."""
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId=None, seller: UserId=None, timestamp: int=0) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer # "SUBMISSION" if you were the buyer
        self.seller = seller # "SUBMISSION" if you were the seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + self.buyer + " << " + self.seller + ", " + str(self.price) + ", " + str(self.quantity) + ", " + str(self.timestamp) + ")"

class TradingState(object):
    """The full state of the market for a single iteration."""
    def __init__(self,
                 traderData: str,
                 timestamp: Time,
                 listings: Dict[Symbol, Listing],
                 order_depths: Dict[Symbol, OrderDepth],
                 own_trades: Dict[Symbol, List[Trade]],
                 market_trades: Dict[Symbol, List[Trade]],
                 position: Dict[Product, Position],
                 observations: Observation):
        self.traderData = traderData # Persisted state from previous iteration
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position # Your current holdings
        self.observations = observations
        
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    
class ProsperityEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__
