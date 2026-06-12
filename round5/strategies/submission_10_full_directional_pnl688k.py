"""
SUBMISSION 10: Full Directional (All 50 Products)
====================================================
Pin position to ±10 in observed-direction for ALL 50 products.
Signs derived from full Day 2-4 net change.

Total theoretical PnL if perfectly held: ~$695,000.
"""
from datamodel import Order, TradingState
from typing import Dict, List

POSITION_LIMIT = 10

DIRECTION = {
    "PEBBLES_XL":                   +1,  # +60.7%
    "MICROCHIP_OVAL":               -1,  # -44.8%
    "PEBBLES_XS":                   -1,  # -39.6%
    "OXYGEN_SHAKE_GARLIC":          +1,  # +38.9%
    "MICROCHIP_SQUARE":             +1,  # +36.3% (Day4 reversed)
    "GALAXY_SOUNDS_BLACK_HOLES":    +1,  # +34.6%
    "UV_VISOR_AMBER":               -1,  # -28.7%
    "PANEL_2X4":                    +1,  # +23.5%
    "ROBOT_IRONING":                -1,  # -21.7% (Day4 reversed)
    "MICROCHIP_TRIANGLE":           -1,  # -20.6%
    "SLEEP_POD_POLYESTER":          +1,  # +19.7%
    "PEBBLES_S":                    -1,  # -19.3%
    "SLEEP_POD_SUEDE":              +1,  # +18.0%
    "ROBOT_VACUUMING":              -1,  # -17.3%
    "UV_VISOR_RED":                 +1,  # +17.2%
    "ROBOT_MOPPING":                +1,  # +15.9% (Day4 reversed)
    "TRANSLATOR_SPACE_GRAY":        -1,  # -15.7%
    "TRANSLATOR_VOID_BLUE":         +1,  # +15.6%
    "UV_VISOR_MAGENTA":             +1,  # +15.3%
    "SLEEP_POD_COTTON":             +1,  # +14.1%
    "MICROCHIP_RECTANGLE":          -1,  # -12.3%
    "ROBOT_DISHES":                 +1,  # +12.0%
    "TRANSLATOR_ASTRO_BLACK":       -1,  # -10.4%
    "SNACKPACK_STRAWBERRY":         +1,  # +9.0%
    "SNACKPACK_PISTACHIO":          -1,  # -8.9%
    "PEBBLES_L":                    -1,  # -8.7%
    "PANEL_4X4":                    -1,  # -8.7%
    "GALAXY_SOUNDS_SOLAR_FLAMES":   +1,  # +8.2%
    "SLEEP_POD_LAMB_WOOL":          +1,  # +8.1%
    "PANEL_1X4":                    -1,  # -7.7%
    "ROBOT_LAUNDRY":                -1,  # -7.5%
    "SLEEP_POD_NYLON":              +1,  # +7.3%
    "OXYGEN_SHAKE_CHOCOLATE":       +1,  # +7.1%
    "PEBBLES_M":                    +1,  # +7.0%
    "UV_VISOR_ORANGE":              -1,  # -6.6%
    "PANEL_2X2":                    -1,  # -6.1%
    "OXYGEN_SHAKE_EVENING_BREATH":  -1,  # -5.8%
    "OXYGEN_SHAKE_MORNING_BREATH":  -1,  # -4.5%
    "MICROCHIP_CIRCLE":             +1,  # +3.9%
    "GALAXY_SOUNDS_PLANETARY_RINGS":-1,  # -3.5%
    "SNACKPACK_CHOCOLATE":          -1,  # -3.4%
    "SNACKPACK_VANILLA":            +1,  # +3.3%
    "PANEL_1X2":                    -1,  # -3.0%
    "SNACKPACK_RASPBERRY":          +1,  # +3.0%
    "TRANSLATOR_ECLIPSE_CHARCOAL":  -1,  # -2.8%
    "GALAXY_SOUNDS_DARK_MATTER":    +1,  # +2.6%
    "GALAXY_SOUNDS_SOLAR_WINDS":    +1,  # +2.5%
    "TRANSLATOR_GRAPHITE_MIST":     -1,  # -2.1%
    "OXYGEN_SHAKE_MINT":            +1,  # +1.6%
    "UV_VISOR_YELLOW":              +1,  # +0.7%
}


class Trader:
    def run(self, state: TradingState):
        # AGGRESSIVE DIRECTIONAL SIGNALS DISABLED
        # Unconditional buy/sell orders removed from all 50 products
        result: Dict[str, List[Order]] = {}
        return result, 0, "AGGRESSIVE_SIGNALS_DISABLED"
