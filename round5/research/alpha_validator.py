import pandas as pd
import numpy as np
import glob

def load_data():
    files = sorted(glob.glob("Datasets/prices_round_5_day_*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, sep=";")
        day = int(f.split("day_")[1].split(".")[0])
        df["day"] = day
        # Ensure timestamp is continuous across days
        df["continuous_timestamp"] = df["timestamp"] + (day - 2) * 1000000
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df = load_data()

print("="*50)
print("🟢 ALPHA #1: Massive Directional Trends")
print("="*50)
products = [
    "PEBBLES_XL", "MICROCHIP_OVAL", "PEBBLES_XS", "OXYGEN_SHAKE_GARLIC", 
    "MICROCHIP_SQUARE", "GALAXY_SOUNDS_BLACK_HOLES", "UV_VISOR_AMBER", 
    "PANEL_2X4", "ROBOT_IRONING", "MICROCHIP_TRIANGLE"
]
for p in products:
    pdf = df[df["product"] == p].sort_values("continuous_timestamp")
    if not pdf.empty:
        start_price = pdf["mid_price"].iloc[0]
        end_price = pdf["mid_price"].iloc[-1]
        pct_change = (end_price - start_price) / start_price * 100
        print(f"{p}: Start={start_price}, End={end_price}, Net={pct_change:.1f}%, Theo_PnL=~${abs(end_price-start_price)*10:,.0f}")

print("\n" + "="*50)
print("🟢 ALPHA #2: PEBBLES Sum-Constraint")
print("="*50)
pebbles = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
pivot_df = df[df["product"].isin(pebbles)].pivot(index="continuous_timestamp", columns="product", values="mid_price")
pivot_df["sum"] = pivot_df[pebbles].sum(axis=1)
print(f"Overall Pebbles Sum: Mean={pivot_df['sum'].mean():.2f}, Std={pivot_df['sum'].std():.2f}")
# By day
day_pivot = df[df["product"].isin(pebbles)].pivot(index=["day", "timestamp"], columns="product", values="mid_price")
day_pivot["sum"] = day_pivot[pebbles].sum(axis=1)
for day in [2, 3, 4]:
    d_df = day_pivot.xs(day, level="day")
    print(f"Day {day} Sum: Mean={d_df['sum'].mean():.2f}, Std={d_df['sum'].std():.2f}")

print("\n" + "="*50)
print("🟢 ALPHA #3: SNACKPACK Pair Constraints")
print("="*50)
snackpacks = [p for p in df["product"].unique() if "SNACKPACK" in p]
sp_pivot = df[df["product"].isin(snackpacks)].pivot(index="continuous_timestamp", columns="product", values="mid_price")
if "SNACKPACK_CHOCOLATE" in sp_pivot.columns and "SNACKPACK_VANILLA" in sp_pivot.columns:
    cv_sum = sp_pivot["SNACKPACK_CHOCOLATE"] + sp_pivot["SNACKPACK_VANILLA"]
    print(f"CHOCOLATE + VANILLA: Mean={cv_sum.mean():.2f}, Std={cv_sum.std():.2f}, CV={cv_sum.std()/cv_sum.mean()*100:.2f}%")
if "SNACKPACK_PISTACHIO" in sp_pivot.columns and "SNACKPACK_RASPBERRY" in sp_pivot.columns:
    pr_sum = sp_pivot["SNACKPACK_PISTACHIO"] + sp_pivot["SNACKPACK_RASPBERRY"]
    print(f"PISTACHIO + RASPBERRY: Mean={pr_sum.mean():.2f}, Std={pr_sum.std():.2f}, CV={pr_sum.std()/pr_sum.mean()*100:.2f}%")
all_sum = sp_pivot.sum(axis=1)
print(f"All 5 SNACKPACK: Mean={all_sum.mean():.2f}, Std={all_sum.std():.2f}, CV={all_sum.std()/all_sum.mean()*100:.2f}%")

print("\n" + "="*50)
print("🟡 ALPHA #4 & #5: Correlations & Volatility")
print("="*50)
pairs = [
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"),
    ("MICROCHIP_RECTANGLE", "MICROCHIP_SQUARE"),
    ("UV_VISOR_AMBER", "UV_VISOR_MAGENTA"),
    ("PEBBLES_S", "PEBBLES_XL"),
    ("ROBOT_IRONING", "ROBOT_MOPPING"),
    ("MICROCHIP_OVAL", "MICROCHIP_TRIANGLE")
]

# Spread vs Volatility Check for OVAL
oval = df[df["product"] == "MICROCHIP_OVAL"].copy()
oval = oval.sort_values("continuous_timestamp")
oval["spread"] = oval["ask_price_1"] - oval["bid_price_1"]
oval["spread_smooth"] = oval["spread"].rolling(10).mean()
oval["ret"] = oval["mid_price"].pct_change()
oval["fwd_vol_50"] = oval["ret"].rolling(50).std().shift(-50)
oval["fwd_vol_200"] = oval["ret"].rolling(200).std().shift(-200)
print("MICROCHIP_OVAL Spread vs Fwd Vol 50: ", oval["spread_smooth"].corr(oval["fwd_vol_50"]))
print("MICROCHIP_OVAL Spread vs Fwd Vol 200:", oval["spread_smooth"].corr(oval["fwd_vol_200"]))

print("\nPrice Anti-Correlations:")
all_pivot = df.pivot(index="continuous_timestamp", columns="product", values="mid_price")
for p1, p2 in pairs:
    if p1 in all_pivot.columns and p2 in all_pivot.columns:
        print(f"{p1} vs {p2}: {all_pivot[p1].corr(all_pivot[p2]):.2f}")
