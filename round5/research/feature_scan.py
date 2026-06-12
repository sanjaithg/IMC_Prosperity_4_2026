"""
Scan all products × lag horizons to find which microstructure features
actually predict future price moves. Outputs a summary table.
"""
import pandas as pd
import numpy as np
import glob, os, warnings
warnings.filterwarnings("ignore")

# --- Load data ---
data_dir = "Datasets"
frames = []
for f in sorted(glob.glob(os.path.join(data_dir, "prices_round_*_day_*.csv"))):
    frames.append(pd.read_csv(f, sep=";"))
prices = pd.concat(frames, ignore_index=True)
prices = prices.sort_values(["day", "product", "timestamp"]).reset_index(drop=True)

# --- Compute features ---
prices["abs_spread"] = prices["ask_price_1"] - prices["bid_price_1"]
prices["rel_spread"] = prices["abs_spread"] / prices["mid_price"]

bid_vols = prices["bid_volume_1"].fillna(0)
ask_vols = prices["ask_volume_1"].fillna(0)
denom = bid_vols + ask_vols
prices["imbalance"] = np.where(denom > 0, (bid_vols - ask_vols) / denom, 0.0)

bid_cols = [c for c in prices.columns if c.startswith("bid_volume")]
ask_cols = [c for c in prices.columns if c.startswith("ask_volume")]
prices["book_depth"] = prices[bid_cols].fillna(0).sum(axis=1) + prices[ask_cols].fillna(0).sum(axis=1)
prices["rel_depth"] = prices["book_depth"] / prices["mid_price"]

prices["ret"] = prices.groupby(["day", "product"])["mid_price"].pct_change().fillna(0.0)
prices["volatility"] = prices.groupby(["day", "product"])["ret"].transform(
    lambda x: x.rolling(50, min_periods=2).std(ddof=0)
).fillna(0.0)

feature_names = ["rel_spread", "imbalance", "volatility", "rel_depth"]
lags = [1, 3, 5, 10, 25, 50, 100]
products = sorted(prices["product"].unique())

print(f"Scanning {len(products)} products × {len(lags)} lags × {len(feature_names)} features...\n")

# --- Per-feature correlation with future price change at each lag ---
# For each (product, lag, feature): Pearson correlation of feature[t] with (mid[t+lag] - mid[t])
rows = []
for prod in products:
    df = prices[prices["product"] == prod].copy()
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)

    for lag in lags:
        delta = df["mid_price"].shift(-lag) - df["mid_price"]
        mask = ~delta.isna()

        for feat in feature_names:
            x = df[feat].values
            y = delta.values
            valid = mask.values & ~np.isnan(x) & ~np.isnan(y)
            if valid.sum() < 30:
                continue

            xv, yv = x[valid], y[valid]
            corr = np.corrcoef(xv, yv)[0, 1]
            dir_agree = np.mean(np.sign(xv - xv.mean()) == np.sign(yv)) if len(yv) > 0 else 0.5

            rows.append({
                "product": prod,
                "lag": lag,
                "feature": feat,
                "correlation": corr,
                "dir_accuracy": dir_agree,
                "n_samples": int(valid.sum())
            })

results = pd.DataFrame(rows)

# --- Summary 1: Average correlation across all products, per feature × lag ---
print("=" * 80)
print("AVERAGE CORRELATION (feature → future Δprice) across all products")
print("=" * 80)
pivot_corr = results.pivot_table(values="correlation", index="feature", columns="lag", aggfunc="mean")
pivot_corr = pivot_corr[lags]
print(pivot_corr.round(4).to_string())

print()
print("=" * 80)
print("AVERAGE DIRECTION ACCURACY across all products")
print("=" * 80)
pivot_dir = results.pivot_table(values="dir_accuracy", index="feature", columns="lag", aggfunc="mean")
pivot_dir = pivot_dir[lags]
print((pivot_dir * 100).round(1).to_string())

# --- Summary 2: Best product-feature-lag combos ---
print()
print("=" * 80)
print("TOP 30 STRONGEST CORRELATIONS (any product × feature × lag)")
print("=" * 80)
results["abs_corr"] = results["correlation"].abs()
top = results.nlargest(30, "abs_corr")
for _, r in top.iterrows():
    print(f"  {r['product']:<40} lag={r['lag']:>3}  {r['feature']:<12}  corr={r['correlation']:+.4f}  dir_acc={r['dir_accuracy']:.1%}")

# --- Summary 3: Which products have the MOST predictable spreads? ---
print()
print("=" * 80)
print("PRODUCTS RANKED BY MAX |correlation| (any feature, any lag)")
print("=" * 80)
prod_best = results.groupby("product")["abs_corr"].max().sort_values(ascending=False)
for prod, best_corr in prod_best.items():
    best_row = results[(results["product"] == prod) & (results["abs_corr"] == best_corr)].iloc[0]
    print(f"  {prod:<40} best |corr|={best_corr:.4f}  (lag={best_row['lag']}, feat={best_row['feature']})")
