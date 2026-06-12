import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx

# --- CONFIGURATION ---
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "..", "Datasets")
FIG_DIR = os.path.join(_HERE, "..", "figs")
DAYS = [2, 3, 4]
BINS = 15  # For Mutual Information calculation

os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    """Loads and pivots mid_price data for all products."""
    dfs = []
    for day in DAYS:
        path = os.path.join(DATA_DIR, f"prices_round_5_day_{day}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, sep=";")
            dfs.append(df)

    full_df = pd.concat(dfs)
    # Create unique time key
    full_df["time_key"] = full_df["day"] * 1000000 + full_df["timestamp"]
    pivot_df = full_df.pivot(index="time_key", columns="product", values="mid_price")
    # Fill tiny gaps if any, then calculate returns
    returns = pivot_df.ffill().pct_change().dropna()
    return returns


def calc_mi_matrix(df):
    """Calculates a normalized Mutual Information matrix."""
    n = df.shape[1]
    mi_matrix = np.zeros((n, n))

    # Discretize for MI calculation
    kbd = KBinsDiscretizer(n_bins=BINS, encode="ordinal", strategy="uniform")
    df_discrete = kbd.fit_transform(df)

    for i in range(n):
        for j in range(i, n):
            mi = mutual_info_score(df_discrete[:, i], df_discrete[:, j])
            mi_matrix[i, j] = mi_matrix[j, i] = mi

    # Normalize MI to [0, 1] range: MI(X,Y) / sqrt(H(X)H(Y)) is complex,
    # let's use a simpler min-max or just raw for clustering.
    return pd.DataFrame(mi_matrix, index=df.columns, columns=df.columns)


def plot_mst(corr_matrix, title, filename):
    """Generates a Minimum Spanning Tree graph to show the 'skeleton' of relationships."""
    # Distance = sqrt(2 * (1 - rho))
    dist = np.sqrt(2 * (1 - corr_matrix.clip(-1, 1)))
    G = nx.from_pandas_adjacency(dist)
    mst = nx.minimum_spanning_tree(G)

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(mst, k=0.5, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(mst, pos, node_size=100, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_edges(mst, pos, alpha=0.3)

    # Draw labels with slight offset
    label_pos = {k: [v[0], v[1] + 0.02] for k, v in pos.items()}
    nx.draw_networkx_labels(mst, label_pos, font_size=7, font_weight="bold")

    plt.title(f"Minimum Spanning Tree: {title}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.close()


def run_analysis():
    print("Step 1: Loading Data...")
    returns = load_data()

    print("Step 2: Calculating Dependency Matrices...")
    corr = returns.corr()
    mi = calc_mi_matrix(returns)

    # --- CLUSTERING ---
    print("Step 3: Finding Clusters (Agglomerative)...")
    # Distance metric for hierarchical clustering
    dist_matrix = np.sqrt(2 * (1 - corr.clip(-1, 1)))
    Z = linkage(dist_matrix, method="ward")

    plt.figure(figsize=(15, 8))
    dendrogram(Z, labels=returns.columns, leaf_rotation=90, leaf_font_size=8)
    plt.title("Hierarchical Clustering Dendrogram (Price Returns)")
    plt.axhline(y=1.2, color="r", linestyle="--")
    plt.savefig(os.path.join(FIG_DIR, "dendrogram_returns.png"))
    plt.close()

    # --- AFFINITY PROPAGATION (Finding Leaders) ---
    print("Step 4: Identifying Cluster Exemplars (Leaders)...")
    af = AffinityPropagation(affinity="precomputed", random_state=42)
    af.fit(corr)

    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    clusters = {}
    for i, center_idx in enumerate(cluster_centers_indices):
        leader = returns.columns[center_idx]
        members = returns.columns[labels == i].tolist()
        clusters[leader] = members

    print("\nDetected Clusters & Leaders:")
    for leader, members in clusters.items():
        print(f"  * {leader} leads: {', '.join(members)}")

    # --- NETWORK ANALYSIS ---
    print("Step 5: Generating Minimum Spanning Tree...")
    plot_mst(corr, "Correlation Strength", "mst_correlation.png")

    # --- DIMENSION REDUCTION (ICA) ---
    # Independent Component Analysis is better at finding 'hidden sources' than PCA
    print("Step 6: ICA Factor Analysis...")
    ica = FastICA(n_components=5, random_state=42)
    ica_results = ica.fit_transform(returns)
    ica_components = pd.DataFrame(
        ica.components_.T,
        index=returns.columns,
        columns=[f"Source_{i + 1}" for i in range(5)],
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(ica_components, cmap="RdBu_r", center=0)
    plt.title("ICA Factor Loadings (Independent Economic Drivers)")
    plt.savefig(os.path.join(FIG_DIR, "ica_loadings.png"))
    plt.close()

    print(f"\nAll analysis complete. Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    run_analysis()
