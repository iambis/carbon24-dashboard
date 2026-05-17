import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


INPUT_PATH = "data/kmeans_cluster_assignments_best.csv"
OUTPUT_PATH = "data/kmeans_cluster_assignments_dashboard.csv"


def detect_cluster_column(df):
    cluster_candidates = [
        "cluster_kmeans",
        "cluster",
        "label",
        "labels",
        "kmeans_label",
        "kmeans_cluster"
    ]

    cluster_col = None

    for c in cluster_candidates:
        if c in df.columns:
            cluster_col = c
            break

    if cluster_col is None:
        cluster_like = [c for c in df.columns if "cluster" in c.lower()]
        if len(cluster_like) > 0:
            cluster_col = cluster_like[0]

    if cluster_col is None:
        raise ValueError("Không tìm thấy cột cluster trong CSV.")

    return cluster_col


def create_relative_energy_mev(df):
    df = df.copy()

    if "relative_energy_meV" not in df.columns:
        if "relative_energy" in df.columns:
            df["relative_energy"] = pd.to_numeric(df["relative_energy"], errors="coerce")
            df["relative_energy_meV"] = df["relative_energy"] * 1000
            print("Created relative_energy_meV from relative_energy")

        elif "energy" in df.columns:
            df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
            df["relative_energy"] = df["energy"] - df["energy"].min()
            df["relative_energy_meV"] = df["relative_energy"] * 1000
            print("Created relative_energy and relative_energy_meV from energy")

        else:
            print("No energy or relative_energy column found.")

    return df


def compute_pca_3d(df):
    df = df.copy()

    # Xóa PC cũ nếu có để tính lại nhất quán
    for col in ["PC1", "PC2", "PC3"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    exclude_keywords = [
        "cluster",
        "label",
        "algorithm",
        "row_index",
        "id",
        "source",
        "split",
        "formula",
        "elements",
        "crystal_system",
        "space_group_symbol",
        "PC1",
        "PC2",
        "PC3",
        "UMAP1",
        "UMAP2",
        "UMAP3",
    ]

    numeric_cols = []

    for col in df.columns:
        if any(k.lower() in col.lower() for k in exclude_keywords):
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        numeric_ratio = converted.notna().mean()

        if numeric_ratio >= 0.7:
            numeric_cols.append(col)
            df[col] = converted

    print("Numeric columns used for PCA 3D:")
    for c in numeric_cols:
        print("-", c)

    if len(numeric_cols) < 3:
        raise ValueError(
            "Không đủ numeric columns để tính PCA 3D. "
            f"numeric_cols={numeric_cols}"
        )

    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    df["PC3"] = X_pca[:, 2]

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained:", pca.explained_variance_ratio_.sum())

    return df


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    print("Input shape:", df.shape)
    print("Input columns:")
    for i, col in enumerate(df.columns):
        print(i, col)

    # 1. Detect cluster column
    cluster_col = detect_cluster_column(df)

    print("Detected cluster column:", cluster_col)

    # Rename cluster column for dashboard
    if cluster_col != "cluster_kmeans":
        df = df.rename(columns={cluster_col: "cluster_kmeans"})
        cluster_col = "cluster_kmeans"

    # 2. Add algorithm column
    if "algorithm" not in df.columns:
        df["algorithm"] = "KMeans"

    # 3. Ensure row_index
    if "row_index" not in df.columns:
        df["row_index"] = np.arange(len(df))

    # 4. Create relative_energy_meV
    df = create_relative_energy_mev(df)

    # 5. Compute PC1, PC2, PC3
    df = compute_pca_3d(df)

    # 6. Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Output shape:", df.shape)
    print("Output columns:")
    for i, col in enumerate(df.columns):
        print(i, col)

    print("Preview:")
    print(df[["algorithm", "cluster_kmeans", "PC1", "PC2", "PC3"]].head())


if __name__ == "__main__":
    main()