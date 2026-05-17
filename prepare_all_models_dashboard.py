import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "all_models_dashboard.csv")


MODEL_FILES = {
    "KMeans": [
        "kmeans_cluster_assignments_dashboard.csv",
        "kmeans_cluster_assignments_best.csv",
        "kmeans_cluster_assignments.csv",
    ],
    "Hierarchical": [
        "hierarchical_cluster_assignments_dashboard.csv",
        "hierarchical_cluster_assignments_best.csv",
        "hierarchical_cluster_assignments.csv",
    ],
    "HDBSCAN": [
        "hdbscan_cluster_assignments_dashboard.csv",
        "hdbscan_cluster_assignments_best.csv",
        "hdbscan_cluster_assignments.csv",
    ],
    "GMM": [
        "gmm_cluster_assignments_dashboard.csv",
        "gmm_cluster_assignments_best.csv",
        "gmm_cluster_assignments.csv",
    ],
}


CLUSTER_CANDIDATES = {
    "KMeans": [
        "cluster_kmeans",
        "kmeans_cluster",
        "kmeans_label",
        "cluster_label",
        "cluster",
        "label",
        "labels",
    ],
    "Hierarchical": [
        "cluster_hierarchical",
        "hierarchical_cluster",
        "cluster_label",
        "cluster",
        "label",
        "labels",
    ],
    "HDBSCAN": [
        "cluster_hdbscan",
        "hdbscan_cluster",
        "cluster_label",
        "cluster",
        "label",
        "labels",
    ],
    "GMM": [
        "cluster_gmm",
        "gmm_cluster",
        "cluster_label",
        "cluster",
        "label",
        "labels",
    ],
}


EXTRA_MODEL_COLUMNS = [
    "hdbscan_probability",
    "hdbscan_outlier_score",
    "gmm_max_probability",
    "gmm_second_probability",
    "gmm_uncertainty",
]


BASE_KEEP_COLUMNS = [
    "row_index",

    "PC1",
    "PC2",
    "PC3",

    "relative_energy",
    "relative_energy_meV",
    "energy",

    "density",
    "volume",
    "volume_per_atom",
    "num_atoms",

    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",

    "b_over_a",
    "c_over_a",
    "angle_deviation",
    "lattice_anisotropy",
    "angle_mean",
    "angle_std",

    "mean_bond_length",
    "std_bond_length",
    "min_bond_length",
    "max_bond_length",

    "mean_coordination",
    "std_coordination",
    "min_coordination",
    "max_coordination",

    "crystal_system",
    "space_group_number",
    "space_group_symbol",

    "formula",
    "elements",
    "split",
    "source_file",
]


def find_existing_file(file_list):
    for fname in file_list:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            return path
    return None


def detect_cluster_col(df, algorithm):
    candidates = CLUSTER_CANDIDATES.get(algorithm, [])

    for c in candidates:
        if c in df.columns:
            return c

    cluster_like = [c for c in df.columns if "cluster" in c.lower()]

    if len(cluster_like) > 0:
        return cluster_like[0]

    return None


def ensure_row_index(df):
    df = df.copy()

    if "row_index" not in df.columns:
        df["row_index"] = np.arange(len(df))

    return df


def add_relative_energy_mev(df):
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
            print("No energy / relative_energy column found. Skip relative_energy_meV.")

    return df


def compute_shared_pca(df, n_components=3):
    """
    Luôn tính lại PC1, PC2, PC3 từ các feature numeric.
    PCA này dùng chung cho tất cả thuật toán để dashboard so sánh công bằng.
    """
    df = df.copy()

    # Xóa PC cũ nếu có
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

    if len(numeric_cols) < n_components:
        raise ValueError(
            f"Không đủ numeric columns để tính PCA {n_components}D. "
            f"numeric_cols={numeric_cols}"
        )

    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    for i in range(n_components):
        df[f"PC{i+1}"] = X_pca[:, i]

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("Total explained:", pca.explained_variance_ratio_.sum())

    return df


def load_model_files():
    loaded = {}

    for algo, file_list in MODEL_FILES.items():
        path = find_existing_file(file_list)

        if path is None:
            print(f"[SKIP] {algo}: no file found")
            continue

        df = pd.read_csv(path)
        df = ensure_row_index(df)
        df = add_relative_energy_mev(df)

        cluster_col = detect_cluster_col(df, algo)

        if cluster_col is None:
            print(f"[SKIP] {algo}: cannot detect cluster column")
            continue

        print(f"[LOAD] {algo}: {path}")
        print(f"       shape={df.shape}, cluster_col={cluster_col}")

        loaded[algo] = {
            "df": df,
            "path": path,
            "cluster_col": cluster_col,
        }

    return loaded


def build_base_dataframe(loaded):
    """
    Lấy file đầu tiên làm base feature table.
    Tính PCA 3D trên base này.
    """
    first_algo = list(loaded.keys())[0]
    base_df = loaded[first_algo]["df"].copy()

    base_df = ensure_row_index(base_df)
    base_df = add_relative_energy_mev(base_df)

    base_df = compute_shared_pca(base_df, n_components=3)

    keep_cols = [c for c in BASE_KEEP_COLUMNS if c in base_df.columns]

    base_small = base_df[keep_cols].copy()

    print("Base algorithm:", first_algo)
    print("Base dataframe shape:", base_small.shape)
    print("Base columns:", base_small.columns.tolist())

    return base_small


def build_long_dashboard_dataframe(loaded, base_small):
    """
    Chuyển các kết quả thuật toán thành long format:
    mỗi cấu trúc có nhiều dòng, mỗi dòng tương ứng một algorithm.
    """
    long_dfs = []

    for algo, obj in loaded.items():
        df_algo = obj["df"].copy()
        df_algo = ensure_row_index(df_algo)

        cluster_col = obj["cluster_col"]

        keep_cols = ["row_index", cluster_col]

        for c in EXTRA_MODEL_COLUMNS:
            if c in df_algo.columns:
                keep_cols.append(c)

        cluster_df = df_algo[keep_cols].copy()
        cluster_df = cluster_df.rename(columns={cluster_col: "cluster_label"})

        merged = base_small.merge(
            cluster_df,
            on="row_index",
            how="left"
        )

        merged["algorithm"] = algo

        merged = merged.dropna(subset=["cluster_label"]).copy()
        merged["cluster_label"] = merged["cluster_label"].astype(str)

        long_dfs.append(merged)

        print(f"[MERGE] {algo}: {merged.shape}")

    all_models_df = pd.concat(long_dfs, ignore_index=True, sort=False)

    # Đưa algorithm và cluster_label lên đầu
    first_cols = ["algorithm", "cluster_label", "row_index"]
    first_cols = [c for c in first_cols if c in all_models_df.columns]

    remaining_cols = [c for c in all_models_df.columns if c not in first_cols]

    all_models_df = all_models_df[first_cols + remaining_cols]

    return all_models_df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    loaded = load_model_files()

    if len(loaded) == 0:
        raise RuntimeError(
            "Không tìm thấy file clustering nào trong thư mục data/. "
            "Cần ít nhất kmeans_cluster_assignments_best.csv."
        )

    base_small = build_base_dataframe(loaded)

    all_models_df = build_long_dashboard_dataframe(loaded, base_small)

    all_models_df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 80)
    print("Saved:", OUTPUT_PATH)
    print("Final shape:", all_models_df.shape)

    print("Algorithms:")
    print(all_models_df["algorithm"].value_counts())

    print("Columns:")
    print(all_models_df.columns.tolist())

    print("Preview:")
    print(all_models_df.head())

    required_cols = ["algorithm", "cluster_label", "PC1", "PC2", "PC3"]

    missing_required = [
        c for c in required_cols
        if c not in all_models_df.columns
    ]

    if len(missing_required) > 0:
        print("WARNING: Missing required columns:", missing_required)
    else:
        print("All required dashboard columns are present.")


if __name__ == "__main__":
    main()