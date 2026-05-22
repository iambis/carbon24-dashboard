
"""
Carbon-24 Pipeline 3 Tầng
==========================

Tầng 1 — HDBSCAN Noise Detection:
    Lọc 786 điểm nhiễu (7.74%) → đây là "Anomaly Detection" của đề tài

Tầng 2 — K-means Macro-clustering (k=3):
    Chạy trên 9,367 điểm sạch → 3 nhóm lớn:
    Cluster A: Bền vững (energy thấp, coordination cao)
    Cluster B: Nén (density cao, volume nhỏ)
    Cluster C: Phức tạp (mixed coordination, angle deviation cao)

Tầng 3 — GMM Micro-clustering (k=10):
    Bẻ nhỏ 3 nhóm lớn → 10 phân nhóm xác suất (polymorphs)
    Mỗi nhóm lớn ~3-4 sub-clusters

Pipeline output:
    carbon24_pipeline_results/
    ├── tier1_noise_analysis.csv
    ├── tier2_kmeans_clean.csv
    ├── tier2_kmeans_profile.csv
    ├── tier3_gmm_clean.csv
    ├── tier3_gmm_profile.csv
    └── pipeline_final.csv          ← file tổng hợp dùng cho dashboard
"""

from pathlib import Path
import warnings
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

RANDOM_STATE = 42
KMEANS_K     = 3
GMM_K        = 10

PROJECT_DIR  = Path.cwd()
OUTPUT_DIR   = PROJECT_DIR / "carbon24_pipeline_results"
FIGURE_DIR   = OUTPUT_DIR / "figures"

HDBSCAN_PATH = PROJECT_DIR / "hdbscan_phuc" / "hdbscan_results.csv"
FEATURES_PATH = PROJECT_DIR / "carbon24_features" / "carbon24_project" / "data" / "carbon24_features.csv"

# 19 features cau truc (khong co energy)
STRUCTURE_FEATURES = [
    "num_atoms",
    "a", "b", "c",
    "alpha", "beta", "gamma",
    "volume", "volume_per_atom",
    "b_over_a", "c_over_a",
    "angle_deviation",
    "mean_bond_length", "std_bond_length",
    "min_bond_length", "max_bond_length",
    "std_coordination", "min_coordination", "max_coordination",
]

# Ten nhan cho 3 macro-clusters (se cap nhat sau khi phan tich)
CLUSTER_LABELS = {0: "Cluster-A", 1: "Cluster-B", 2: "Cluster-C"}


# ============================================================================
# UTILITIES
# ============================================================================

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")


def load_data():
    hdb  = pd.read_csv(HDBSCAN_PATH)
    feat = pd.read_csv(FEATURES_PATH)
    # Merge features vao hdbscan results theo material_id
    avail = [c for c in STRUCTURE_FEATURES if c in feat.columns]
    merged = hdb.merge(feat[["material_id"] + avail], on="material_id", how="left")
    print(f"Loaded: {merged.shape}  |  structure features: {len(avail)}")
    return merged, avail


def prepare_X(df, features):
    X = df[features].copy()
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, scaler, imp


def metrics_report(X, labels, name):
    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return {}
    sil = silhouette_score(X[mask], labels[mask], sample_size=min(5000, mask.sum()), random_state=RANDOM_STATE)
    db  = davies_bouldin_score(X[mask], labels[mask])
    ch  = calinski_harabasz_score(X[mask], labels[mask])
    print(f"  [{name}] Silhouette={sil:.4f}  DB={db:.4f}  CH={ch:.1f}")
    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


# ============================================================================
# TANG 1: HDBSCAN NOISE DETECTION
# ============================================================================

def tier1_noise_analysis(df):
    print("\n" + "="*65)
    print("TANG 1: HDBSCAN NOISE DETECTION")
    print("="*65)

    noise_mask  = df["hdbscan_cluster"] == -1
    clean_mask  = df["hdbscan_cluster"] != -1

    n_noise = noise_mask.sum()
    n_clean = clean_mask.sum()
    print(f"  Noise  : {n_noise:,}  ({n_noise/len(df):.2%})")
    print(f"  Clean  : {n_clean:,}  ({n_clean/len(df):.2%})")

    # Phan tich energy cua noise
    if "relative_energy" in df.columns:
        e_noise = df.loc[noise_mask, "relative_energy"]
        e_clean = df.loc[clean_mask, "relative_energy"]
        diff = e_noise.mean() - e_clean.mean()
        print(f"  Energy noise: {e_noise.mean():.4f}  |  clean: {e_clean.mean():.4f}  |  diff: {diff:+.4f}")
        print(f"  -> Noise kem on dinh hon {diff:.4f} eV/atom")

    # Phan tich crystal system cua noise
    if "crystal_system" in df.columns:
        print("\n  Crystal system distribution (noise vs clean):")
        cs_noise = df.loc[noise_mask, "crystal_system"].value_counts(normalize=True) * 100
        cs_clean = df.loc[clean_mask, "crystal_system"].value_counts(normalize=True) * 100
        for cs in cs_clean.index:
            n_pct = cs_noise.get(cs, 0)
            c_pct = cs_clean.get(cs, 0)
            print(f"    {cs:<15}: noise={n_pct:.1f}%  clean={c_pct:.1f}%")

    # Luu noise analysis
    noise_df = df[noise_mask].copy()
    noise_df["tier1_label"] = "noise"
    noise_df.to_csv(OUTPUT_DIR / "tier1_noise_analysis.csv", index=False)
    print(f"\n  Saved: tier1_noise_analysis.csv  ({len(noise_df):,} rows)")

    return df[clean_mask].copy().reset_index(drop=True)


# ============================================================================
# TANG 2: K-MEANS MACRO-CLUSTERING
# ============================================================================

def tier2_kmeans(clean_df, features):
    print("\n" + "="*65)
    print(f"TANG 2: K-MEANS MACRO-CLUSTERING  (k={KMEANS_K})")
    print("="*65)

    X_scaled, scaler, imp = prepare_X(clean_df, features)

    km = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    labels = km.fit_predict(X_scaled)
    clean_df["kmeans_cluster"] = labels

    print(f"  Cluster distribution:")
    for c, n in sorted(zip(*np.unique(labels, return_counts=True))):
        print(f"    Cluster {c}: {n:,}  ({n/len(labels):.2%})")

    metrics_report(X_scaled, labels, "K-means")

    # Profile: trung binh cac features va energy theo cluster
    profile_cols = features + (["relative_energy", "energy", "crystal_system"] if "relative_energy" in clean_df.columns else [])
    profile_cols = [c for c in profile_cols if c in clean_df.columns]

    profile = clean_df.groupby("kmeans_cluster")[
        [c for c in profile_cols if clean_df[c].dtype != object]
    ].agg(["mean", "std", "median"]).round(4)

    # Dat ten cluster dua tren energy
    if "relative_energy" in clean_df.columns:
        energy_by_cluster = clean_df.groupby("kmeans_cluster")["relative_energy"].mean().sort_values()
        label_map = {}
        names = ["Ben vung (Low Energy)", "Trung gian", "Phuc tap (High Energy)"]
        for rank, (cid, _) in enumerate(energy_by_cluster.items()):
            label_map[cid] = names[rank]
        clean_df["kmeans_label"] = clean_df["kmeans_cluster"].map(label_map)
        print(f"\n  Cluster labels (by energy):")
        for cid, lbl in label_map.items():
            e = energy_by_cluster[cid]
            n = (labels == cid).sum()
            print(f"    Cluster {cid} -> '{lbl}'  (mean energy={e:.4f}, n={n:,})")

    # Luu
    clean_df.to_csv(OUTPUT_DIR / "tier2_kmeans_clean.csv", index=False)
    profile.to_csv(OUTPUT_DIR / "tier2_kmeans_profile.csv")
    print(f"\n  Saved: tier2_kmeans_clean.csv  tier2_kmeans_profile.csv")

    return clean_df, X_scaled, label_map if "relative_energy" in clean_df.columns else {}


# ============================================================================
# TANG 3: GMM MICRO-CLUSTERING
# ============================================================================

def tier3_gmm(clean_df, features, kmeans_label_map):
    print("\n" + "="*65)
    print(f"TANG 3: GMM MICRO-CLUSTERING  (k={GMM_K})")
    print("="*65)

    X_scaled, scaler, imp = prepare_X(clean_df, features)

    gmm = GaussianMixture(
        n_components=GMM_K,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=5,
        max_iter=300,
    )
    gmm_labels = gmm.fit_predict(X_scaled)
    gmm_proba  = gmm.predict_proba(X_scaled)

    clean_df["gmm_cluster"]     = gmm_labels
    clean_df["gmm_probability"] = gmm_proba.max(axis=1)

    print(f"  GMM converged: {gmm.converged_}")
    print(f"  BIC: {gmm.bic(X_scaled):.1f}  |  AIC: {gmm.aic(X_scaled):.1f}")
    print(f"  Cluster distribution:")
    for c, n in sorted(zip(*np.unique(gmm_labels, return_counts=True))):
        print(f"    GMM-{c}: {n:,}  ({n/len(gmm_labels):.2%})")

    metrics_report(X_scaled, gmm_labels, "GMM")

    # Lien ket GMM cluster voi K-means cluster
    if "kmeans_cluster" in clean_df.columns:
        print(f"\n  GMM sub-clusters within each K-means macro-cluster:")
        for km_c in sorted(clean_df["kmeans_cluster"].unique()):
            mask = clean_df["kmeans_cluster"] == km_c
            gmm_in_km = clean_df.loc[mask, "gmm_cluster"].value_counts().sort_index()
            lbl = kmeans_label_map.get(km_c, f"Cluster-{km_c}")
            print(f"    K-means {km_c} ({lbl}): GMM sub-clusters = {gmm_in_km.to_dict()}")

    # Profile GMM
    profile_num_cols = [c for c in features + ["relative_energy"] if c in clean_df.columns]
    gmm_profile = clean_df.groupby("gmm_cluster")[profile_num_cols].mean().round(4)
    if "kmeans_cluster" in clean_df.columns:
        # Them thong tin kmeans majority
        gmm_profile["kmeans_majority"] = (
            clean_df.groupby("gmm_cluster")["kmeans_cluster"]
            .agg(lambda x: x.mode()[0])
        )
        if kmeans_label_map:
            gmm_profile["kmeans_label"] = gmm_profile["kmeans_majority"].map(kmeans_label_map)

    # Luu
    clean_df.to_csv(OUTPUT_DIR / "tier3_gmm_clean.csv", index=False)
    gmm_profile.to_csv(OUTPUT_DIR / "tier3_gmm_profile.csv")
    print(f"\n  Saved: tier3_gmm_clean.csv  tier3_gmm_profile.csv")

    return clean_df, gmm_profile


# ============================================================================
# TONG HOP PIPELINE FINAL
# ============================================================================

def build_final_output(full_df, clean_df):
    """Ghep noise + clean vao 1 file tong hop cho dashboard."""
    print("\n" + "="*65)
    print("TONG HOP PIPELINE FINAL")
    print("="*65)

    # Noise rows
    noise_rows = full_df[full_df["hdbscan_cluster"] == -1].copy()
    noise_rows["pipeline_stage"]  = "noise"
    noise_rows["kmeans_cluster"]  = -1
    noise_rows["kmeans_label"]    = "Noise"
    noise_rows["gmm_cluster"]     = -1
    noise_rows["gmm_probability"] = 0.0

    # Clean rows
    clean_rows = clean_df.copy()
    clean_rows["pipeline_stage"] = "clean"

    final = pd.concat([clean_rows, noise_rows], ignore_index=True)
    final.to_csv(OUTPUT_DIR / "pipeline_final.csv", index=False)
    print(f"  pipeline_final.csv: {final.shape}")

    # Summary
    print(f"\n  PIPELINE SUMMARY:")
    print(f"    Total samples  : {len(final):,}")
    print(f"    Noise (Tier 1) : {(final['pipeline_stage']=='noise').sum():,}  ({(final['pipeline_stage']=='noise').mean():.2%})")
    print(f"    Clean (Tier 2+): {(final['pipeline_stage']=='clean').sum():,}  ({(final['pipeline_stage']=='clean').mean():.2%})")
    if "kmeans_label" in final.columns:
        print(f"\n    K-means macro-clusters:")
        for lbl, n in final[final["pipeline_stage"]=="clean"]["kmeans_label"].value_counts().items():
            print(f"      {lbl}: {n:,}  ({n/len(final):.2%})")
    if "gmm_cluster" in final.columns:
        print(f"\n    GMM micro-clusters: {final[final['pipeline_stage']=='clean']['gmm_cluster'].nunique()} clusters")

    return final


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_tier1_noise(full_df):
    """PCA scatter: noise vs clean."""
    if "pca1" not in full_df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    clean = full_df[full_df["hdbscan_cluster"] != -1]
    noise = full_df[full_df["hdbscan_cluster"] == -1]
    ax.scatter(clean["pca1"], clean["pca2"], c="steelblue", alpha=0.2, s=8, label=f"Clean ({len(clean):,})")
    ax.scatter(noise["pca1"], noise["pca2"], c="crimson",   alpha=0.8, s=25, label=f"Noise ({len(noise):,})",
               edgecolors="darkred", linewidths=0.4)
    ax.set_xlabel("PCA1"); ax.set_ylabel("PCA2")
    ax.set_title(f"Tier 1: HDBSCAN Noise Detection\n{len(noise):,} noise points ({len(noise)/len(full_df):.2%})",
                 fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tier1_noise_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  tier1_noise_pca.png")


def plot_tier2_kmeans(clean_df):
    """PCA scatter: 3 K-means clusters."""
    if "pca1" not in clean_df.columns or "kmeans_cluster" not in clean_df.columns:
        return
    colors = ["#2ecc71", "#3498db", "#e74c3c"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA scatter
    ax = axes[0]
    for c in sorted(clean_df["kmeans_cluster"].unique()):
        mask = clean_df["kmeans_cluster"] == c
        lbl  = clean_df.loc[mask, "kmeans_label"].iloc[0] if "kmeans_label" in clean_df.columns else f"Cluster {c}"
        ax.scatter(clean_df.loc[mask, "pca1"], clean_df.loc[mask, "pca2"],
                   c=colors[c % len(colors)], alpha=0.3, s=10, label=f"{lbl} ({mask.sum():,})")
    ax.set_xlabel("PCA1"); ax.set_ylabel("PCA2")
    ax.set_title("Tier 2: K-means Macro-clusters (k=3)", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Energy boxplot
    ax = axes[1]
    if "relative_energy" in clean_df.columns and "kmeans_label" in clean_df.columns:
        labels_order = clean_df.groupby("kmeans_label")["relative_energy"].mean().sort_values().index.tolist()
        data = [clean_df.loc[clean_df["kmeans_label"]==lbl, "relative_energy"].values for lbl in labels_order]
        bp = ax.boxplot(data, tick_labels=labels_order, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Relative Energy (eV/atom)")
        ax.set_title("Energy Distribution by K-means Cluster", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tier2_kmeans.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  tier2_kmeans.png")


def plot_tier3_gmm(clean_df, gmm_profile):
    """PCA scatter: 10 GMM micro-clusters + heatmap profile."""
    if "pca1" not in clean_df.columns or "gmm_cluster" not in clean_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PCA scatter GMM
    ax = axes[0]
    cmap = plt.cm.get_cmap("tab10", GMM_K)
    for c in sorted(clean_df["gmm_cluster"].unique()):
        mask = clean_df["gmm_cluster"] == c
        ax.scatter(clean_df.loc[mask, "pca1"], clean_df.loc[mask, "pca2"],
                   c=[cmap(c)], alpha=0.4, s=10, label=f"GMM-{c} ({mask.sum():,})")
    ax.set_xlabel("PCA1"); ax.set_ylabel("PCA2")
    ax.set_title("Tier 3: GMM Micro-clusters (k=10)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # Heatmap: GMM profile (normalized)
    ax = axes[1]
    num_cols = [c for c in ["relative_energy", "volume_per_atom", "mean_bond_length",
                             "std_coordination", "angle_deviation", "density"]
                if c in gmm_profile.columns]
    if num_cols:
        profile_sub = gmm_profile[num_cols].copy()
        profile_norm = (profile_sub - profile_sub.min()) / (profile_sub.max() - profile_sub.min() + 1e-9)
        sns.heatmap(profile_norm, annot=True, fmt=".2f", cmap="RdYlGn_r",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Normalized value"})
        ax.set_title("GMM Cluster Profile (normalized)", fontweight="bold")
        ax.set_xlabel("Feature"); ax.set_ylabel("GMM Cluster")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "tier3_gmm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  tier3_gmm.png")


def plot_pipeline_overview(final_df):
    """Sankey-style bar: flow from noise -> kmeans -> gmm."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Carbon-24 Pipeline 3 Tầng — Tổng quan", fontsize=14, fontweight="bold")

    # Tier 1
    ax = axes[0]
    counts = final_df["pipeline_stage"].value_counts()
    ax.bar(["Clean", "Noise"], [counts.get("clean", 0), counts.get("noise", 0)],
           color=["steelblue", "crimson"], alpha=0.8, edgecolor="black")
    for i, (lbl, v) in enumerate(zip(["Clean", "Noise"],
                                      [counts.get("clean", 0), counts.get("noise", 0)])):
        ax.text(i, v + 50, f"{v:,}\n({v/len(final_df):.1%})", ha="center", fontweight="bold")
    ax.set_title("Tier 1: Noise Filtering", fontweight="bold")
    ax.set_ylabel("Số mẫu"); ax.grid(axis="y", alpha=0.3)

    # Tier 2
    ax = axes[1]
    if "kmeans_label" in final_df.columns:
        clean_only = final_df[final_df["pipeline_stage"] == "clean"]
        km_counts = clean_only["kmeans_label"].value_counts()
        colors2 = ["#2ecc71", "#3498db", "#e74c3c"]
        bars = ax.bar(range(len(km_counts)), km_counts.values,
                      color=colors2[:len(km_counts)], alpha=0.8, edgecolor="black")
        ax.set_xticks(range(len(km_counts)))
        ax.set_xticklabels(km_counts.index, rotation=15, ha="right", fontsize=9)
        for bar, v in zip(bars, km_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                    f"{v:,}", ha="center", fontweight="bold", fontsize=9)
    ax.set_title("Tier 2: K-means Macro (k=3)", fontweight="bold")
    ax.set_ylabel("Số mẫu"); ax.grid(axis="y", alpha=0.3)

    # Tier 3
    ax = axes[2]
    if "gmm_cluster" in final_df.columns:
        clean_only = final_df[final_df["pipeline_stage"] == "clean"]
        gmm_counts = clean_only["gmm_cluster"].value_counts().sort_index()
        cmap = plt.cm.get_cmap("tab10", GMM_K)
        bars = ax.bar(gmm_counts.index, gmm_counts.values,
                      color=[cmap(i) for i in gmm_counts.index], alpha=0.8, edgecolor="black")
        for bar, v in zip(bars, gmm_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{v:,}", ha="center", fontsize=8, fontweight="bold")
    ax.set_title("Tier 3: GMM Micro (k=10)", fontweight="bold")
    ax.set_xlabel("GMM Cluster ID"); ax.set_ylabel("Số mẫu"); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "pipeline_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  pipeline_overview.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*65)
    print("CARBON-24 PIPELINE 3 TANG")
    print("="*65)
    print("Tier 1: HDBSCAN Noise  ->  Tier 2: K-means (k=3)  ->  Tier 3: GMM (k=10)")

    ensure_dirs()
    full_df, features = load_data()

    # Tier 1
    clean_df = tier1_noise_analysis(full_df)

    # Tier 2
    clean_df, X_scaled, kmeans_label_map = tier2_kmeans(clean_df, features)

    # Tier 3
    clean_df, gmm_profile = tier3_gmm(clean_df, features, kmeans_label_map)

    # Final output
    final_df = build_final_output(full_df, clean_df)

    # Plots
    print("\n" + "="*65)
    print("VISUALIZATIONS")
    print("="*65)
    plot_tier1_noise(full_df)
    plot_tier2_kmeans(clean_df)
    plot_tier3_gmm(clean_df, gmm_profile)
    plot_pipeline_overview(final_df)

    print("\n" + "="*65)
    print("HOAN TAT")
    print("="*65)
    print(f"Output: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size:>8.1f} KB")


if __name__ == "__main__":
    main()
