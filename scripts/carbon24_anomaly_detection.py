"""
Carbon-24 Anomaly Detection Module
===================================

Phát hiện dị biệt dựa trên kết quả phân cụm HDBSCAN.

Ý nghĩa khoa học:
- HDBSCAN Noise: cấu trúc không thuộc cụm nào → hình học bất thường
- Low Probability: membership yếu → nằm ở biên cụm, không điển hình
- Isolation Forest: dễ bị cô lập trong không gian features cấu trúc → outlier đa chiều

Nguyên tắc quan trọng:
- Chỉ dùng features CẤU TRÚC (lattice, bond, coordination) làm input
- KHÔNG dùng energy/relative_energy làm input (tránh data leakage)
- Energy chỉ dùng để DIỄN GIẢI kết quả sau khi phát hiện
"""

from pathlib import Path
import sys
import warnings

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

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
HDBSCAN_PROBABILITY_THRESHOLD = 0.5
ISOLATION_FOREST_CONTAMINATION = 0.077  # ~7.7% khớp với noise ratio của HDBSCAN

PROJECT_DIR = Path.cwd()
HDBSCAN_DIR = PROJECT_DIR / "hdbscan_phuc"
OUTPUT_DIR = PROJECT_DIR / "carbon24_anomaly_detection"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Input files — thử nhiều đường dẫn có thể có
HDBSCAN_RESULTS_PATH = HDBSCAN_DIR / "hdbscan_results.csv"

FEATURES_CANDIDATES = [
    PROJECT_DIR / "carbon24_features.csv",
    PROJECT_DIR / "carbon24_features" / "carbon24_project" / "data" / "carbon24_features.csv",
    PROJECT_DIR / "carbon24_features_v2" / "carbon24_project_v2" / "data" / "carbon24_features_v2.csv",
]

# Output files
ANOMALY_RESULTS_PATH  = OUTPUT_DIR / "anomaly_detection_results.csv"
ANOMALY_SUMMARY_PATH  = OUTPUT_DIR / "anomaly_summary.csv"
ANOMALY_COMPARISON_PATH = OUTPUT_DIR / "anomaly_method_comparison.csv"
ANOMALY_DETAILS_PATH  = OUTPUT_DIR / "anomaly_details.csv"

# Features cấu trúc dùng cho Isolation Forest (KHÔNG có energy)
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


# ============================================================================
# DATA LOADING
# ============================================================================

def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✅ Output dir: {OUTPUT_DIR}")


def load_hdbscan_results() -> pd.DataFrame:
    if not HDBSCAN_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy: {HDBSCAN_RESULTS_PATH}")
    df = pd.read_csv(HDBSCAN_RESULTS_PATH)
    print(f"✅ HDBSCAN results: {df.shape}  |  cols: {df.columns.tolist()}")
    return df


def load_structure_features() -> pd.DataFrame | None:
    """Load file features cấu trúc gốc để dùng cho Isolation Forest."""
    for path in FEATURES_CANDIDATES:
        if path.exists():
            feat = pd.read_csv(path)
            available = [c for c in STRUCTURE_FEATURES if c in feat.columns]
            if len(available) >= 5:
                print(f"✅ Features file: {path}  |  {len(available)} structure features")
                return feat[["material_id"] + available]
    print("⚠️  Không tìm thấy file features cấu trúc — Isolation Forest sẽ dùng PCA")
    return None


# ============================================================================
# PHƯƠNG PHÁP 1: HDBSCAN NOISE
# ============================================================================

def detect_hdbscan_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Điểm có cluster = -1 → HDBSCAN không thể gán vào cụm nào.
    Ý nghĩa: cấu trúc có hình học quá khác biệt, nằm ở vùng thưa.
    """
    print("\n" + "="*70)
    print("PHƯƠNG PHÁP 1: HDBSCAN NOISE  (cluster = -1)")
    print("="*70)

    cluster_col = next(
        (c for c in df.columns if "cluster" in c.lower() and "hdbscan" in c.lower()),
        None
    )
    if cluster_col is None:
        raise ValueError("Không tìm thấy cột hdbscan_cluster")

    noise_mask = df[cluster_col] == -1
    print(f"  Noise points : {noise_mask.sum():,} / {len(df):,}  ({noise_mask.mean():.2%})")
    print(f"  Ý nghĩa      : cấu trúc không thuộc cụm nào → hình học bất thường")

    df["is_hdbscan_noise"] = noise_mask.astype(int)
    return df


# ============================================================================
# PHƯƠNG PHÁP 2: LOW MEMBERSHIP PROBABILITY
# ============================================================================

def detect_low_probability(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Điểm có membership probability < threshold.
    Ý nghĩa: thuộc cụm nhưng không điển hình, nằm ở biên cụm.
    """
    print("\n" + "="*70)
    print(f"PHƯƠNG PHÁP 2: LOW MEMBERSHIP PROBABILITY  (< {threshold})")
    print("="*70)

    prob_col = next(
        (c for c in df.columns if "prob" in c.lower() and "hdbscan" in c.lower()),
        None
    )
    if prob_col is None:
        raise ValueError("Không tìm thấy cột hdbscan_probability")

    low_mask = df[prob_col] < threshold
    print(f"  Low-prob points : {low_mask.sum():,} / {len(df):,}  ({low_mask.mean():.2%})")
    print(f"  Probability stats:")
    print(df[prob_col].describe().to_string())
    print(f"  Ý nghĩa         : membership yếu → cấu trúc không điển hình cho cụm")

    df["is_low_probability"] = low_mask.astype(int)
    return df


# ============================================================================
# PHƯƠNG PHÁP 3: ISOLATION FOREST
# ============================================================================

def detect_isolation_forest(
    df: pd.DataFrame,
    feat_df: pd.DataFrame | None,
    contamination: float = 0.077,
) -> pd.DataFrame:
    """
    Isolation Forest trên features CẤU TRÚC gốc (không dùng energy).
    Ý nghĩa: cấu trúc dễ bị cô lập trong không gian đa chiều → outlier thực sự.

    Nếu không có features gốc, fallback sang PCA (kết quả kém tin cậy hơn).
    """
    print("\n" + "="*70)
    print(f"PHƯƠNG PHÁP 3: ISOLATION FOREST  (contamination={contamination:.1%})")
    print("="*70)

    # --- Chọn features ---
    if feat_df is not None:
        # Merge features gốc vào df theo material_id
        available = [c for c in STRUCTURE_FEATURES if c in feat_df.columns]
        merged = df[["material_id"]].merge(
            feat_df[["material_id"] + available],
            on="material_id", how="left"
        )
        X_raw = merged[available].copy()
        feature_source = f"{len(available)} structure features (lattice + bond + coordination)"
        using_pca_fallback = False
    else:
        # Fallback: chỉ dùng PCA — kết quả kém tin cậy
        pca_cols = [c for c in df.columns if c.lower().startswith("pca")]
        X_raw = df[pca_cols].copy()
        feature_source = f"PCA fallback ({len(pca_cols)} dims) — kém tin cậy hơn"
        using_pca_fallback = True

    if X_raw.shape[1] == 0:
        print("❌ Không có features nào!")
        df["is_isolation_forest_anomaly"] = 0
        df["isolation_forest_score"] = 0.0
        return df

    print(f"  Features: {feature_source}")
    if using_pca_fallback:
        print("  ⚠️  CẢNH BÁO: Đang dùng PCA fallback vì thiếu file features gốc.")
        print("      Kết quả Isolation Forest sẽ kém tin cậy hơn.")

    # --- Tiền xử lý ---
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # --- Fit Isolation Forest ---
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    preds  = iso.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
    scores = iso.score_samples(X_scaled) # càng âm càng bất thường

    is_anomaly = (preds == -1).astype(int)
    print(f"  Anomaly points  : {is_anomaly.sum():,} / {len(df):,}  ({is_anomaly.mean():.2%})")
    print(f"  Score range     : [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  Ý nghĩa         : cấu trúc dễ bị cô lập trong feature space → outlier đa chiều")

    df["is_isolation_forest_anomaly"] = is_anomaly
    df["isolation_forest_score"] = scores
    df["isolation_forest_used_pca"] = int(using_pca_fallback)
    return df


# ============================================================================
# KẾT HỢP & PHÂN TÍCH
# ============================================================================

def combine_methods(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*70)
    print("KẾT HỢP CÁC PHƯƠNG PHÁP")
    print("="*70)

    df["anomaly_vote_count"] = (
        df["is_hdbscan_noise"] +
        df["is_low_probability"] +
        df["is_isolation_forest_anomaly"]
    )
    df["is_anomaly_any"]       = (df["anomaly_vote_count"] >= 1).astype(int)
    df["is_anomaly_consensus"] = (df["anomaly_vote_count"] >= 2).astype(int)
    df["is_anomaly_all"]       = (df["anomaly_vote_count"] == 3).astype(int)

    print(f"  Any (≥1 method)  : {df['is_anomaly_any'].sum():,}  ({df['is_anomaly_any'].mean():.2%})")
    print(f"  Consensus (≥2)   : {df['is_anomaly_consensus'].sum():,}  ({df['is_anomaly_consensus'].mean():.2%})")
    print(f"  All 3 methods    : {df['is_anomaly_all'].sum():,}  ({df['is_anomaly_all'].mean():.2%})")

    print("\n  Vote distribution:")
    print(df["anomaly_vote_count"].value_counts().sort_index().to_string())
    return df


def analyze_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diễn giải ý nghĩa: anomalies có năng lượng như thế nào?
    Đây là bước DIỄN GIẢI, không phải input.
    """
    print("\n" + "="*70)
    print("DIỄN GIẢI: NĂNG LƯỢNG CỦA ANOMALIES")
    print("="*70)

    rows = []
    for col, label in [
        ("is_hdbscan_noise",            "HDBSCAN Noise"),
        ("is_low_probability",          "Low Probability"),
        ("is_isolation_forest_anomaly", "Isolation Forest"),
        ("is_anomaly_consensus",        "Consensus (≥2)"),
        ("is_anomaly_all",              "All 3 methods"),
    ]:
        if col not in df.columns:
            continue
        a = df.loc[df[col] == 1, "relative_energy"]
        n = df.loc[df[col] == 0, "relative_energy"]
        diff = a.mean() - n.mean()
        interpretation = (
            "↑ kém ổn định hơn" if diff > 0.01
            else "↓ ổn định hơn (cấu trúc đặc biệt)" if diff < -0.01
            else "≈ tương đương"
        )
        rows.append({
            "Phương pháp": label,
            "N anomaly": len(a),
            "Energy anomaly (mean)": f"{a.mean():.4f}",
            "Energy normal (mean)":  f"{n.mean():.4f}",
            "Chênh lệch (eV/atom)":  f"{diff:+.4f}",
            "Diễn giải": interpretation,
        })
        print(f"  {label:25s}: diff={diff:+.4f} eV/atom  → {interpretation}")

    return pd.DataFrame(rows)


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_pca_comparison(df: pd.DataFrame):
    pca_cols = [c for c in df.columns if c.lower().startswith("pca")]
    if len(pca_cols) < 2:
        print("⚠️  Không đủ cột PCA để vẽ")
        return

    px, py = pca_cols[0], pca_cols[1]
    methods = [
        ("is_hdbscan_noise",            "HDBSCAN Noise"),
        ("is_low_probability",          "Low Probability"),
        ("is_isolation_forest_anomaly", "Isolation Forest"),
        ("is_anomaly_consensus",        "Consensus (≥2)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Anomaly Detection — PCA Space", fontsize=15, fontweight="bold")

    for ax, (col, title) in zip(axes.flat, methods):
        normal  = df[df[col] == 0]
        anomaly = df[df[col] == 1]
        ax.scatter(normal[px],  normal[py],  c="steelblue", alpha=0.2, s=8,  label="Normal")
        ax.scatter(anomaly[px], anomaly[py], c="crimson",   alpha=0.7, s=25, label="Anomaly",
                   edgecolors="darkred", linewidths=0.4)
        ax.set_title(f"{title}\n{len(anomaly):,} anomalies ({len(anomaly)/len(df):.1%})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(px); ax.set_ylabel(py)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    p = FIGURE_DIR / "anomaly_pca_comparison.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {p.name}")


def plot_energy_boxplot(df: pd.DataFrame):
    methods = [
        ("is_hdbscan_noise",            "HDBSCAN\nNoise"),
        ("is_low_probability",          "Low\nProbability"),
        ("is_isolation_forest_anomaly", "Isolation\nForest"),
        ("is_anomaly_consensus",        "Consensus\n(≥2)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle("Relative Energy: Anomaly vs Normal", fontsize=13, fontweight="bold")

    for ax, (col, label) in zip(axes, methods):
        data = [
            df.loc[df[col] == 0, "relative_energy"].dropna().values,
            df.loc[df[col] == 1, "relative_energy"].dropna().values,
        ]
        bp = ax.boxplot(data, tick_labels=["Normal", "Anomaly"],
                        patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("crimson")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Relative Energy (eV/atom)" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = FIGURE_DIR / "anomaly_energy_boxplot.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {p.name}")


def plot_vote_distribution(df: pd.DataFrame):
    vc = df["anomaly_vote_count"].value_counts().sort_index()
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(vc.index, vc.values, color=colors[:len(vc)], edgecolor="black", alpha=0.85)
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{v:,}\n({v/len(df):.1%})", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0\n(Normal)", "1 method", "2 methods", "3 methods"])
    ax.set_xlabel("Số phương pháp phát hiện", fontsize=11)
    ax.set_ylabel("Số mẫu", fontsize=11)
    ax.set_title("Mức độ đồng thuận giữa các phương pháp", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = FIGURE_DIR / "anomaly_vote_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {p.name}")


def plot_isolation_forest_score(df: pd.DataFrame):
    if "isolation_forest_score" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    normal  = df.loc[df["is_isolation_forest_anomaly"] == 0, "isolation_forest_score"]
    anomaly = df.loc[df["is_isolation_forest_anomaly"] == 1, "isolation_forest_score"]
    ax.hist(normal,  bins=60, alpha=0.55, color="steelblue", label="Normal",  density=True)
    ax.hist(anomaly, bins=60, alpha=0.55, color="crimson",   label="Anomaly", density=True)
    ax.axvline(normal.mean(),  color="steelblue", linestyle="--", lw=2, label=f"Normal mean={normal.mean():.3f}")
    ax.axvline(anomaly.mean(), color="crimson",   linestyle="--", lw=2, label=f"Anomaly mean={anomaly.mean():.3f}")
    ax.set_xlabel("Isolation Forest Score (càng âm = càng bất thường)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Isolation Forest Score Distribution", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = FIGURE_DIR / "isolation_forest_score_distribution.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✅ {p.name}")


def create_all_plots(df: pd.DataFrame):
    print("\n" + "="*70)
    print("TẠO BIỂU ĐỒ")
    print("="*70)
    plot_pca_comparison(df)
    plot_energy_boxplot(df)
    plot_vote_distribution(df)
    plot_isolation_forest_score(df)


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(df: pd.DataFrame, energy_summary: pd.DataFrame):
    print("\n" + "="*70)
    print("LƯU KẾT QUẢ")
    print("="*70)

    df.to_csv(ANOMALY_RESULTS_PATH, index=False)
    print(f"  ✅ Full results  : {ANOMALY_RESULTS_PATH.name}  {df.shape}")

    energy_summary.to_csv(ANOMALY_SUMMARY_PATH, index=False)
    print(f"  ✅ Energy summary: {ANOMALY_SUMMARY_PATH.name}")

    # Method comparison
    rows = []
    for col in ["is_hdbscan_noise", "is_low_probability",
                "is_isolation_forest_anomaly",
                "is_anomaly_consensus", "is_anomaly_any", "is_anomaly_all"]:
        if col in df.columns:
            n = int(df[col].sum())
            rows.append({"method": col, "n_anomalies": n,
                         "anomaly_ratio": n / len(df),
                         "n_normal": len(df) - n,
                         "normal_ratio": 1 - n / len(df)})
    pd.DataFrame(rows).to_csv(ANOMALY_COMPARISON_PATH, index=False)
    print(f"  ✅ Comparison    : {ANOMALY_COMPARISON_PATH.name}")

    # Consensus anomaly details
    if "is_anomaly_consensus" in df.columns:
        detail_cols = ["material_id", "anomaly_vote_count",
                       "is_hdbscan_noise", "is_low_probability",
                       "is_isolation_forest_anomaly",
                       "relative_energy", "energy",
                       "isolation_forest_score"]
        cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
        prob_col    = next((c for c in df.columns if "prob" in c.lower() and "hdbscan" in c.lower()), None)
        if cluster_col: detail_cols.append(cluster_col)
        if prob_col:    detail_cols.append(prob_col)

        avail = [c for c in detail_cols if c in df.columns]
        details = (df[df["is_anomaly_consensus"] == 1][avail]
                   .sort_values("anomaly_vote_count", ascending=False))
        details.to_csv(ANOMALY_DETAILS_PATH, index=False)
        print(f"  ✅ Details       : {ANOMALY_DETAILS_PATH.name}  ({len(details):,} consensus anomalies)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CARBON-24 ANOMALY DETECTION")
    print("="*70)
    print("Nguyên tắc: features cấu trúc → phát hiện → diễn giải bằng energy")

    ensure_output_dirs()

    # Load data
    df       = load_hdbscan_results()
    feat_df  = load_structure_features()

    # 3 phương pháp
    df = detect_hdbscan_noise(df)
    df = detect_low_probability(df, threshold=HDBSCAN_PROBABILITY_THRESHOLD)
    df = detect_isolation_forest(df, feat_df, contamination=ISOLATION_FOREST_CONTAMINATION)

    # Kết hợp
    df = combine_methods(df)

    # Diễn giải bằng energy
    energy_summary = analyze_energy(df)

    # Visualize
    create_all_plots(df)

    # Lưu
    save_results(df, energy_summary)

    # Tổng kết
    print("\n" + "="*70)
    print("✅ HOÀN TẤT")
    print("="*70)
    using_pca = df.get("isolation_forest_used_pca", pd.Series([0])).iloc[0]
    if using_pca:
        print("⚠️  Isolation Forest dùng PCA fallback — kết quả kém tin cậy hơn.")
        print("   Để cải thiện: đặt carbon24_features.csv vào thư mục gốc.")
    print(f"\n  HDBSCAN Noise    : {df['is_hdbscan_noise'].sum():,}  ({df['is_hdbscan_noise'].mean():.2%})")
    print(f"  Low Probability  : {df['is_low_probability'].sum():,}  ({df['is_low_probability'].mean():.2%})")
    print(f"  Isolation Forest : {df['is_isolation_forest_anomaly'].sum():,}  ({df['is_isolation_forest_anomaly'].mean():.2%})")
    print(f"  Consensus (≥2)   : {df['is_anomaly_consensus'].sum():,}  ({df['is_anomaly_consensus'].mean():.2%})")
    print(f"  All 3 methods    : {df['is_anomaly_all'].sum():,}  ({df['is_anomaly_all'].mean():.2%})")


if __name__ == "__main__":
    main()
