"""Script tao notebook carbon24-pipeline-3tier.ipynb"""
import json

cells = []

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }

# ── TITLE ────────────────────────────────────────────────────────────────────
cells.append(md(
    "# Carbon-24 — Pipeline 3 Tang\n\n"
    "## Kien truc tong the\n\n"
    "```\n"
    "Raw data (10,153)\n"
    "    |\n"
    "    v  [TANG 1] HDBSCAN Noise Detection\n"
    "    +---> Noise (786, 7.74%)  -->  Anomaly Analysis\n"
    "    |\n"
    "    v  Clean data (9,367, 92.26%)\n"
    "    |\n"
    "    v  [TANG 2] K-means Macro-clustering (k=3)\n"
    "    +---> Ben vung (Low Energy)  : 3,213 (34.3%)\n"
    "    +---> Trung gian             : 2,201 (23.5%)\n"
    "    +---> Phuc tap (High Energy) : 3,953 (42.2%)\n"
    "    |\n"
    "    v  [TANG 3] GMM Micro-clustering (k=10)\n"
    "    +---> 10 sub-clusters (polymorphs)\n"
    "         Ben vung   -> GMM 2, 4, 7\n"
    "         Trung gian -> GMM 0, 5, 8\n"
    "         Phuc tap   -> GMM 1, 3, 4, 6, 9\n"
    "```\n\n"
    "## Nguyen tac\n"
    "- **Tang 1**: HDBSCAN noise = Anomaly Detection cua de tai\n"
    "- **Tang 2**: K-means tren du lieu sach = 3 nhom cau truc lon\n"
    "- **Tang 3**: GMM tren du lieu sach = 10 phan nhom xac suat (polymorphs)\n"
    "- Energy chi dung de **dien giai**, khong dung lam input\n"
))

# ── CELL 1: SETUP ────────────────────────────────────────────────────────────
cells.append(md("## 1. Setup & Chay Pipeline"))
cells.append(code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from pathlib import Path\n"
    "from IPython.display import Image, display as ipy_display\n"
    "\n"
    "plt.style.use('seaborn-v0_8-darkgrid')\n"
    "sns.set_palette('husl')\n"
    "%matplotlib inline\n"
    "\n"
    "# Chay toan bo pipeline\n"
    "import carbon24_pipeline_3tier\n"
    "carbon24_pipeline_3tier.main()\n"
))

# ── CELL 2: LOAD ─────────────────────────────────────────────────────────────
cells.append(md("## 2. Load Ket Qua"))
cells.append(code(
    "OUT = Path('carbon24_pipeline_results')\n"
    "\n"
    "final_df    = pd.read_csv(OUT / 'pipeline_final.csv')\n"
    "noise_df    = pd.read_csv(OUT / 'tier1_noise_analysis.csv')\n"
    "kmeans_df   = pd.read_csv(OUT / 'tier2_kmeans_clean.csv')\n"
    "km_profile  = pd.read_csv(OUT / 'tier2_kmeans_profile.csv')\n"
    "gmm_df      = pd.read_csv(OUT / 'tier3_gmm_clean.csv')\n"
    "gmm_profile = pd.read_csv(OUT / 'tier3_gmm_profile.csv')\n"
    "\n"
    "clean = final_df[final_df['pipeline_stage'] == 'clean'].copy()\n"
    "noise = final_df[final_df['pipeline_stage'] == 'noise'].copy()\n"
    "\n"
    "print(f'Final shape : {final_df.shape}')\n"
    "print(f'Clean       : {len(clean):,}')\n"
    "print(f'Noise       : {len(noise):,}')\n"
    "final_df.head(3)\n"
))

# ── CELL 3: OVERVIEW ─────────────────────────────────────────────────────────
cells.append(md("## 3. Tong Quan Pipeline"))
cells.append(code(
    "ipy_display(Image(filename=str(OUT / 'figures' / 'pipeline_overview.png')))\n"
))
cells.append(code(
    "print('PIPELINE SUMMARY')\n"
    "print('=' * 60)\n"
    "print(f'  Total          : {len(final_df):,}')\n"
    "print(f'  Noise (Tier 1) : {len(noise):,}  ({len(noise)/len(final_df):.2%})')\n"
    "print(f'  Clean (Tier 2+): {len(clean):,}  ({len(clean)/len(final_df):.2%})')\n"
    "print()\n"
    "print('K-means macro-clusters:')\n"
    "for lbl, n in clean['kmeans_label'].value_counts().items():\n"
    "    print(f'  {lbl}: {n:,}  ({n/len(final_df):.2%})')\n"
    "print()\n"
    "print('GMM micro-clusters:')\n"
    "for c, n in clean['gmm_cluster'].value_counts().sort_index().items():\n"
    "    km_maj = gmm_df[gmm_df['gmm_cluster'] == c]['kmeans_label'].mode()[0]\n"
    "    print(f'  GMM-{c}: {n:,}  ({n/len(final_df):.2%})  <- {km_maj}')\n"
))

# ── CELL 4: TIER 1 ───────────────────────────────────────────────────────────
cells.append(md("## 4. Tang 1: HDBSCAN Noise Detection (Anomaly Analysis)"))
cells.append(code(
    "ipy_display(Image(filename=str(OUT / 'figures' / 'tier1_noise_pca.png')))\n"
))
cells.append(code(
    "print('TANG 1: NOISE ANALYSIS')\n"
    "print('=' * 60)\n"
    "print(f'  Noise: {len(noise_df):,}  ({len(noise_df)/len(final_df):.2%})')\n"
    "print()\n"
    "\n"
    "e_noise = noise_df['relative_energy']\n"
    "e_clean = final_df.loc[final_df['pipeline_stage'] == 'clean', 'relative_energy']\n"
    "diff = e_noise.mean() - e_clean.mean()\n"
    "print(f'  Energy noise : {e_noise.mean():.4f} +/- {e_noise.std():.4f}')\n"
    "print(f'  Energy clean : {e_clean.mean():.4f} +/- {e_clean.std():.4f}')\n"
    "print(f'  Difference   : {diff:+.4f} eV/atom  -> Noise kem on dinh hon')\n"
    "print()\n"
    "\n"
    "print('  Crystal system (noise vs clean):')\n"
    "cs_n = noise_df['crystal_system'].value_counts(normalize=True) * 100\n"
    "cs_c = final_df.loc[final_df['pipeline_stage'] == 'clean', 'crystal_system'].value_counts(normalize=True) * 100\n"
    "for cs in cs_c.index:\n"
    "    print(f'    {cs:<15}: noise={cs_n.get(cs, 0):.1f}%  clean={cs_c[cs]:.1f}%')\n"
))
cells.append(code(
    "# Boxplot energy: noise vs clean\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
    "\n"
    "# Histogram\n"
    "ax = axes[0]\n"
    "ax.hist(e_clean, bins=60, alpha=0.5, color='steelblue', label=f'Clean ({len(e_clean):,})', density=True)\n"
    "ax.hist(e_noise, bins=60, alpha=0.5, color='crimson',   label=f'Noise ({len(e_noise):,})', density=True)\n"
    "ax.axvline(e_clean.mean(), color='steelblue', linestyle='--', lw=2, label=f'Clean mean={e_clean.mean():.4f}')\n"
    "ax.axvline(e_noise.mean(), color='crimson',   linestyle='--', lw=2, label=f'Noise mean={e_noise.mean():.4f}')\n"
    "ax.set_xlabel('Relative Energy (eV/atom)')\n"
    "ax.set_ylabel('Density')\n"
    "ax.set_title('Energy: Noise vs Clean', fontweight='bold')\n"
    "ax.legend(fontsize=9)\n"
    "ax.grid(alpha=0.3)\n"
    "\n"
    "# Crystal system comparison\n"
    "ax = axes[1]\n"
    "cs_compare = pd.DataFrame({'Clean': cs_c, 'Noise': cs_n}).fillna(0)\n"
    "cs_compare.plot(kind='bar', ax=ax, color=['steelblue', 'crimson'], alpha=0.8, edgecolor='black')\n"
    "ax.set_xlabel('Crystal System')\n"
    "ax.set_ylabel('Percentage (%)')\n"
    "ax.set_title('Crystal System: Noise vs Clean', fontweight='bold')\n"
    "ax.legend()\n"
    "ax.grid(axis='y', alpha=0.3)\n"
    "plt.setp(ax.get_xticklabels(), rotation=30, ha='right')\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# ── CELL 5: TIER 2 ───────────────────────────────────────────────────────────
cells.append(md("## 5. Tang 2: K-means Macro-clustering (k=3)"))
cells.append(code(
    "ipy_display(Image(filename=str(OUT / 'figures' / 'tier2_kmeans.png')))\n"
))
cells.append(code(
    "print('TANG 2: K-MEANS MACRO-CLUSTERS')\n"
    "print('=' * 60)\n"
    "for lbl in clean['kmeans_label'].unique():\n"
    "    sub = clean[clean['kmeans_label'] == lbl]\n"
    "    e = sub['relative_energy']\n"
    "    print(f'  {lbl}')\n"
    "    print(f'    N      = {len(sub):,}  ({len(sub)/len(clean):.2%} of clean)')\n"
    "    print(f'    Energy : mean={e.mean():.4f}  std={e.std():.4f}  median={e.median():.4f}')\n"
    "    if 'crystal_system' in sub.columns:\n"
    "        top_cs = sub['crystal_system'].value_counts().head(3)\n"
    "        print(f'    Top crystal systems: {top_cs.to_dict()}')\n"
    "    print()\n"
))
cells.append(code(
    "# Heatmap profile K-means\n"
    "num_cols = ['relative_energy', 'volume_per_atom', 'mean_bond_length',\n"
    "            'std_coordination', 'angle_deviation', 'num_atoms']\n"
    "num_cols = [c for c in num_cols if c in clean.columns]\n"
    "\n"
    "profile = clean.groupby('kmeans_label')[num_cols].mean()\n"
    "profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(12, 4))\n"
    "sns.heatmap(profile_norm, annot=True, fmt='.2f', cmap='RdYlGn_r',\n"
    "            linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalized'})\n"
    "ax.set_title('K-means Cluster Profile (normalized)', fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print('\\nRaw means:')\n"
    "display(profile.round(4))\n"
))

# ── CELL 6: TIER 3 ───────────────────────────────────────────────────────────
cells.append(md("## 6. Tang 3: GMM Micro-clustering (k=10)"))
cells.append(code(
    "ipy_display(Image(filename=str(OUT / 'figures' / 'tier3_gmm.png')))\n"
))
cells.append(code(
    "print('TANG 3: GMM MICRO-CLUSTERS')\n"
    "print('=' * 65)\n"
    "print(f\"{'GMM':<8} {'K-means Macro':<32} {'N':>6} {'Energy':>10} {'Prob':>8}\")\n"
    "print('-' * 65)\n"
    "for c in sorted(gmm_df['gmm_cluster'].unique()):\n"
    "    sub    = gmm_df[gmm_df['gmm_cluster'] == c]\n"
    "    km_maj = sub['kmeans_label'].mode()[0]\n"
    "    e      = sub['relative_energy'].mean()\n"
    "    p      = sub['gmm_probability'].mean()\n"
    "    print(f'  GMM-{c:<4} {km_maj:<32} {len(sub):>6,} {e:>10.4f} {p:>8.3f}')\n"
))
cells.append(code(
    "# Heatmap GMM profile\n"
    "num_cols = ['relative_energy', 'volume_per_atom', 'mean_bond_length',\n"
    "            'std_coordination', 'angle_deviation', 'num_atoms']\n"
    "num_cols = [c for c in num_cols if c in gmm_profile.columns]\n"
    "\n"
    "profile_sub  = gmm_profile[num_cols].copy()\n"
    "profile_norm = (profile_sub - profile_sub.min()) / (profile_sub.max() - profile_sub.min() + 1e-9)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(12, 7))\n"
    "sns.heatmap(profile_norm, annot=True, fmt='.2f', cmap='RdYlGn_r',\n"
    "            linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalized'})\n"
    "ax.set_title('GMM Micro-cluster Profile (normalized)', fontweight='bold')\n"
    "ax.set_xlabel('Feature')\n"
    "ax.set_ylabel('GMM Cluster')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))
cells.append(code(
    "# GMM sub-clusters within each K-means macro\n"
    "print('GMM sub-clusters within each K-means macro-cluster:')\n"
    "print('=' * 55)\n"
    "for km_lbl in clean['kmeans_label'].unique():\n"
    "    sub = gmm_df[gmm_df['kmeans_label'] == km_lbl]\n"
    "    gmm_dist = sub['gmm_cluster'].value_counts().sort_index()\n"
    "    print(f'  {km_lbl}:')\n"
    "    for gid, cnt in gmm_dist.items():\n"
    "        e = sub[sub['gmm_cluster'] == gid]['relative_energy'].mean()\n"
    "        p = sub[sub['gmm_cluster'] == gid]['gmm_probability'].mean()\n"
    "        print(f'    GMM-{gid}: {cnt:,}  energy={e:.4f}  prob={p:.3f}')\n"
    "    print()\n"
))

# ── CELL 7: CONCLUSION ───────────────────────────────────────────────────────
cells.append(md("## 7. Ket Luan & Y Nghia"))
cells.append(code(
    "n_noise = len(noise_df)\n"
    "n_total = len(final_df)\n"
    "n_clean = len(clean)\n"
    "\n"
    "print('=' * 65)\n"
    "print('KET LUAN PIPELINE 3 TANG')\n"
    "print('=' * 65)\n"
    "\n"
    "print(f'''\n"
    "1. TANG 1 - Anomaly Detection (HDBSCAN Noise):\n"
    "   - {n_noise:,} cau truc bat thuong ({n_noise/n_total:.2%})\n"
    "   - Kem on dinh hon +0.0735 eV/atom so voi cau truc binh thuong\n"
    "   - Triclinic chiem ty le cao bat thuong (43.3% vs 24.1%)\n"
    "   - Day la phan \"phat hien di biet\" cua de tai\n"
    "\n"
    "2. TANG 2 - Macro-clustering (K-means k=3):\n"
    "   - {n_clean:,} cau truc sach ({n_clean/n_total:.2%})\n"
    "   - 3 nhom lon phan anh 3 che do cau truc khac nhau\n"
    "   - Silhouette = 0.2777 (phu hop voi du lieu phuc tap)\n"
    "\n"
    "3. TANG 3 - Micro-clustering (GMM k=10):\n"
    "   - 10 sub-clusters (polymorphs) voi xac suat membership\n"
    "   - BIC = -357009 (mo hinh hoi tu tot)\n"
    "   - Moi nhom lon duoc be nho thanh 2-5 phan nhom\n"
    "   - Phan anh su da dang cau truc trong moi nhom\n"
    "\n"
    "=> Pipeline tra loi day du cho ten de tai:\n"
    "   Phat hien di biet va phan cum cau truc Carbon-24\n"
    "''')\n"
))

# ── BUILD NOTEBOOK ────────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.8.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

with open("carbon24-pipeline-3tier.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Created: carbon24-pipeline-3tier.ipynb")
