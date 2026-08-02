
"""
Carbon-24 Ground-Truth Visualization
======================================
Tao cac bieu do truc quan hoa ket qua so khop cluster voi Materials Project.

Figures:
  1. PCA scatter — mau theo scientific label
  2. Energy boxplot — so sanh cluster vs MP reference
  3. Space group heatmap — overlap matrix
  4. Crystal system comparison — cluster vs stable MP
  5. GMM micro-cluster profile heatmap (normalized)
  6. Diamond / Graphite / Amorphous distribution
  7. Summary dashboard figure (4-panel)
"""

from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

OUT   = Path("carbon24_pipeline_results")
FIGS  = OUT / "figures"
FIGS.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
gmm    = pd.read_csv(OUT / "tier3_gmm_labeled.csv")
labels = pd.read_csv(OUT / "ground_truth_labels.csv")
sci    = pd.read_csv(OUT / "cluster_scientific_names.csv")
ref    = pd.read_csv("carbon.csv").rename(columns={
    "Crystal System":    "crystal_system",
    "Space Group Symbol":"space_group_symbol",
    "Energy Above Hull": "e_above_hull",
    "Band Gap":          "band_gap",
    "Is Metal":          "is_metal",
})
noise  = pd.read_csv(OUT / "tier1_noise_analysis.csv")

print(f"Loaded: gmm={gmm.shape}, ref={ref.shape}")

# ── Color palettes ────────────────────────────────────────────────────────────
SCI_COLORS = {
    "Diamond-like (Fd-3m)":    "#1a9850",   # dark green
    "Graphite-like":           "#4393c3",   # blue
    "Layered Carbon (C2/m)":   "#74add1",   # light blue
    "Layered Carbon (sp2)":    "#74add1",
    "Mixed sp2/sp3 Carbon":    "#fee090",   # yellow
    "Amorphous Carbon":        "#d73027",   # red
    "High-energy Carbon":      "#a50026",   # dark red
    "Unclassified Carbon":     "#bababa",   # grey
}

def get_color(name):
    for k, v in SCI_COLORS.items():
        if k in str(name):
            return v
    return "#bababa"

gmm["sci_color"] = gmm["scientific_label"].apply(get_color)

# ============================================================================
# FIG 1: PCA SCATTER — Scientific Labels
# ============================================================================
print("Fig 1: PCA scatter — scientific labels...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Carbon-24 Cluster Scientific Labeling\n(Ground-truth from Materials Project)",
             fontsize=14, fontweight="bold")

# Panel A: K-means macro
ax = axes[0]
km_sci = {
    0: ("Mixed sp2/sp3 Carbon",    "#fee090"),
    1: ("High-energy Carbon",      "#a50026"),
    2: ("Diamond-like + Layered",  "#1a9850"),
}
for cid, (name, color) in km_sci.items():
    mask = gmm["kmeans_cluster"] == cid
    ax.scatter(gmm.loc[mask, "pca1"], gmm.loc[mask, "pca2"],
               c=color, alpha=0.25, s=8, label=f"K{cid}: {name} ({mask.sum():,})")
# Noise overlay
ax.scatter(noise["pca1"], noise["pca2"],
           c="black", alpha=0.6, s=12, marker="x", label=f"Noise ({len(noise):,})", zorder=5)
ax.set_xlabel("PCA1", fontsize=11); ax.set_ylabel("PCA2", fontsize=11)
ax.set_title("K-means Macro-clusters (k=3)", fontweight="bold")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)

# Panel B: GMM micro — scientific name
ax = axes[1]
sci_groups = gmm.groupby("scientific_label")
for name, grp in sci_groups:
    color = get_color(name)
    ax.scatter(grp["pca1"], grp["pca2"],
               c=color, alpha=0.3, s=8, label=f"{name} ({len(grp):,})")
ax.scatter(noise["pca1"], noise["pca2"],
           c="black", alpha=0.6, s=12, marker="x", label=f"Noise ({len(noise):,})", zorder=5)
ax.set_xlabel("PCA1", fontsize=11); ax.set_ylabel("PCA2", fontsize=11)
ax.set_title("GMM Micro-clusters — Scientific Labels", fontweight="bold")
ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / "gt_pca_scientific_labels.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gt_pca_scientific_labels.png")

# ============================================================================
# FIG 2: ENERGY BOXPLOT — Cluster vs MP Reference
# ============================================================================
print("Fig 2: Energy boxplot...")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Energy Comparison: Clusters vs Materials Project Reference",
             fontsize=13, fontweight="bold")

# Panel A: K-means energy vs MP stability classes
ax = axes[0]
km_order = ["Ben vung (Low Energy)", "Trung gian", "Phuc tap (High Energy)"]
km_colors = ["#1a9850", "#fee090", "#a50026"]
data_km = [gmm[gmm["kmeans_label"] == lbl]["relative_energy"].dropna().values
           for lbl in km_order]
bp = ax.boxplot(data_km, tick_labels=km_order, patch_artist=True, notch=False,
                medianprops=dict(color="black", linewidth=2.5))
for patch, color in zip(bp["boxes"], km_colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)

# Them duong tham chieu tu MP
stable_e  = ref[ref["e_above_hull"] < 0.05]["e_above_hull"].mean()
meta_e    = ref[(ref["e_above_hull"] >= 0.05) & (ref["e_above_hull"] < 0.3)]["e_above_hull"].mean()
# Chuyen doi: relative_energy trong carbon24 ~ e_above_hull trong MP (ca hai do tu min)
ax.axhline(0.05,  color="green",  linestyle="--", lw=1.5, alpha=0.7, label="MP Stable threshold (0.05)")
ax.axhline(0.30,  color="orange", linestyle="--", lw=1.5, alpha=0.7, label="MP Metastable threshold (0.30)")
ax.set_ylabel("Relative Energy (eV/atom)", fontsize=11)
ax.set_title("K-means Macro-clusters", fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# Panel B: GMM energy sorted
ax = axes[1]
gmm_order = labels.sort_values("energy_mean")["gmm_cluster"].tolist()
gmm_sci_names = labels.set_index("gmm_cluster")["scientific_name"].to_dict() if "scientific_name" in labels.columns else labels.set_index("gmm_cluster")["scientific_label"].to_dict() if "scientific_label" in labels.columns else {}
gmm_colors_list = [get_color(gmm_sci_names.get(c, "")) for c in gmm_order]
data_gmm = [gmm[gmm["gmm_cluster"] == c]["relative_energy"].dropna().values
            for c in gmm_order]
tick_labels_gmm = [f"GMM-{c}\n{gmm_sci_names.get(c,'')[:18]}" for c in gmm_order]
bp2 = ax.boxplot(data_gmm, tick_labels=tick_labels_gmm, patch_artist=True, notch=False,
                 medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp2["boxes"], gmm_colors_list):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.axhline(0.05,  color="green",  linestyle="--", lw=1.5, alpha=0.7)
ax.axhline(0.30,  color="orange", linestyle="--", lw=1.5, alpha=0.7)
ax.set_ylabel("Relative Energy (eV/atom)", fontsize=11)
ax.set_title("GMM Micro-clusters (sorted by energy)", fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

plt.tight_layout()
plt.savefig(FIGS / "gt_energy_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gt_energy_boxplot.png")

# ============================================================================
# FIG 3: SPACE GROUP OVERLAP HEATMAP
# ============================================================================
print("Fig 3: Space group overlap heatmap...")

# Top space groups trong stable MP
stable_sgs = ref[ref["e_above_hull"] < 0.05]["space_group_symbol"].value_counts().head(10).index.tolist()

# Tinh phan tram moi SG trong moi GMM cluster
sg_matrix = pd.DataFrame(index=sorted(gmm["gmm_cluster"].unique()),
                          columns=stable_sgs, dtype=float)
for cid in sg_matrix.index:
    sub = gmm[gmm["gmm_cluster"] == cid]
    total = len(sub)
    for sg in stable_sgs:
        sg_matrix.loc[cid, sg] = (sub["space_group_symbol"] == sg).sum() / total * 100

sg_matrix = sg_matrix.astype(float)
sg_matrix.index = [f"GMM-{i}" for i in sg_matrix.index]

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(sg_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "% of cluster samples"},
            vmin=0)
ax.set_title("Space Group Overlap: GMM Clusters vs Stable MP Structures (%)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Space Group (present in stable MP structures)", fontsize=11)
ax.set_ylabel("GMM Cluster", fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "gt_spacegroup_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gt_spacegroup_heatmap.png")

# ============================================================================
# FIG 4: CRYSTAL SYSTEM COMPARISON — Cluster vs MP
# ============================================================================
print("Fig 4: Crystal system comparison...")

cs_order = ["monoclinic", "triclinic", "orthorhombic", "hexagonal",
            "trigonal", "cubic", "tetragonal"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Crystal System Distribution: Clusters vs Materials Project",
             fontsize=14, fontweight="bold")

# MP reference
ax = axes[0, 0]
mp_cs = ref["crystal_system"].str.lower().value_counts(normalize=True) * 100
mp_cs = mp_cs.reindex(cs_order, fill_value=0)
bars = ax.bar(cs_order, mp_cs.values, color="#4393c3", alpha=0.8, edgecolor="black")
for bar, v in zip(bars, mp_cs.values):
    if v > 1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_title("Materials Project Reference (64 samples)", fontweight="bold")
ax.set_ylabel("Percentage (%)"); ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# K-means clusters
km_configs = [
    (1, "Ben vung (Low Energy)",   "#1a9850", axes[0, 1]),
    (2, "Trung gian",              "#fee090", axes[1, 0]),
    (3, "Phuc tap (High Energy)",  "#a50026", axes[1, 1]),
]
for km_id, (_, lbl, color, ax) in enumerate(km_configs):
    sub = gmm[gmm["kmeans_label"] == lbl]
    cs  = sub["crystal_system"].str.lower().value_counts(normalize=True) * 100
    cs  = cs.reindex(cs_order, fill_value=0)
    bars = ax.bar(cs_order, cs.values, color=color, alpha=0.8, edgecolor="black")
    # MP overlay (line)
    ax.plot(range(len(cs_order)), mp_cs.values, "b--o", markersize=5,
            linewidth=1.5, label="MP Reference", alpha=0.7)
    for bar, v in zip(bars, cs.values):
        if v > 2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_title(f"K-means: {lbl} (n={len(sub):,})", fontweight="bold")
    ax.set_ylabel("Percentage (%)"); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig(FIGS / "gt_crystal_system_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gt_crystal_system_comparison.png")

# ============================================================================
# FIG 5: DIAMOND / GRAPHITE / AMORPHOUS DISTRIBUTION IN PCA
# ============================================================================
print("Fig 5: Diamond / Graphite / Amorphous PCA...")

# Phan loai theo space group
gmm["structure_type"] = "Other"
gmm.loc[gmm["space_group_symbol"] == "Fd-3m",                          "structure_type"] = "Diamond (Fd-3m)"
gmm.loc[gmm["space_group_symbol"].isin(["P6_3/mmc", "R-3m", "P6/mmm"]),"structure_type"] = "Graphite/Graphene"
gmm.loc[gmm["space_group_symbol"] == "C2/m",                           "structure_type"] = "Layered C2/m"
gmm.loc[gmm["space_group_symbol"].isin(["P1", "P-1"]),                 "structure_type"] = "Amorphous (P1/P-1)"

type_colors = {
    "Diamond (Fd-3m)":    "#1a9850",
    "Graphite/Graphene":  "#4393c3",
    "Layered C2/m":       "#74add1",
    "Amorphous (P1/P-1)": "#d73027",
    "Other":              "#bababa",
}
type_order = ["Diamond (Fd-3m)", "Graphite/Graphene", "Layered C2/m",
              "Amorphous (P1/P-1)", "Other"]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Structure Type Distribution in PCA Space\n(Identified via Space Group Matching)",
             fontsize=13, fontweight="bold")

# Panel A: PCA scatter
ax = axes[0]
for stype in type_order:
    mask = gmm["structure_type"] == stype
    if mask.sum() == 0:
        continue
    alpha = 0.7 if stype != "Other" else 0.15
    size  = 20  if stype != "Other" else 6
    ax.scatter(gmm.loc[mask, "pca1"], gmm.loc[mask, "pca2"],
               c=type_colors[stype], alpha=alpha, s=size,
               label=f"{stype} ({mask.sum():,})",
               edgecolors="none" if stype == "Other" else type_colors[stype],
               zorder=3 if stype != "Other" else 1)
ax.scatter(noise["pca1"], noise["pca2"],
           c="black", alpha=0.5, s=10, marker="x",
           label=f"Noise ({len(noise):,})", zorder=5)
ax.set_xlabel("PCA1", fontsize=11); ax.set_ylabel("PCA2", fontsize=11)
ax.set_title("Structure Types in PCA Space", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel B: Stacked bar — structure type per K-means cluster
ax = axes[1]
km_labels_order = ["Ben vung (Low Energy)", "Trung gian", "Phuc tap (High Energy)"]
bottom = np.zeros(3)
for stype in type_order:
    vals = []
    for lbl in km_labels_order:
        sub = gmm[gmm["kmeans_label"] == lbl]
        pct = (sub["structure_type"] == stype).sum() / len(sub) * 100
        vals.append(pct)
    ax.bar(km_labels_order, vals, bottom=bottom,
           color=type_colors[stype], label=stype, alpha=0.85, edgecolor="white")
    # Label if > 3%
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 3:
            ax.text(i, b + v/2, f"{v:.1f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
    bottom += np.array(vals)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_title("Structure Type Composition per K-means Cluster", fontweight="bold")
ax.legend(fontsize=8, loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

plt.tight_layout()
plt.savefig(FIGS / "gt_structure_type_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gt_structure_type_distribution.png")

# ============================================================================
# FIG 6: SUMMARY DASHBOARD — 4 panels
# ============================================================================
print("Fig 6: Summary dashboard...")

fig = plt.figure(figsize=(20, 16))
fig.suptitle("Carbon-24 Ground-Truth Labeling Summary\n"
             "Reference: Materials Project (64 Carbon structures)",
             fontsize=15, fontweight="bold", y=0.98)

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# ── Panel 1: PCA với scientific labels ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for stype in type_order:
    mask = gmm["structure_type"] == stype
    if mask.sum() == 0:
        continue
    ax1.scatter(gmm.loc[mask, "pca1"], gmm.loc[mask, "pca2"],
                c=type_colors[stype],
                alpha=0.6 if stype != "Other" else 0.12,
                s=15 if stype != "Other" else 5,
                label=f"{stype} ({mask.sum():,})", zorder=3 if stype != "Other" else 1)
ax1.scatter(noise["pca1"], noise["pca2"],
            c="black", alpha=0.5, s=10, marker="x",
            label=f"Noise ({len(noise):,})", zorder=5)
ax1.set_xlabel("PCA1"); ax1.set_ylabel("PCA2")
ax1.set_title("PCA Space — Structure Types", fontweight="bold")
ax1.legend(fontsize=7, ncol=2); ax1.grid(alpha=0.3)

# ── Panel 2: Pie chart structure types ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
type_counts = gmm["structure_type"].value_counts()
colors_pie  = [type_colors.get(t, "#bababa") for t in type_counts.index]
wedges, texts, autotexts = ax2.pie(
    type_counts.values, labels=None,
    colors=colors_pie, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=1.5),
)
for at in autotexts:
    at.set_fontsize(8)
ax2.legend(wedges, type_counts.index, fontsize=7, loc="lower center",
           bbox_to_anchor=(0.5, -0.25), ncol=1)
ax2.set_title("Structure Type\nDistribution", fontweight="bold")

# ── Panel 3: Energy violin per scientific name ───────────────────────────────
ax3 = fig.add_subplot(gs[1, :])
sci_order_e = (gmm.groupby("scientific_label")["relative_energy"]
               .mean().sort_values().index.tolist())
data_violin = [gmm[gmm["scientific_label"] == s]["relative_energy"].dropna().values
               for s in sci_order_e]
colors_violin = [get_color(s) for s in sci_order_e]
parts = ax3.violinplot(data_violin, positions=range(len(sci_order_e)),
                       showmedians=True, showextrema=True)
for i, (pc, color) in enumerate(zip(parts["bodies"], colors_violin)):
    pc.set_facecolor(color); pc.set_alpha(0.7)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
# MP reference lines
ax3.axhline(0.05, color="green",  linestyle="--", lw=1.5, alpha=0.8,
            label="MP Stable (E<0.05)")
ax3.axhline(0.30, color="orange", linestyle="--", lw=1.5, alpha=0.8,
            label="MP Metastable (E<0.30)")
ax3.set_xticks(range(len(sci_order_e)))
ax3.set_xticklabels(sci_order_e, rotation=20, ha="right", fontsize=9)
ax3.set_ylabel("Relative Energy (eV/atom)", fontsize=11)
ax3.set_title("Energy Distribution per Scientific Label (vs MP Thresholds)",
              fontweight="bold")
ax3.legend(fontsize=9); ax3.grid(axis="y", alpha=0.3)

# ── Panel 4: Space group overlap bar ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :2])
top_sgs_stable = (ref[ref["e_above_hull"] < 0.05]["space_group_symbol"]
                  .value_counts().head(8).index.tolist())
sg_overlap_data = {}
for lbl in ["Ben vung (Low Energy)", "Trung gian", "Phuc tap (High Energy)"]:
    sub = gmm[gmm["kmeans_label"] == lbl]
    sg_overlap_data[lbl] = [(sub["space_group_symbol"] == sg).sum() / len(sub) * 100
                             for sg in top_sgs_stable]

x = np.arange(len(top_sgs_stable))
w = 0.25
km_colors_bar = ["#1a9850", "#fee090", "#a50026"]
for i, (lbl, vals) in enumerate(sg_overlap_data.items()):
    ax4.bar(x + i*w, vals, w, label=lbl,
            color=km_colors_bar[i], alpha=0.8, edgecolor="black")
ax4.set_xticks(x + w)
ax4.set_xticklabels(top_sgs_stable, rotation=30, ha="right", fontsize=9)
ax4.set_ylabel("% of cluster samples", fontsize=10)
ax4.set_title("Space Group Overlap with Stable MP Structures",
              fontweight="bold")
ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.3)

# ── Panel 5: Key findings text ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
ax5.axis("off")
n_diamond  = (gmm["structure_type"] == "Diamond (Fd-3m)").sum()
n_graphite = (gmm["structure_type"] == "Graphite/Graphene").sum()
n_layered  = (gmm["structure_type"] == "Layered C2/m").sum()
n_amorph   = (gmm["structure_type"] == "Amorphous (P1/P-1)").sum()
n_noise_t  = len(noise)

findings = (
    "KEY FINDINGS\n"
    "─────────────────────\n"
    f"Diamond (Fd-3m):\n  {n_diamond:,} samples\n"
    f"  Energy ≈ 0.011 eV/atom\n"
    f"  → Ổn định nhất\n\n"
    f"Graphite/Graphene:\n  {n_graphite:,} samples\n"
    f"  → sp2 layered\n\n"
    f"Layered C2/m:\n  {n_layered:,} samples\n"
    f"  → Mixed sp2/sp3\n\n"
    f"Amorphous:\n  {n_amorph:,} samples\n"
    f"  → Disordered\n\n"
    f"Noise (anomalies):\n  {n_noise_t:,} samples\n"
    f"  → Kém ổn định\n"
    f"  +0.074 eV/atom"
)
ax5.text(0.05, 0.95, findings, transform=ax5.transAxes,
         fontsize=9, verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.savefig(FIGS / "gt_summary_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gt_summary_dashboard.png")

# ============================================================================
# FIG 7: GMM PROFILE HEATMAP (normalized) with scientific labels
# ============================================================================
print("Fig 7: GMM profile heatmap with scientific labels...")

num_cols = ["relative_energy", "volume_per_atom", "density",
            "mean_bond_length", "std_coordination", "angle_deviation", "num_atoms"]
num_cols = [c for c in num_cols if c in gmm.columns]

profile = gmm.groupby("gmm_cluster")[num_cols].mean()
profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

# Them scientific label vao index
sci_map = labels.set_index("gmm_cluster")["scientific_name"].to_dict() if "scientific_name" in labels.columns else labels.set_index("gmm_cluster")["scientific_label"].to_dict() if "scientific_label" in labels.columns else {}
profile_norm.index = [f"GMM-{i}\n{sci_map.get(i,'')[:20]}" for i in profile_norm.index]

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(profile_norm, annot=True, fmt=".2f", cmap="RdYlGn_r",
            linewidths=0.8, ax=ax,
            cbar_kws={"label": "Normalized value (0=min, 1=max)"},
            vmin=0, vmax=1)
ax.set_title("GMM Micro-cluster Profile — Scientific Labels\n"
             "(Normalized: 0=lowest, 1=highest within each feature)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Structural Feature", fontsize=11)
ax.set_ylabel("GMM Cluster (Scientific Label)", fontsize=11)
plt.tight_layout()
plt.savefig(FIGS / "gt_gmm_profile_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: gt_gmm_profile_heatmap.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 65)
print("ALL FIGURES SAVED:")
for f in sorted(FIGS.glob("gt_*.png")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size_kb:>7.1f} KB")
print("=" * 65)
