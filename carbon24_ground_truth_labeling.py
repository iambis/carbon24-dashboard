"""
Carbon-24 Ground-Truth Labeling
================================
So khop cac cum K-means / GMM voi du lieu chuan Materials Project (carbon.csv)
de gan nhan khoa hoc chinh xac cho tung cum.

Phuong phap so khop:
1. Space Group Symbol overlap truc tiep voi stable MP structures
2. Crystal System distribution similarity
3. Energy range matching
4. Band Gap / Is Metal prediction tu reference

Output:
    carbon24_pipeline_results/ground_truth_labels.csv
    carbon24_pipeline_results/cluster_scientific_names.csv
    carbon24_pipeline_results/matching_report.txt
    carbon24_pipeline_results/tier3_gmm_labeled.csv
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("carbon24_pipeline_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 65)
print("GROUND-TRUTH LABELING — Materials Project Reference")
print("=" * 65)

ref = pd.read_csv("carbon.csv")
gmm = pd.read_csv(OUTPUT_DIR / "tier3_gmm_clean.csv")

ref = ref.rename(columns={
    "Crystal System":    "crystal_system",
    "Space Group Symbol":"space_group_symbol",
    "Energy Above Hull": "e_above_hull",
    "Band Gap":          "band_gap",
    "Is Metal":          "is_metal",
    "Volume":            "volume",
    "Sites":             "sites",
    "Density":           "density",
})
ref["volume_per_atom"] = ref["volume"] / ref["sites"]

print(f"\nReference (Materials Project): {len(ref)} samples")
print(f"Pipeline data (clean):         {len(gmm)} samples")

# ============================================================================
# 2. PHAN LOAI REFERENCE
# ============================================================================

ref["stability_class"] = pd.cut(
    ref["e_above_hull"],
    bins=[-0.001, 0.05, 0.30, 1.0, 99],
    labels=["Stable", "Metastable", "Unstable", "Highly Unstable"],
)
ref["electronic_class"] = ref.apply(
    lambda r: "Metal" if r["is_metal"]
    else ("Insulator" if r["band_gap"] > 1.0
          else ("Semiconductor" if r["band_gap"] > 0.1
                else "Semi-metal")),
    axis=1,
)

# Stable reference (E < 0.05)
stable_ref = ref[ref["e_above_hull"] < 0.05].copy()
stable_sgs = set(stable_ref["space_group_symbol"].unique())

# ============================================================================
# 3. SPACE GROUP — CARBON STRUCTURE MAPPING (tu Materials Project)
# ============================================================================

# Mapping space group -> ten cau truc Carbon noi tieng
SG_TO_STRUCTURE = {
    # Diamond & Diamond-like
    "Fd-3m":    ("Diamond",          "sp3, tetrahedral, insulator, E_hull~0"),
    "Fd-3":     ("Diamond-like",     "sp3 cubic, insulator"),
    "Pa-3":     ("Pyrite-C",         "sp3 cubic, semiconductor"),
    # Graphite & Graphene
    "P6_3/mmc": ("Graphite (2H)",    "sp2 layered, semi-metal, E_hull~0"),
    "R-3m":     ("Graphite (3R)",    "sp2 rhombohedral, semi-metal, E_hull~0"),
    "P6/mmm":   ("Graphene/Graphite","sp2 hexagonal, semi-metal, E_hull~0"),
    "P-3m1":    ("Graphene-like",    "sp2 trigonal, semi-metal"),
    # Lonsdaleite
    "P6_3/mmc": ("Lonsdaleite",      "hexagonal diamond, sp3, high-pressure"),
    # Layered / Intercalated
    "C2/m":     ("Layered Carbon",   "monoclinic layered, sp2/sp3 mixed, E_hull~0"),
    "Cmme":     ("Ortho-graphite",   "orthorhombic layered, sp2, E_hull~0"),
    "Fmmm":     ("Ortho-graphite",   "orthorhombic layered, sp2, E_hull~0"),
    "Cmcm":     ("Layered ortho",    "orthorhombic, sp2 chains"),
    "Cmmm":     ("Layered ortho",    "orthorhombic, sp2 sheets"),
    "Pnnm":     ("Compressed C",     "orthorhombic, high-pressure"),
    "I4/mmm":   ("Tetragonal C",     "sp2/sp3 mixed, tetragonal"),
    "Pm-3m":    ("Simple cubic C",   "metallic, high-pressure"),
    "Im-3m":    ("BCC Carbon",       "metallic, high-pressure"),
    # Amorphous / Complex
    "P1":       ("Amorphous/Complex","disordered, mixed coordination"),
    "P-1":      ("Amorphous/Complex","triclinic disordered"),
    "P2_1/m":   ("Monoclinic C",     "low-symmetry, mixed"),
    "P2/m":     ("Monoclinic C",     "low-symmetry, mixed"),
}

# ============================================================================
# 4. TINH PROFILE CHO TUNG CUM
# ============================================================================

def cluster_profile(df, cluster_col):
    rows = []
    for cid in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == cid]
        # Space group overlap voi stable MP
        sub_sgs = set(sub["space_group_symbol"].unique()) if "space_group_symbol" in sub.columns else set()
        sg_overlap = sub_sgs & stable_sgs
        sg_overlap_pct = len(sub[sub["space_group_symbol"].isin(stable_sgs)]) / len(sub) if "space_group_symbol" in sub.columns else 0

        row = {
            "cluster_id":          cid,
            "n":                   len(sub),
            "energy_mean":         sub["relative_energy"].mean(),
            "energy_std":          sub["relative_energy"].std(),
            "energy_median":       sub["relative_energy"].median(),
            "sg_overlap_with_stable": list(sg_overlap),
            "sg_overlap_pct":      sg_overlap_pct,
        }
        if "crystal_system" in sub.columns:
            cs = sub["crystal_system"].value_counts(normalize=True)
            row["top_crystal_system"] = cs.index[0]
            row["top_cs_pct"]         = cs.iloc[0]
        if "space_group_symbol" in sub.columns:
            sg = sub["space_group_symbol"].value_counts()
            row["top_space_group"]  = sg.index[0]
            row["top_sg_pct"]       = sg.iloc[0] / len(sub)
            row["top5_space_groups"] = sg.head(5).to_dict()
        if "volume_per_atom" in sub.columns:
            row["volume_per_atom_mean"] = sub["volume_per_atom"].mean()
        if "density" in sub.columns:
            row["density_mean"] = sub["density"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


kmeans_profile = cluster_profile(gmm, "kmeans_cluster")
gmm_profile_df = cluster_profile(gmm, "gmm_cluster")
gmm_profile_df["kmeans_label"] = gmm_profile_df["cluster_id"].map(
    gmm.groupby("gmm_cluster")["kmeans_label"].agg(lambda x: x.mode()[0])
)

# ============================================================================
# 5. MATCHING FUNCTION — DUA TREN SPACE GROUP OVERLAP
# ============================================================================

def match_cluster(profile_row, ref_df, stable_ref_df):
    """
    So khop cluster voi cau truc Carbon noi tieng dua tren:
    1. Space group overlap voi stable MP (trong so cao nhat)
    2. Crystal system
    3. Energy range
    """
    top_sg  = profile_row.get("top_space_group", "")
    top_cs  = profile_row.get("top_crystal_system", "").lower()
    e_mean  = profile_row["energy_mean"]
    sg_overlap = profile_row.get("sg_overlap_with_stable", [])
    sg_overlap_pct = profile_row.get("sg_overlap_pct", 0)
    top5_sgs = profile_row.get("top5_space_groups", {})

    # --- Uu tien: Fd-3m (Diamond) ---
    fd3m_pct = 0
    if isinstance(top5_sgs, dict):
        total = sum(top5_sgs.values())
        fd3m_pct = top5_sgs.get("Fd-3m", 0) / profile_row["n"] if profile_row["n"] > 0 else 0

    if fd3m_pct > 0.05:
        return "Diamond-like", fd3m_pct, f"Fd-3m present ({fd3m_pct:.1%}), sp3 tetrahedral"

    # --- Graphite / Graphene (hexagonal, trigonal, E thap) ---
    if top_cs in ["hexagonal", "trigonal"] and e_mean < 0.20:
        return "Graphite/Graphene-like", 0.85, "Hexagonal/trigonal, low energy, sp2 layered"

    # --- Layered Carbon (C2/m, monoclinic, E thap) ---
    if top_sg in ["C2/m", "Cmme", "Fmmm", "Cmcm", "Cmmm"] and e_mean < 0.25:
        return "Layered Carbon (sp2)", 0.75, f"Space group {top_sg}, monoclinic/orthorhombic layered"

    # --- Graphite-like (monoclinic C2/m, E rat thap) ---
    if top_sg == "C2/m" and e_mean < 0.15:
        return "Graphite-like (C2/m)", 0.80, "C2/m monoclinic, very low energy, sp2"

    # --- Amorphous / Disordered (triclinic P1/P-1, E cao) ---
    if top_cs == "triclinic" and e_mean > 0.30:
        return "Amorphous Carbon", 0.90, "Triclinic P1/P-1, high energy, disordered"

    # --- Mixed sp2/sp3 (monoclinic, E trung binh) ---
    if top_cs == "monoclinic" and 0.15 < e_mean < 0.35:
        return "Mixed sp2/sp3 Carbon", 0.65, "Monoclinic, intermediate energy, mixed hybridization"

    # --- High-energy complex ---
    if e_mean > 0.35:
        return "High-energy Carbon", 0.60, "High energy, likely metastable/unstable phases"

    # --- Default ---
    return "Unclassified Carbon", 0.40, "Insufficient evidence for classification"


# ============================================================================
# 6. CHAY MATCHING
# ============================================================================

print("\n" + "=" * 65)
print("SO KHOP K-MEANS MACRO-CLUSTERS")
print("=" * 65)

kmeans_matches = {}
for _, row in kmeans_profile.iterrows():
    cid   = int(row["cluster_id"])
    label = gmm[gmm["kmeans_cluster"] == cid]["kmeans_label"].mode()[0]
    match_name, score, reason = match_cluster(row, ref, stable_ref)
    kmeans_matches[cid] = {
        "kmeans_label":    label,
        "scientific_name": match_name,
        "match_score":     round(score, 3),
        "reason":          reason,
        "top_cs":          row.get("top_crystal_system", "N/A"),
        "top_sg":          row.get("top_space_group", "N/A"),
        "energy_mean":     round(row["energy_mean"], 4),
        "sg_overlap":      row.get("sg_overlap_with_stable", []),
        "sg_overlap_pct":  round(row.get("sg_overlap_pct", 0), 4),
        "n":               int(row["n"]),
    }
    print(f"\n  Cluster {cid} — {label}")
    print(f"    N = {int(row['n']):,}")
    print(f"    Top CS / SG    : {row.get('top_crystal_system','?')} / {row.get('top_space_group','?')}")
    print(f"    Energy mean    : {row['energy_mean']:.4f} eV/atom")
    print(f"    SG overlap     : {row.get('sg_overlap_with_stable',[])}  ({row.get('sg_overlap_pct',0):.1%})")
    print(f"    Top 5 SGs      : {row.get('top5_space_groups',{})}")
    print(f"    => Scientific  : {match_name}  (score={score:.2f})")
    print(f"    => Reason      : {reason}")

print("\n" + "=" * 65)
print("SO KHOP GMM MICRO-CLUSTERS")
print("=" * 65)

gmm_matches = []
for _, row in gmm_profile_df.iterrows():
    cid      = int(row["cluster_id"])
    km_label = row.get("kmeans_label", "N/A")
    match_name, score, reason = match_cluster(row, ref, stable_ref)

    # Electronic property prediction tu reference
    top_cs = row.get("top_crystal_system", "").lower()
    ref_cs = ref[ref["crystal_system"].str.lower() == top_cs]
    if len(ref_cs) > 0:
        metal_pct = ref_cs["is_metal"].mean()
        bg_mean   = ref_cs["band_gap"].mean()
        if metal_pct > 0.6:
            elec = "Likely Metal/Semi-metal"
        elif bg_mean > 2.0:
            elec = "Likely Insulator"
        elif bg_mean > 0.3:
            elec = "Likely Semiconductor"
        else:
            elec = "Likely Semi-metal"
    else:
        elec = "Unknown"

    gmm_matches.append({
        "gmm_cluster":       cid,
        "kmeans_macro":      km_label,
        "n":                 int(row["n"]),
        "energy_mean":       round(row["energy_mean"], 4),
        "top_crystal_system":row.get("top_crystal_system", "N/A"),
        "top_space_group":   row.get("top_space_group", "N/A"),
        "sg_overlap_pct":    round(row.get("sg_overlap_pct", 0), 4),
        "scientific_name":   match_name,
        "match_score":       round(score, 3),
        "reason":            reason,
        "electronic_pred":   elec,
        "ref_n_cs":          len(ref_cs),
        "ref_bg_mean":       round(ref_cs["band_gap"].mean(), 4) if len(ref_cs) > 0 else None,
    })

    print(f"\n  GMM-{cid}  [{km_label}]  n={int(row['n']):,}")
    print(f"    Energy mean  : {row['energy_mean']:.4f}")
    print(f"    Top CS / SG  : {row.get('top_crystal_system','?')} / {row.get('top_space_group','?')}")
    print(f"    SG overlap   : {row.get('sg_overlap_with_stable',[])}  ({row.get('sg_overlap_pct',0):.1%})")
    print(f"    => Match     : {match_name}  (score={score:.2f})")
    print(f"    => Electronic: {elec}")

gmm_matches_df = pd.DataFrame(gmm_matches)

# ============================================================================
# 7. DIAMOND ANALYSIS — Fd-3m trong cluster "Ben vung"
# ============================================================================

print("\n" + "=" * 65)
print("PHAN TICH DIAMOND (Fd-3m) TRONG CLUSTER 'BEN VUNG'")
print("=" * 65)

ben_vung = gmm[gmm["kmeans_label"] == "Ben vung (Low Energy)"]
diamond_in_bv = ben_vung[ben_vung["space_group_symbol"] == "Fd-3m"]
graphite_in_bv = ben_vung[ben_vung["space_group_symbol"].isin(["P6_3/mmc", "R-3m", "P6/mmm"])]
c2m_in_bv = ben_vung[ben_vung["space_group_symbol"] == "C2/m"]

print(f"\n  'Ben vung' cluster total: {len(ben_vung):,}")
print(f"  Fd-3m (Diamond-like)    : {len(diamond_in_bv):,}  ({len(diamond_in_bv)/len(ben_vung):.2%})")
print(f"  Graphite SGs            : {len(graphite_in_bv):,}  ({len(graphite_in_bv)/len(ben_vung):.2%})")
print(f"  C2/m (Layered)          : {len(c2m_in_bv):,}  ({len(c2m_in_bv)/len(ben_vung):.2%})")

if len(diamond_in_bv) > 0:
    print(f"\n  Diamond-like (Fd-3m) energy stats:")
    print(f"    mean   = {diamond_in_bv['relative_energy'].mean():.4f}")
    print(f"    median = {diamond_in_bv['relative_energy'].median():.4f}")
    print(f"    min    = {diamond_in_bv['relative_energy'].min():.4f}")

# So sanh voi Diamond trong MP
mp_diamond = ref[ref["space_group_symbol"] == "Fd-3m"]
if len(mp_diamond) > 0:
    print(f"\n  Diamond in MP (Fd-3m): {len(mp_diamond)} samples")
    print(f"    E_above_hull: {mp_diamond['e_above_hull'].values}")
    print(f"    Band Gap    : {mp_diamond['band_gap'].values}")

# ============================================================================
# 8. SAVE
# ============================================================================

# Gan nhan vao gmm dataframe
gmm["scientific_name"] = gmm["gmm_cluster"].map(
    gmm_matches_df.set_index("gmm_cluster")["scientific_name"]
)
gmm["electronic_pred"] = gmm["gmm_cluster"].map(
    gmm_matches_df.set_index("gmm_cluster")["electronic_pred"]
)
gmm["match_score"] = gmm["gmm_cluster"].map(
    gmm_matches_df.set_index("gmm_cluster")["match_score"]
)

# Them nhan chi tiet cho Fd-3m
gmm.loc[gmm["space_group_symbol"] == "Fd-3m", "scientific_name"] = "Diamond-like (Fd-3m)"
gmm.loc[gmm["space_group_symbol"].isin(["P6_3/mmc", "R-3m"]), "scientific_name"] = "Graphite-like"
gmm.loc[gmm["space_group_symbol"] == "C2/m", "scientific_name"] = "Layered Carbon (C2/m)"

gmm.to_csv(OUTPUT_DIR / "tier3_gmm_labeled.csv", index=False)
gmm_matches_df.to_csv(OUTPUT_DIR / "ground_truth_labels.csv", index=False)

# Summary
sci_rows = []
for cid, info in kmeans_matches.items():
    sci_rows.append({
        "level": "K-means", "cluster_id": cid,
        "original_label": info["kmeans_label"],
        "scientific_name": info["scientific_name"],
        "match_score": info["match_score"],
        "n": info["n"], "energy_mean": info["energy_mean"],
        "top_sg": info["top_sg"], "sg_overlap_pct": info["sg_overlap_pct"],
        "reason": info["reason"],
    })
for _, row in gmm_matches_df.iterrows():
    sci_rows.append({
        "level": "GMM", "cluster_id": int(row["gmm_cluster"]),
        "original_label": row["kmeans_macro"],
        "scientific_name": row["scientific_name"],
        "match_score": row["match_score"],
        "n": int(row["n"]), "energy_mean": row["energy_mean"],
        "top_sg": row["top_space_group"], "sg_overlap_pct": row["sg_overlap_pct"],
        "reason": row["reason"],
    })
pd.DataFrame(sci_rows).to_csv(OUTPUT_DIR / "cluster_scientific_names.csv", index=False)

# Text report
lines = [
    "=" * 65,
    "CARBON-24 CLUSTER SCIENTIFIC LABELING REPORT",
    "Reference: Materials Project (64 Carbon structures)",
    "=" * 65, "",
    "STABLE STRUCTURES IN REFERENCE (E_above_hull < 0.05):",
]
for _, r in stable_ref.iterrows():
    lines.append(f"  {r['Material ID']:<12} {r['crystal_system']:<14} "
                 f"{r['space_group_symbol']:<12} E={r['e_above_hull']:.4f} "
                 f"BG={r['band_gap']:.4f} Metal={r['is_metal']}")
lines += ["", "K-MEANS MACRO-CLUSTERS:"]
for cid, info in kmeans_matches.items():
    lines += [
        f"  Cluster {cid} ({info['kmeans_label']}):",
        f"    N={info['n']:,}  Energy={info['energy_mean']:.4f}",
        f"    Scientific: {info['scientific_name']} (score={info['match_score']})",
        f"    Reason: {info['reason']}", "",
    ]
lines += ["", "GMM MICRO-CLUSTERS:"]
for _, r in gmm_matches_df.iterrows():
    lines += [
        f"  GMM-{int(r['gmm_cluster'])} [{r['kmeans_macro']}]:",
        f"    N={int(r['n']):,}  Energy={r['energy_mean']:.4f}",
        f"    Scientific: {r['scientific_name']} (score={r['match_score']})",
        f"    Electronic: {r['electronic_pred']}",
        f"    Reason: {r['reason']}", "",
    ]
with open(OUTPUT_DIR / "matching_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("\n" + "=" * 65)
print("SAVED:")
print("  tier3_gmm_labeled.csv")
print("  ground_truth_labels.csv")
print("  cluster_scientific_names.csv")
print("  matching_report.txt")
print("=" * 65)


from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("carbon24_pipeline_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("=" * 65)
print("GROUND-TRUTH LABELING — Materials Project Reference")
print("=" * 65)

ref = pd.read_csv("carbon.csv")
gmm = pd.read_csv(OUTPUT_DIR / "tier3_gmm_clean.csv")

# Chuan hoa ten cot
ref = ref.rename(columns={
    "Crystal System":    "crystal_system",
    "Space Group Symbol":"space_group_symbol",
    "Energy Above Hull": "e_above_hull",
    "Band Gap":          "band_gap",
    "Is Metal":          "is_metal",
    "Volume":            "volume",
    "Sites":             "sites",
    "Density":           "density",
})

# Volume per atom trong ref
ref["volume_per_atom"] = ref["volume"] / ref["sites"]

print(f"\nReference (Materials Project): {len(ref)} samples")
print(f"Pipeline data (clean):         {len(gmm)} samples")

# ============================================================================
# 2. PHAN LOAI REFERENCE THEO E_ABOVE_HULL
# ============================================================================

ref["stability_class"] = pd.cut(
    ref["e_above_hull"],
    bins=[-0.001, 0.05, 0.30, 1.0, 99],
    labels=["Stable", "Metastable", "Unstable", "Highly Unstable"],
)

ref["electronic_class"] = ref.apply(
    lambda r: "Metal" if r["is_metal"]
    else ("Insulator" if r["band_gap"] > 1.0
          else ("Semiconductor" if r["band_gap"] > 0.1
                else "Semi-metal")),
    axis=1,
)

print("\nReference stability classes:")
print(ref["stability_class"].value_counts().to_string())
print("\nReference electronic classes:")
print(ref["electronic_class"].value_counts().to_string())

# ============================================================================
# 3. TINH PROFILE CHO TUNG CUM
# ============================================================================

def cluster_profile(df, cluster_col):
    """Tinh profile trung binh cho tung cum."""
    rows = []
    for cid in sorted(df[cluster_col].unique()):
        sub = df[df[cluster_col] == cid]
        row = {
            "cluster_id": cid,
            "n": len(sub),
            "energy_mean":       sub["relative_energy"].mean(),
            "energy_std":        sub["relative_energy"].std(),
            "energy_median":     sub["relative_energy"].median(),
            "volume_per_atom_mean": sub["volume_per_atom"].mean() if "volume_per_atom" in sub else np.nan,
            "density_mean":      sub["density"].mean() if "density" in sub else np.nan,
        }
        # Crystal system distribution
        if "crystal_system" in sub.columns:
            cs = sub["crystal_system"].value_counts(normalize=True)
            row["top_crystal_system"] = cs.index[0]
            row["top_cs_pct"]         = cs.iloc[0]
            for cs_name in ["monoclinic", "triclinic", "orthorhombic",
                            "hexagonal", "trigonal", "cubic", "tetragonal"]:
                row[f"cs_{cs_name}"] = cs.get(cs_name, 0)
        # Space group
        if "space_group_symbol" in sub.columns:
            sg = sub["space_group_symbol"].value_counts()
            row["top_space_group"] = sg.index[0]
            row["top_sg_pct"]      = sg.iloc[0] / len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


kmeans_profile = cluster_profile(gmm, "kmeans_cluster")
gmm_profile    = cluster_profile(gmm, "gmm_cluster")

# Them kmeans_label vao gmm_profile
gmm_profile["kmeans_label"] = gmm_profile["cluster_id"].map(
    gmm.groupby("gmm_cluster")["kmeans_label"].agg(lambda x: x.mode()[0])
)

# ============================================================================
# 4. XAY DUNG REFERENCE PROFILES THEO LOAI CAU TRUC CARBON NOI TIENG
# ============================================================================

# Cac cau truc Carbon dien hinh tu Materials Project
KNOWN_STRUCTURES = {
    "Diamond": {
        "crystal_system": "cubic",
        "space_groups":   ["Fd-3m"],
        "e_above_hull_max": 0.05,
        "band_gap_min":   4.0,
        "is_metal":       False,
        "coordination":   4,
        "description":    "sp3 hybridization, tetrahedral bonds, insulator",
    },
    "Graphite": {
        "crystal_system": "hexagonal",
        "space_groups":   ["P6_3/mmc", "P6/mmm"],
        "e_above_hull_max": 0.02,
        "band_gap_max":   0.1,
        "is_metal":       True,
        "coordination":   3,
        "description":    "sp2 hybridization, layered, semi-metal/conductor",
    },
    "Graphene-like": {
        "crystal_system": "hexagonal",
        "space_groups":   ["P6/mmm", "P6_3/mmc"],
        "e_above_hull_max": 0.05,
        "band_gap_max":   0.5,
        "coordination":   3,
        "description":    "2D sp2 carbon layers",
    },
    "Lonsdaleite": {
        "crystal_system": "hexagonal",
        "space_groups":   ["P6_3/mmc"],
        "e_above_hull_max": 0.10,
        "band_gap_min":   3.0,
        "coordination":   4,
        "description":    "Hexagonal diamond, sp3, high pressure phase",
    },
    "Amorphous/Complex": {
        "crystal_system": "triclinic",
        "space_groups":   ["P1"],
        "e_above_hull_max": 2.0,
        "band_gap_max":   2.0,
        "description":    "Disordered, mixed coordination, high energy",
    },
    "Fullerene-like": {
        "crystal_system": "cubic",
        "space_groups":   ["Pa-3", "Fm-3m", "Im-3"],
        "e_above_hull_max": 1.5,
        "band_gap_min":   1.0,
        "description":    "Cage-like sp2 structures",
    },
    "Carbyne/Chain": {
        "crystal_system": "hexagonal",
        "space_groups":   ["P6/mmm"],
        "e_above_hull_max": 0.5,
        "coordination":   2,
        "description":    "sp hybridization, linear chains",
    },
}

# ============================================================================
# 5. SO KHOP KMEANS CLUSTERS VOI REFERENCE
# ============================================================================

def match_kmeans_to_reference(profile_row, ref_df):
    """
    So khop 1 cluster voi reference dua tren:
    - Crystal system distribution
    - Energy range
    - Volume per atom
    """
    scores = {}

    # Lay thong tin cluster
    e_mean = profile_row["energy_mean"]
    # relative_energy trong carbon24 ~ 0-0.5 eV/atom
    # E_above_hull trong MP ~ 0-3 eV/atom
    # Mapping: relative_energy thap -> E_above_hull thap -> stable
    top_cs = profile_row.get("top_crystal_system", "").lower()

    # Score dua tren crystal system
    cs_scores = {
        "Diamond":        1.0 if top_cs == "cubic"      else 0.2,
        "Graphite":       1.0 if top_cs == "hexagonal"  else 0.3,
        "Graphene-like":  0.8 if top_cs == "hexagonal"  else 0.2,
        "Lonsdaleite":    0.7 if top_cs == "hexagonal"  else 0.1,
        "Amorphous/Complex": 1.0 if top_cs == "triclinic" else 0.3,
        "Fullerene-like": 0.6 if top_cs == "cubic"      else 0.2,
        "Carbyne/Chain":  0.5 if top_cs == "hexagonal"  else 0.1,
    }

    # Score dua tren energy (relative_energy thap = stable)
    e_scores = {
        "Diamond":        1.0 if e_mean < 0.15 else 0.3,
        "Graphite":       1.0 if e_mean < 0.15 else 0.3,
        "Graphene-like":  0.8 if e_mean < 0.20 else 0.3,
        "Lonsdaleite":    0.7 if e_mean < 0.20 else 0.2,
        "Amorphous/Complex": 1.0 if e_mean > 0.35 else 0.2,
        "Fullerene-like": 0.6 if 0.20 < e_mean < 0.45 else 0.2,
        "Carbyne/Chain":  0.5 if 0.15 < e_mean < 0.35 else 0.2,
    }

    for name in KNOWN_STRUCTURES:
        scores[name] = 0.5 * cs_scores.get(name, 0.1) + 0.5 * e_scores.get(name, 0.1)

    best = max(scores, key=scores.get)
    return best, scores[best], scores


print("\n" + "=" * 65)
print("SO KHOP K-MEANS MACRO-CLUSTERS")
print("=" * 65)

kmeans_matches = {}
for _, row in kmeans_profile.iterrows():
    cid   = int(row["cluster_id"])
    label = gmm[gmm["kmeans_cluster"] == cid]["kmeans_label"].mode()[0]
    best, score, all_scores = match_kmeans_to_reference(row, ref)
    kmeans_matches[cid] = {
        "kmeans_label":    label,
        "best_match":      best,
        "match_score":     round(score, 3),
        "top_cs":          row.get("top_crystal_system", "N/A"),
        "energy_mean":     round(row["energy_mean"], 4),
        "n":               int(row["n"]),
        "description":     KNOWN_STRUCTURES[best]["description"],
    }
    print(f"\n  Cluster {cid} — {label}")
    print(f"    N = {int(row['n']):,}")
    print(f"    Top crystal system : {row.get('top_crystal_system','N/A')}  ({row.get('top_cs_pct',0):.1%})")
    print(f"    Energy mean        : {row['energy_mean']:.4f} eV/atom")
    print(f"    Top space group    : {row.get('top_space_group','N/A')}")
    print(f"    Best match         : {best}  (score={score:.3f})")
    print(f"    Description        : {KNOWN_STRUCTURES[best]['description']}")
    print(f"    All scores         : { {k: round(v,2) for k,v in all_scores.items()} }")

# ============================================================================
# 6. SO KHOP GMM MICRO-CLUSTERS
# ============================================================================

print("\n" + "=" * 65)
print("SO KHOP GMM MICRO-CLUSTERS")
print("=" * 65)

gmm_matches = []
for _, row in gmm_profile.iterrows():
    cid       = int(row["cluster_id"])
    km_label  = row.get("kmeans_label", "N/A")
    best, score, all_scores = match_khop_to_reference(row, ref) if False else match_kmeans_to_reference(row, ref)

    # Tinh them: so sanh voi ref theo crystal system
    top_cs = row.get("top_crystal_system", "").lower()
    ref_cs_match = ref[ref["crystal_system"].str.lower() == top_cs]
    ref_e_range  = (ref_cs_match["e_above_hull"].min(), ref_cs_match["e_above_hull"].max()) if len(ref_cs_match) > 0 else (None, None)
    ref_bg_mean  = ref_cs_match["band_gap"].mean() if len(ref_cs_match) > 0 else None
    ref_metal_pct = ref_cs_match["is_metal"].mean() if len(ref_cs_match) > 0 else None

    # Electronic property prediction
    if ref_metal_pct is not None:
        if ref_metal_pct > 0.6:
            elec_pred = "Likely Metal/Semi-metal"
        elif ref_bg_mean and ref_bg_mean > 2.0:
            elec_pred = "Likely Insulator"
        elif ref_bg_mean and ref_bg_mean > 0.3:
            elec_pred = "Likely Semiconductor"
        else:
            elec_pred = "Likely Semi-metal"
    else:
        elec_pred = "Unknown"

    gmm_matches.append({
        "gmm_cluster":      cid,
        "kmeans_macro":     km_label,
        "n":                int(row["n"]),
        "energy_mean":      round(row["energy_mean"], 4),
        "top_crystal_system": row.get("top_crystal_system", "N/A"),
        "top_space_group":  row.get("top_space_group", "N/A"),
        "best_match":       best,
        "match_score":      round(score, 3),
        "electronic_pred":  elec_pred,
        "ref_n_cs_match":   len(ref_cs_match),
        "ref_e_min":        round(ref_e_range[0], 4) if ref_e_range[0] is not None else None,
        "ref_e_max":        round(ref_e_range[1], 4) if ref_e_range[1] is not None else None,
        "ref_band_gap_mean":round(ref_bg_mean, 4) if ref_bg_mean is not None else None,
        "scientific_label": f"{best} ({row.get('top_crystal_system','?')})",
        "description":      KNOWN_STRUCTURES[best]["description"],
    })

    print(f"\n  GMM-{cid}  [{km_label}]  n={int(row['n']):,}")
    print(f"    Energy mean     : {row['energy_mean']:.4f}")
    print(f"    Top CS          : {row.get('top_crystal_system','N/A')}  ({row.get('top_cs_pct',0):.1%})")
    print(f"    Best match      : {best}  (score={score:.3f})")
    print(f"    Electronic pred : {elec_pred}")
    print(f"    Ref CS matches  : {len(ref_cs_match)} samples in MP")

gmm_matches_df = pd.DataFrame(gmm_matches)

# ============================================================================
# 7. SCIENTIFIC CLUSTER NAMES
# ============================================================================

print("\n" + "=" * 65)
print("SCIENTIFIC CLUSTER NAMES (FINAL)")
print("=" * 65)

# K-means scientific names
kmeans_sci = {}
for cid, info in kmeans_matches.items():
    name = info["best_match"]
    label = info["kmeans_label"]
    sci_name = f"{name}-like Phase ({label})"
    kmeans_sci[cid] = sci_name
    print(f"  K-means {cid}: {sci_name}")

print()
# GMM scientific names
print("  GMM micro-clusters:")
for _, row in gmm_matches_df.iterrows():
    print(f"  GMM-{int(row['gmm_cluster'])}: {row['scientific_label']}  [{row['kmeans_macro']}]")

# ============================================================================
# 8. SO SANH TRUC TIEP VOI REFERENCE
# ============================================================================

print("\n" + "=" * 65)
print("SO SANH TRUC TIEP: CLUSTER vs MATERIALS PROJECT")
print("=" * 65)

# Lay stable structures tu MP
stable_ref = ref[ref["e_above_hull"] < 0.05].copy()
print(f"\nStable structures in MP (E_above_hull < 0.05): {len(stable_ref)}")
print(stable_ref[["Material ID", "crystal_system", "space_group_symbol",
                   "e_above_hull", "band_gap", "is_metal"]].to_string(index=False))

# So sanh voi cluster "Ben vung"
ben_vung = gmm[gmm["kmeans_label"] == "Ben vung (Low Energy)"]
print(f"\nCluster 'Ben vung' stats:")
print(f"  N = {len(ben_vung):,}")
print(f"  Energy mean = {ben_vung['relative_energy'].mean():.4f}")
print(f"  Top crystal systems:")
print(ben_vung["crystal_system"].value_counts().head(5).to_string())
print(f"  Top space groups:")
print(ben_vung["space_group_symbol"].value_counts().head(5).to_string())

# Overlap space groups
ben_vung_sgs = set(ben_vung["space_group_symbol"].unique())
stable_sgs   = set(stable_ref["space_group_symbol"].unique())
overlap_sgs  = ben_vung_sgs & stable_sgs
print(f"\n  Space group overlap with stable MP structures: {overlap_sgs}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================

# Ground truth labels cho toan bo dataset
gmm["scientific_label"] = gmm["gmm_cluster"].map(
    gmm_matches_df.set_index("gmm_cluster")["scientific_label"]
)
gmm["electronic_pred"] = gmm["gmm_cluster"].map(
    gmm_matches_df.set_index("gmm_cluster")["electronic_pred"]
)
gmm["best_match"] = gmm["gmm_cluster"].map(
    gmm_matches_df.set_index("gmm_cluster")["best_match"]
)

gmm.to_csv(OUTPUT_DIR / "tier3_gmm_labeled.csv", index=False)
gmm_matches_df.to_csv(OUTPUT_DIR / "ground_truth_labels.csv", index=False)

# Scientific names summary
sci_summary = []
for cid, info in kmeans_matches.items():
    sci_summary.append({
        "level":          "K-means",
        "cluster_id":     cid,
        "original_label": info["kmeans_label"],
        "scientific_name":kmeans_sci[cid],
        "best_match":     info["best_match"],
        "match_score":    info["match_score"],
        "n":              info["n"],
        "energy_mean":    info["energy_mean"],
        "description":    info["description"],
    })
for _, row in gmm_matches_df.iterrows():
    sci_summary.append({
        "level":          "GMM",
        "cluster_id":     int(row["gmm_cluster"]),
        "original_label": row["kmeans_macro"],
        "scientific_name":row["scientific_label"],
        "best_match":     row["best_match"],
        "match_score":    row["match_score"],
        "n":              int(row["n"]),
        "energy_mean":    row["energy_mean"],
        "description":    row["description"],
    })

sci_df = pd.DataFrame(sci_summary)
sci_df.to_csv(OUTPUT_DIR / "cluster_scientific_names.csv", index=False)

# Text report
report_lines = [
    "=" * 65,
    "CARBON-24 CLUSTER SCIENTIFIC LABELING REPORT",
    "Reference: Materials Project (64 Carbon structures)",
    "=" * 65,
    "",
    "K-MEANS MACRO-CLUSTERS:",
]
for cid, info in kmeans_matches.items():
    report_lines += [
        f"  Cluster {cid} ({info['kmeans_label']}):",
        f"    N = {info['n']:,}",
        f"    Energy mean = {info['energy_mean']:.4f} eV/atom",
        f"    Best match  = {info['best_match']} (score={info['match_score']})",
        f"    Description = {info['description']}",
        "",
    ]

report_lines += ["", "GMM MICRO-CLUSTERS:"]
for _, row in gmm_matches_df.iterrows():
    report_lines += [
        f"  GMM-{int(row['gmm_cluster'])} [{row['kmeans_macro']}]:",
        f"    N = {int(row['n']):,}",
        f"    Energy mean = {row['energy_mean']:.4f} eV/atom",
        f"    Scientific label = {row['scientific_label']}",
        f"    Electronic pred  = {row['electronic_pred']}",
        f"    MP reference CS  = {int(row['ref_n_cs_match'])} samples",
        "",
    ]

report_lines += [
    "",
    "STABLE STRUCTURES (E_above_hull < 0.05 eV/atom) IN REFERENCE:",
]
for _, row in stable_ref.iterrows():
    report_lines.append(
        f"  {row['Material ID']:<15} {row['crystal_system']:<15} "
        f"{row['space_group_symbol']:<12} E={row['e_above_hull']:.4f}  "
        f"BG={row['band_gap']:.4f}  Metal={row['is_metal']}"
    )

with open(OUTPUT_DIR / "matching_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n" + "=" * 65)
print("SAVED:")
print(f"  tier3_gmm_labeled.csv          (with scientific labels)")
print(f"  ground_truth_labels.csv        (GMM matching details)")
print(f"  cluster_scientific_names.csv   (summary)")
print(f"  matching_report.txt            (full report)")
print("=" * 65)
