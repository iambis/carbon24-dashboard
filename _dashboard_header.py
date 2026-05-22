import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
import json
import os

# Page config
st.set_page_config(
    page_title="Carbon-24 Data Mining Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">⚛️ Carbon-24 Structure Analysis</div>', unsafe_allow_html=True)
st.markdown("**Khám phá, Phân loại và Dự đoán năng lượng cấu trúc Carbon-24 từ dữ liệu mô phỏng DFT**")

# Sidebar — 4-step workflow
with st.sidebar:
    st.markdown("## ⚛️ Carbon-24")
    st.markdown("---")
    st.markdown("### 🗺️ Workflow")

    page = st.radio(
        "",
        [
            "🔍 Bước 1 — Khảo sát & Lọc nhiễu",
            "🧬 Bước 2 — Định danh Pha thù hình",
            "🏆 Bước 3 — Model Leaderboard",
            "⚡ Bước 4 — Dự đoán Năng lượng",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.info(
        "**Carbon-24** (AIRSS)\n\n"
        "- 10,153 cấu trúc DFT\n"
        "- 786 noise (7.74%)\n"
        "- 9,367 clean samples\n"
        "- 3 macro-clusters\n"
        "- 10 micro-clusters\n"
        "- Best model: CatBoost R²=0.975"
    )

# Load data
@st.cache_data
def load_data():
    try:
        # Try to load clustered data
        df = pd.read_csv('carbon24_kmeans_results/carbon24_clustered.csv')
        has_clusters = True
    except:
        # Fallback to preprocessed data
        df = pd.read_csv('carbon24_preprocessing_results/carbon24_feature_selected.csv')
        has_clusters = False
    
    return df, has_clusters

@st.cache_data
def load_feature_info():
    try:
        with open('carbon24_preprocessing_results/selected_features.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_clustering_report():
    try:
        with open('carbon24_kmeans_results/clustering_report.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_data
def load_comparison_results():
    try:
        methods_overview = pd.read_csv('carbon24_clustering_comparison_results/methods_overview.csv')
        quality_metrics = pd.read_csv('carbon24_clustering_comparison_results/quality_metrics.csv')
        method_ranking = pd.read_csv('carbon24_clustering_comparison_results/method_ranking.csv')
        return methods_overview, quality_metrics, method_ranking
    except:
        return None, None, None

@st.cache_data
def load_gmm_results():
    try:
        df = pd.read_csv('carbon24_gmm_results/results/carbon24_gmm_results.csv')
        with open('carbon24_gmm_results/gmm_clustering_report.json', 'r') as f:
            report = json.load(f)
        cluster_profile = pd.read_csv('carbon24_gmm_results/tables/gmm_cluster_profile.csv')
        return df, report, cluster_profile
    except:
        return None, None, None

@st.cache_data
def load_hierarchical_results():
    try:
        df = pd.read_csv('carbon24_hierarchical_baseline/results/carbon24_hierarchical_results.csv')
        cluster_interpretation = pd.read_csv('carbon24_hierarchical_baseline/tables/hierarchical_cluster_interpretation.csv')
        return df, cluster_interpretation
    except:
        return None, None

@st.cache_data
def load_hdbscan_results():
    try:
        df = pd.read_csv('hdbscan_phuc/hdbscan_results.csv')
        cluster_profile = pd.read_csv('hdbscan_phuc/hdbscan_cluster_profile.csv')
        energy_summary = pd.read_csv('hdbscan_phuc/hdbscan_energy_summary.csv')
        noise_outliers = pd.read_csv('hdbscan_phuc/hdbscan_noise_outliers.csv')
        return df, cluster_profile, energy_summary, noise_outliers
    except:
        return None, None, None, None

df, has_clusters = load_data()
feature_info = load_feature_info()
clustering_report = load_clustering_report()
methods_overview, quality_metrics, method_ranking = load_comparison_results()
gmm_df, gmm_report, gmm_cluster_profile = load_gmm_results()
hierarchical_df, hierarchical_interpretation = load_hierarchical_results()
hdbscan_df, hdbscan_cluster_profile, hdbscan_energy_summary, hdbscan_noise_outliers = load_hdbscan_results()

# ============================================================================
# ANOMALY DETECTION DATA LOADING
# ============================================================================

@st.cache_data
@st.cache_data
def load_anomaly_detection_results():
    """Load anomaly detection results"""
    try:
        results_df    = pd.read_csv('carbon24_anomaly_detection/anomaly_detection_results.csv')
        summary_df    = pd.read_csv('carbon24_anomaly_detection/anomaly_summary.csv')
        comparison_df = pd.read_csv('carbon24_anomaly_detection/anomaly_method_comparison.csv')
        details_df    = pd.read_csv('carbon24_anomaly_detection/anomaly_details.csv')
        return results_df, summary_df, comparison_df, details_df
    except Exception as e:
        st.error(f"Lỗi khi load anomaly detection results: {e}")
        return None, None, None, None



# ============================================================================
# GROUND-TRUTH LABELING — DATA LOADING & RENDER
# ============================================================================

@st.cache_data
def load_ground_truth_data():
    try:
        gmm    = pd.read_csv('carbon24_pipeline_results/tier3_gmm_labeled.csv')
        labels = pd.read_csv('carbon24_pipeline_results/ground_truth_labels.csv')
        sci    = pd.read_csv('carbon24_pipeline_results/cluster_scientific_names.csv')
        ref    = pd.read_csv('carbon.csv')
        noise  = pd.read_csv('carbon24_pipeline_results/tier1_noise_analysis.csv')
        return gmm, labels, sci, ref, noise
    except Exception as e:
        st.error(f"Lỗi load ground-truth data: {e}")
        return None, None, None, None, None


SCI_COLORS_DASH = {
    "Diamond-like (Fd-3m)":  "#1a9850",
    "Graphite-like":         "#4393c3",
    "Layered C2/m":          "#74add1",
    "Layered Carbon (sp2)":  "#74add1",
    "Mixed sp2/sp3 Carbon":  "#fee090",
    "Amorphous Carbon":      "#d73027",
    "High-energy Carbon":    "#a50026",
    "Unclassified Carbon":   "#bababa",
}

def get_sci_color(name):
    for k, v in SCI_COLORS_DASH.items():
        if k in str(name):
            return v
    return "#bababa"


def render_ground_truth_tab():
    st.markdown('<div class="sub-header">🔬 Ground-Truth Labeling</div>', unsafe_allow_html=True)

    st.info(
        "**Phương pháp:** So khớp các cụm với 64 cấu trúc Carbon chuẩn từ **Materials Project** "
        "dựa trên Space Group Symbol, Crystal System và Energy range. "
        "Kết quả cho phép gán nhãn khoa học chính xác cho từng cụm."
    )

    gmm, labels, sci, ref, noise = load_ground_truth_data()
    if gmm is None:
        st.warning("Chưa có dữ liệu. Vui lòng chạy:")
        st.code("python carbon24_ground_truth_labeling.py\npython carbon24_ground_truth_viz.py")
        return

    ref = ref.rename(columns={
        "Crystal System": "crystal_system",
        "Space Group Symbol": "space_group_symbol",
        "Energy Above Hull": "e_above_hull",
        "Band Gap": "band_gap",
        "Is Metal": "is_metal",
    })

    # Them structure_type
    gmm["structure_type"] = "Other"
    gmm.loc[gmm["space_group_symbol"] == "Fd-3m",                           "structure_type"] = "Diamond (Fd-3m)"
    gmm.loc[gmm["space_group_symbol"].isin(["P6_3/mmc", "R-3m", "P6/mmm"]), "structure_type"] = "Graphite/Graphene"
    gmm.loc[gmm["space_group_symbol"] == "C2/m",                            "structure_type"] = "Layered C2/m"
    gmm.loc[gmm["space_group_symbol"].isin(["P1", "P-1"]),                  "structure_type"] = "Amorphous (P1/P-1)"

    TYPE_COLORS = {
        "Diamond (Fd-3m)":    "#1a9850",
        "Graphite/Graphene":  "#4393c3",
        "Layered C2/m":       "#74add1",
        "Amorphous (P1/P-1)": "#d73027",
        "Other":              "#bababa",
    }

    # ── METRICS ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 Tổng quan")
    n_diamond  = (gmm["structure_type"] == "Diamond (Fd-3m)").sum()
    n_graphite = (gmm["structure_type"] == "Graphite/Graphene").sum()
    n_layered  = (gmm["structure_type"] == "Layered C2/m").sum()
    n_amorph   = (gmm["structure_type"] == "Amorphous (P1/P-1)").sum()
    n_other    = (gmm["structure_type"] == "Other").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💎 Diamond (Fd-3m)",    f"{n_diamond:,}",  f"{n_diamond/len(gmm):.1%}")
    c2.metric("🪨 Graphite/Graphene",  f"{n_graphite:,}", f"{n_graphite/len(gmm):.1%}")
    c3.metric("📄 Layered C2/m",       f"{n_layered:,}",  f"{n_layered/len(gmm):.1%}")
    c4.metric("🌀 Amorphous",          f"{n_amorph:,}",   f"{n_amorph/len(gmm):.1%}")
    c5.metric("🔴 Noise (anomalies)",  f"{len(noise):,}", f"{len(noise)/(len(gmm)+len(noise)):.1%}")

    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ PCA Labels",
        "⚡ Energy Analysis",
        "🔷 Space Group Overlap",
        "🧬 Structure Types",
        "📋 Cluster Summary",
    ])

    # ── TAB 1: PCA ───────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Phân bố không gian PCA theo loại cấu trúc")

        type_order = ["Diamond (Fd-3m)", "Graphite/Graphene", "Layered C2/m",
                      "Amorphous (P1/P-1)", "Other"]

        col1, col2 = st.columns([3, 1])
        with col1:
            fig = go.Figure()
            for stype in type_order:
                mask = gmm["structure_type"] == stype
                if mask.sum() == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=gmm.loc[mask, "pca1"], y=gmm.loc[mask, "pca2"],
                    mode="markers",
                    name=f"{stype} ({mask.sum():,})",
                    marker=dict(
                        size=5 if stype != "Other" else 3,
                        color=TYPE_COLORS[stype],
                        opacity=0.65 if stype != "Other" else 0.15,
                    ),
                ))
            fig.add_trace(go.Scatter(
                x=noise["pca1"], y=noise["pca2"],
                mode="markers", name=f"Noise ({len(noise):,})",
                marker=dict(size=6, color="black", opacity=0.5, symbol="x"),
            ))
            fig.update_layout(
                title="PCA Space — Structure Types (Ground-Truth Labels)",
                xaxis_title="PCA1", yaxis_title="PCA2",
                height=560, hovermode="closest",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Chú thích:**")
            for stype, color in TYPE_COLORS.items():
                n = (gmm["structure_type"] == stype).sum()
                st.markdown(
                    f'<span style="color:{color}; font-size:18px">■</span> '
                    f'**{stype}**<br><small>{n:,} ({n/len(gmm):.1%})</small>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<span style="color:black; font-size:18px">✕</span> '
                f'**Noise**<br><small>{len(noise):,}</small>',
                unsafe_allow_html=True,
            )

    # ── TAB 2: ENERGY ────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### So sánh năng lượng: Clusters vs Materials Project")
        st.caption("Đường ngang: ngưỡng ổn định từ Materials Project (Stable < 0.05, Metastable < 0.30 eV/atom)")

        col1, col2 = st.columns(2)

        with col1:
            # Violin per structure type
            type_order_e = ["Diamond (Fd-3m)", "Graphite/Graphene", "Layered C2/m",
                            "Amorphous (P1/P-1)", "Other"]
            fig = go.Figure()
            for stype in type_order_e:
                sub = gmm[gmm["structure_type"] == stype]["relative_energy"].dropna()
                if len(sub) == 0:
                    continue
                fig.add_trace(go.Violin(
                    y=sub, name=stype,
                    box_visible=True, meanline_visible=True,
                    fillcolor=TYPE_COLORS[stype], opacity=0.7,
                    line_color="black",
                ))
            fig.add_hline(y=0.05, line_dash="dash", line_color="green",
                          annotation_text="Stable (0.05)")
            fig.add_hline(y=0.30, line_dash="dash", line_color="orange",
                          annotation_text="Metastable (0.30)")
            fig.update_layout(
                title="Energy Distribution per Structure Type",
                yaxis_title="Relative Energy (eV/atom)",
                height=480, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Diamond energy highlight
            diamond_e = gmm[gmm["structure_type"] == "Diamond (Fd-3m)"]["relative_energy"]
            st.markdown("**💎 Diamond (Fd-3m) — Key Finding:**")
            st.metric("Mean energy", f"{diamond_e.mean():.4f} eV/atom")
            st.metric("Min energy",  f"{diamond_e.min():.4f} eV/atom")
            st.metric("Samples",     f"{len(diamond_e):,}")
            st.success(f"Diamond cluster có energy ≈ **{diamond_e.mean():.4f} eV/atom** — "
                       f"gần bằng 0, xác nhận đây là cấu trúc **ổn định nhất** trong dataset.")

            st.markdown("---")
            st.markdown("**📊 Energy summary by structure type:**")
            e_summary = gmm.groupby("structure_type")["relative_energy"].agg(
                ["mean", "std", "min", "max"]
            ).round(4).sort_values("mean")
            st.dataframe(e_summary, use_container_width=True)

    # ── TAB 3: SPACE GROUP OVERLAP ────────────────────────────────────────────
    with tab3:
        st.markdown("#### Space Group Overlap với Stable Materials Project Structures")

        stable_ref = ref[ref["e_above_hull"] < 0.05]
        stable_sgs = stable_ref["space_group_symbol"].value_counts().head(10).index.tolist()

        # Heatmap
        sg_matrix = {}
        for cid in sorted(gmm["gmm_cluster"].unique()):
            sub = gmm[gmm["gmm_cluster"] == cid]
            sg_matrix[f"GMM-{cid}"] = {
                sg: round((sub["space_group_symbol"] == sg).sum() / len(sub) * 100, 1)
                for sg in stable_sgs
            }
        sg_df = pd.DataFrame(sg_matrix).T

        fig = go.Figure(go.Heatmap(
            z=sg_df.values,
            x=sg_df.columns.tolist(),
            y=sg_df.index.tolist(),
            colorscale="YlOrRd",
            text=[[f"{v:.1f}%" for v in row] for row in sg_df.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            zmin=0,
        ))
        fig.update_layout(
            title="% Samples per GMM Cluster with Stable MP Space Groups",
            xaxis_title="Space Group (in stable MP structures)",
            yaxis_title="GMM Cluster",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Stable MP structures (E_above_hull < 0.05 eV/atom):**")
        stable_show = stable_ref[["Material ID", "crystal_system", "space_group_symbol",
                                   "e_above_hull", "band_gap", "is_metal"]].copy()
        stable_show.columns = ["Material ID", "Crystal System", "Space Group",
                                "E_above_hull", "Band Gap", "Is Metal"]
        st.dataframe(stable_show.sort_values("E_above_hull"), use_container_width=True, hide_index=True)

    # ── TAB 4: STRUCTURE TYPES ────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Phân bố loại cấu trúc theo K-means cluster")

        km_labels_order = ["Ben vung (Low Energy)", "Trung gian", "Phuc tap (High Energy)"]
        type_order_bar  = ["Diamond (Fd-3m)", "Graphite/Graphene", "Layered C2/m",
                           "Amorphous (P1/P-1)", "Other"]

        fig = go.Figure()
        for stype in type_order_bar:
            vals = []
            for lbl in km_labels_order:
                sub = gmm[gmm["kmeans_label"] == lbl]
                pct = (sub["structure_type"] == stype).sum() / len(sub) * 100
                vals.append(round(pct, 1))
            fig.add_trace(go.Bar(
                name=stype, x=km_labels_order, y=vals,
                marker_color=TYPE_COLORS[stype],
                text=[f"{v:.1f}%" for v in vals],
                textposition="inside",
            ))
        fig.update_layout(
            barmode="stack",
            title="Structure Type Composition per K-means Macro-cluster",
            yaxis_title="Percentage (%)",
            height=480, legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Crystal system comparison
        st.markdown("**Crystal System: Clusters vs Materials Project**")
        cs_order = ["monoclinic", "triclinic", "orthorhombic",
                    "hexagonal", "trigonal", "cubic", "tetragonal"]
        mp_cs = ref["crystal_system"].str.lower().value_counts(normalize=True) * 100

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="MP Reference", x=cs_order,
            y=[mp_cs.get(cs, 0) for cs in cs_order],
            marker_color="#4393c3", opacity=0.7,
        ))
        km_colors_cs = ["#1a9850", "#fee090", "#a50026"]
        for i, lbl in enumerate(km_labels_order):
            sub = gmm[gmm["kmeans_label"] == lbl]
            cs  = sub["crystal_system"].str.lower().value_counts(normalize=True) * 100
            fig2.add_trace(go.Bar(
                name=lbl, x=cs_order,
                y=[cs.get(c, 0) for c in cs_order],
                marker_color=km_colors_cs[i], opacity=0.7,
            ))
        fig2.update_layout(
            barmode="group", height=400,
            xaxis_title="Crystal System", yaxis_title="Percentage (%)",
            title="Crystal System Distribution: Clusters vs MP Reference",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 5: CLUSTER SUMMARY ────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Scientific Labels — Cluster Summary")

        # K-means summary
        st.markdown("**K-means Macro-clusters:**")
        km_summary = []
        for lbl in km_labels_order:
            sub = gmm[gmm["kmeans_label"] == lbl]
            top_sg = sub["space_group_symbol"].value_counts().index[0]
            top_st = sub["structure_type"].value_counts().index[0]
            km_summary.append({
                "K-means Label": lbl,
                "N": f"{len(sub):,}",
                "Energy (mean)": f"{sub['relative_energy'].mean():.4f}",
                "Top Space Group": top_sg,
                "Top Structure Type": top_st,
                "Scientific Interpretation": (
                    "Diamond-like + Layered Carbon phases" if lbl == "Ben vung (Low Energy)"
                    else "Mixed sp2/sp3 disordered phases" if lbl == "Trung gian"
                    else "High-energy metastable phases"
                ),
            })
        st.dataframe(pd.DataFrame(km_summary), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**GMM Micro-clusters (with scientific labels):**")

        # Lay scientific label tu cot co san
        sci_col = "scientific_label" if "scientific_label" in gmm.columns else "best_match"
        gmm_summary = []
        for cid in sorted(gmm["gmm_cluster"].unique()):
            sub = gmm[gmm["gmm_cluster"] == cid]
            gmm_summary.append({
                "GMM": f"GMM-{cid}",
                "K-means Macro": sub["kmeans_label"].mode()[0],
                "N": f"{len(sub):,}",
                "Energy (mean)": f"{sub['relative_energy'].mean():.4f}",
                "Top Space Group": sub["space_group_symbol"].value_counts().index[0],
                "Scientific Label": sub[sci_col].mode()[0] if sci_col in sub.columns else "N/A",
                "Electronic Pred": sub["electronic_pred"].mode()[0] if "electronic_pred" in sub.columns else "N/A",
            })
        gmm_sum_df = pd.DataFrame(gmm_summary)
        st.dataframe(gmm_sum_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**📥 Download:**")
        col1, col2 = st.columns(2)
        with col1:
            csv = gmm.to_csv(index=False)
            st.download_button("Download Labeled Data (CSV)", csv,
                               "gmm_labeled.csv", "text/csv")
        with col2:
            if labels is not None:
                csv2 = labels.to_csv(index=False)
                st.download_button("Download Ground-Truth Labels (CSV)", csv2,
                                   "ground_truth_labels.csv", "text/csv")


# ============================================================================
# PIPELINE 3 TANG — DATA LOADING & RENDER
# ============================================================================

@st.cache_data
def load_pipeline_results():
    try:
        final_df    = pd.read_csv('carbon24_pipeline_results/pipeline_final.csv')
        noise_df    = pd.read_csv('carbon24_pipeline_results/tier1_noise_analysis.csv')
        km_profile  = pd.read_csv('carbon24_pipeline_results/tier2_kmeans_profile.csv')
        gmm_df      = pd.read_csv('carbon24_pipeline_results/tier3_gmm_clean.csv')
        gmm_profile = pd.read_csv('carbon24_pipeline_results/tier3_gmm_profile.csv')
        return final_df, noise_df, km_profile, gmm_df, gmm_profile
    except Exception as e:
        st.error(f"Lỗi load pipeline results: {e}")
        return None, None, None, None, None


def render_pipeline_tab():
    st.markdown('<div class="sub-header">🔬 Pipeline 3 Tầng</div>', unsafe_allow_html=True)

    st.markdown("""
    | Tầng | Phương pháp | Input | Output |
    |------|-------------|-------|--------|
    | **1** | HDBSCAN Noise | 10,153 mẫu | 786 noise → Anomaly Analysis |
    | **2** | K-means (k=3) | 9,367 sạch | 3 macro-clusters |
    | **3** | GMM (k=10) | 9,367 sạch | 10 micro-clusters (polymorphs) |
    """)

    final_df, noise_df, km_profile, gmm_df, gmm_profile = load_pipeline_results()
    if final_df is None:
        st.warning("Chưa có dữ liệu. Vui lòng chạy:")
        st.code("python carbon24_pipeline_3tier.py")
        return

    # final_df dùng cho Tier 1 (có cả noise + clean, có pca1/pca2)
    # gmm_df   dùng cho Tier 2 & 3 (chỉ clean, có kmeans_label + gmm_cluster)
    noise = final_df[final_df['pipeline_stage'] == 'noise']
    clean = gmm_df   # 9,367 sạch, đầy đủ kmeans + gmm labels

    # ── METRICS ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 Tổng quan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng mẫu", f"{len(final_df):,}")
    c2.metric("Noise (Tier 1)", f"{len(noise):,}", f"{len(noise)/len(final_df):.1%}")
    c3.metric("Clean (Tier 2+)", f"{len(clean):,}", f"{len(clean)/len(final_df):.1%}")
    c4.metric("GMM sub-clusters", "10", "polymorphs")

    st.markdown("---")

    # ── TABS CHO 3 TANG ──────────────────────────────────────────────────────
    t1, t2, t3 = st.tabs(["🔴 Tầng 1 — Noise", "🟡 Tầng 2 — K-means", "🟢 Tầng 3 — GMM"])

    # ── TANG 1 ───────────────────────────────────────────────────────────────
    with t1:
        st.markdown("#### Tầng 1: HDBSCAN Noise Detection")
        st.info(f"**{len(noise):,} điểm nhiễu ({len(noise)/len(final_df):.2%})** — đây là phần Anomaly Detection của đề tài")

        col1, col2 = st.columns(2)

        with col1:
            # PCA scatter
            if 'pca1' in final_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=clean['pca1'], y=clean['pca2'], mode='markers',
                    name=f'Clean ({len(clean):,})',
                    marker=dict(size=4, color='steelblue', opacity=0.2),
                ))
                fig.add_trace(go.Scatter(
                    x=noise['pca1'], y=noise['pca2'], mode='markers',
                    name=f'Noise ({len(noise):,})',
                    marker=dict(size=7, color='crimson', opacity=0.8,
                                line=dict(width=0.5, color='darkred')),
                ))
                fig.update_layout(title="PCA: Noise vs Clean",
                                  xaxis_title="PCA1", yaxis_title="PCA2", height=420)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Energy histogram
            e_noise = noise['relative_energy']
            e_clean = clean['relative_energy']
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=e_clean, name='Clean', opacity=0.55,
                                       marker_color='steelblue', nbinsx=50))
            fig.add_trace(go.Histogram(x=e_noise, name='Noise', opacity=0.55,
                                       marker_color='crimson', nbinsx=50))
            fig.add_vline(x=e_clean.mean(), line_dash='dash', line_color='steelblue',
                          annotation_text=f'Clean {e_clean.mean():.4f}')
            fig.add_vline(x=e_noise.mean(), line_dash='dash', line_color='crimson',
                          annotation_text=f'Noise {e_noise.mean():.4f}')
            diff = e_noise.mean() - e_clean.mean()
            fig.update_layout(
                barmode='overlay', height=420,
                title=f"Energy: Noise kém ổn định hơn {diff:+.4f} eV/atom",
                xaxis_title='Relative Energy (eV/atom)', yaxis_title='Count',
            )
            st.plotly_chart(fig, use_container_width=True)

        # Crystal system
        if 'crystal_system' in final_df.columns:
            cs_n = noise['crystal_system'].value_counts(normalize=True) * 100
            cs_c = clean['crystal_system'].value_counts(normalize=True) * 100
            cs_df = pd.DataFrame({'Clean (%)': cs_c, 'Noise (%)': cs_n}).fillna(0).round(1)
            st.markdown("**Crystal System: Noise vs Clean**")
            st.dataframe(cs_df, use_container_width=True)

    # ── TANG 2 ───────────────────────────────────────────────────────────────
    with t2:
        st.markdown("#### Tầng 2: K-means Macro-clustering (k=3)")
        st.info("Chạy trên **9,367 điểm sạch** — 3 nhóm cấu trúc lớn")

        col1, col2 = st.columns(2)

        with col1:
            if 'pca1' in clean.columns and 'kmeans_cluster' in clean.columns:
                colors_km = ['#2ecc71', '#3498db', '#e74c3c']
                fig = go.Figure()
                for i, c_id in enumerate(sorted(clean['kmeans_cluster'].unique())):
                    mask = clean['kmeans_cluster'] == c_id
                    lbl  = clean.loc[mask, 'kmeans_label'].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=clean.loc[mask, 'pca1'], y=clean.loc[mask, 'pca2'],
                        mode='markers', name=f'{lbl} ({mask.sum():,})',
                        marker=dict(size=5, color=colors_km[i % 3], opacity=0.35),
                    ))
                fig.update_layout(title="PCA: K-means Macro-clusters",
                                  xaxis_title="PCA1", yaxis_title="PCA2", height=420)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'kmeans_label' in clean.columns and 'relative_energy' in clean.columns:
                km_order = clean.groupby('kmeans_label')['relative_energy'].mean().sort_values().index.tolist()
                fig = go.Figure()
                for i, lbl in enumerate(km_order):
                    sub = clean[clean['kmeans_label'] == lbl]['relative_energy']
                    fig.add_trace(go.Box(y=sub, name=lbl,
                                         marker_color=colors_km[i % 3], boxmean=True))
                fig.update_layout(title="Energy Distribution by K-means Cluster",
                                  yaxis_title="Relative Energy (eV/atom)", height=420)
                st.plotly_chart(fig, use_container_width=True)

        # Profile table
        if 'kmeans_label' in clean.columns:
            num_cols = ['relative_energy', 'volume_per_atom', 'mean_bond_length',
                        'std_coordination', 'angle_deviation', 'num_atoms']
            num_cols = [c for c in num_cols if c in clean.columns]
            profile_tbl = clean.groupby('kmeans_label')[num_cols].mean().round(4)
            st.markdown("**Cluster Profile (mean values)**")
            st.dataframe(profile_tbl, use_container_width=True)

    # ── TANG 3 ───────────────────────────────────────────────────────────────
    with t3:
        st.markdown("#### Tầng 3: GMM Micro-clustering (k=10)")
        st.info("10 sub-clusters (polymorphs) với xác suất membership — BIC = -357,009")

        col1, col2 = st.columns(2)

        with col1:
            if 'pca1' in gmm_df.columns and 'gmm_cluster' in gmm_df.columns:
                fig = go.Figure()
                cmap_colors = px.colors.qualitative.Plotly  # 10 màu chuẩn
                for c_id in sorted(gmm_df['gmm_cluster'].unique()):
                    mask = gmm_df['gmm_cluster'] == c_id
                    fig.add_trace(go.Scatter(
                        x=gmm_df.loc[mask, 'pca1'], y=gmm_df.loc[mask, 'pca2'],
                        mode='markers', name=f'GMM-{c_id} ({mask.sum():,})',
                        marker=dict(size=5, color=cmap_colors[c_id % 10], opacity=0.4),
                    ))
                fig.update_layout(title="PCA: GMM Micro-clusters",
                                  xaxis_title="PCA1", yaxis_title="PCA2", height=480)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # GMM summary table
            rows = []
            for c_id in sorted(gmm_df['gmm_cluster'].unique()):
                sub    = gmm_df[gmm_df['gmm_cluster'] == c_id]
                km_maj = sub['kmeans_label'].mode()[0]
                rows.append({
                    'GMM Cluster': f'GMM-{c_id}',
                    'K-means Macro': km_maj,
                    'N': len(sub),
                    'Energy (mean)': round(sub['relative_energy'].mean(), 4),
                    'Prob (mean)': round(sub['gmm_probability'].mean(), 3),
                })
            gmm_tbl = pd.DataFrame(rows)
            st.markdown("**GMM Cluster Summary**")
            st.dataframe(gmm_tbl, use_container_width=True, hide_index=True)

        # Heatmap profile
        num_cols = ['relative_energy', 'volume_per_atom', 'mean_bond_length',
                    'std_coordination', 'angle_deviation', 'num_atoms']
        num_cols = [c for c in num_cols if c in gmm_profile.columns]
        if num_cols:
            profile_sub  = gmm_profile[num_cols].copy()
            profile_norm = (profile_sub - profile_sub.min()) / (profile_sub.max() - profile_sub.min() + 1e-9)

            fig = go.Figure(go.Heatmap(
                z=profile_norm.values,
                x=num_cols,
                y=[f'GMM-{i}' for i in profile_norm.index],
                colorscale='RdYlGn_r',
                text=[[f'{v:.2f}' for v in row] for row in profile_norm.values],
                texttemplate='%{text}',
                textfont=dict(size=11),
                zmin=0, zmax=1,
            ))
            fig.update_layout(title='GMM Micro-cluster Profile (normalized)',
                              xaxis_title='Feature', yaxis_title='GMM Cluster',
                              height=500)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── DOWNLOAD ─────────────────────────────────────────────────────────────
    st.markdown("### 📥 Download Kết quả")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = noise_df.to_csv(index=False)
        st.download_button("Noise (Tier 1)", csv, "tier1_noise.csv", "text/csv")
    with col2:
        csv = clean.to_csv(index=False)
        st.download_button("K-means Clean (Tier 2)", csv, "tier2_kmeans.csv", "text/csv")
    with col3:
        csv = gmm_df.to_csv(index=False)
        st.download_button("GMM Clean (Tier 3)", csv, "tier3_gmm.csv", "text/csv")


def render_anomaly_detection_tab():
    st.markdown('<div class="sub-header">🔍 Phát hiện Dị biệt (Anomaly Detection)</div>', unsafe_allow_html=True)

    st.info(
        "**Nguyên tắc:** Phát hiện dựa trên features **cấu trúc** (lattice, bond, coordination) — "
        "không dùng energy làm input. Energy chỉ dùng để **diễn giải** kết quả sau khi phát hiện."
    )

    results_df, summary_df, comparison_df, details_df = load_anomaly_detection_results()

    if results_df is None:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng chạy:")
        st.code("python carbon24_anomaly_detection.py")
        return

    # ── 1. METRICS ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Tổng quan")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng mẫu", f"{len(results_df):,}")
    c2.metric("HDBSCAN Noise",
              f"{results_df['is_hdbscan_noise'].sum():,}",
              f"{results_df['is_hdbscan_noise'].mean():.1%}")
    c3.metric("Low Probability",
              f"{results_df['is_low_probability'].sum():,}",
              f"{results_df['is_low_probability'].mean():.1%}")
    c4.metric("Isolation Forest",
              f"{results_df['is_isolation_forest_anomaly'].sum():,}",
              f"{results_df['is_isolation_forest_anomaly'].mean():.1%}")
    c5.metric("Consensus ★ (≥2)",
              f"{results_df['is_anomaly_consensus'].sum():,}",
              f"{results_df['is_anomaly_consensus'].mean():.1%}",
              help="Được ít nhất 2/3 phương pháp đồng ý — khuyến nghị sử dụng")

    st.markdown("---")

    # ── 2. SO SÁNH PHƯƠNG PHÁP ──────────────────────────────────────────────
    st.markdown("### 🔬 So sánh các phương pháp")
    col1, col2 = st.columns(2)

    with col1:
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
        fig = go.Figure(go.Bar(
            y=comparison_df['method'],
            x=comparison_df['anomaly_ratio'] * 100,
            orientation='h',
            text=[f"{r:.1%}  ({n:,})"
                  for r, n in zip(comparison_df['anomaly_ratio'], comparison_df['n_anomalies'])],
            textposition='outside',
            marker=dict(color=colors[:len(comparison_df)]),
        ))
        fig.update_layout(title="Tỷ lệ Anomaly theo phương pháp",
                          xaxis_title="Anomaly Ratio (%)", height=380, showlegend=False,
                          margin=dict(l=10, r=120))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        vc = results_df['anomaly_vote_count'].value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=vc.index, y=vc.values,
            text=[f"{v:,}<br>({v/len(results_df):.1%})" for v in vc.values],
            textposition='outside',
            marker=dict(color=['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad'][:len(vc)]),
        ))
        fig.update_layout(
            title="Mức độ đồng thuận giữa các phương pháp",
            xaxis=dict(tickmode='array', tickvals=[0,1,2,3],
                       ticktext=['0<br>(Normal)', '1 method', '2 methods', '3 methods']),
            yaxis_title="Số mẫu", height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 3. PCA VISUALIZATION ────────────────────────────────────────────────
    st.markdown("### 🗺️ Phân bố không gian (PCA)")
    pca_cols = [c for c in results_df.columns if c.lower().startswith('pca')]

    if len(pca_cols) >= 2:
        method_options = {
            'HDBSCAN Noise':     'is_hdbscan_noise',
            'Low Probability':   'is_low_probability',
            'Isolation Forest':  'is_isolation_forest_anomaly',
            'Consensus (>=2) *': 'is_anomaly_consensus',
            'All 3 methods':     'is_anomaly_all',
        }
        sel = st.selectbox("Chon phuong phap:", list(method_options.keys()), index=3)
        mcol = method_options[sel]
        nm = results_df[mcol] == 0
        am = results_df[mcol] == 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df.loc[nm, pca_cols[0]], y=results_df.loc[nm, pca_cols[1]],
            mode='markers', name=f'Normal ({nm.sum():,})',
            marker=dict(size=4, color='steelblue', opacity=0.25),
            hovertemplate='Normal<br>PCA1=%{x:.2f}<br>PCA2=%{y:.2f}<extra></extra>',
        ))
        fig.add_trace(go.Scatter(
            x=results_df.loc[am, pca_cols[0]], y=results_df.loc[am, pca_cols[1]],
            mode='markers', name=f'Anomaly ({am.sum():,})',
            marker=dict(size=8, color='crimson', opacity=0.75,
                        line=dict(width=0.8, color='darkred')),
            hovertemplate='<b>Anomaly</b><br>PCA1=%{x:.2f}<br>PCA2=%{y:.2f}<extra></extra>',
        ))
        fig.update_layout(
            title=f"{sel} — {am.sum():,} anomalies ({am.mean():.2%})",
            xaxis_title=pca_cols[0], yaxis_title=pca_cols[1],
            height=560, hovermode='closest',
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Khong tim thay cot PCA")

    st.markdown("---")

    # ── 4. DIEN GIAI NANG LUONG ─────────────────────────────────────────────
    st.markdown("### ⚡ Diễn giải Năng lượng")
    st.caption("Energy **không** dùng để phát hiện — chỉ dùng để xác nhận ý nghĩa vật lý.")

    if 'relative_energy' in results_df.columns:
        col1, col2 = st.columns(2)

        with col1:
            rows = []
            for c, lbl in [
                ('is_hdbscan_noise',            'HDBSCAN Noise'),
                ('is_low_probability',          'Low Probability'),
                ('is_isolation_forest_anomaly', 'Isolation Forest'),
                ('is_anomaly_consensus',        'Consensus (>=2) *'),
                ('is_anomaly_all',              'All 3 methods'),
            ]:
                if c not in results_df.columns:
                    continue
                a = results_df.loc[results_df[c]==1, 'relative_energy']
                n = results_df.loc[results_df[c]==0, 'relative_energy']
                diff = a.mean() - n.mean()
                rows.append({
                    'Phuong phap': lbl,
                    'Anomaly (eV/atom)': f'{a.mean():.4f}',
                    'Normal (eV/atom)':  f'{n.mean():.4f}',
                    'Chenh lech':        f'{diff:+.4f}',
                    'Dien giai': 'kem on dinh' if diff > 0.01 else 'on dinh hon' if diff < -0.01 else 'tuong duong',
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with col2:
            msel = st.selectbox(
                "Histogram cho phuong phap:",
                ['is_anomaly_consensus', 'is_hdbscan_noise',
                 'is_low_probability', 'is_isolation_forest_anomaly'],
                format_func=lambda x: x.replace('is_', '').replace('_', ' ').title(),
                key='energy_hist_sel',
            )
            ne = results_df.loc[results_df[msel]==0, 'relative_energy']
            ae = results_df.loc[results_df[msel]==1, 'relative_energy']
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ne, name='Normal',  opacity=0.55,
                                       marker_color='steelblue', nbinsx=50))
            fig.add_trace(go.Histogram(x=ae, name='Anomaly', opacity=0.55,
                                       marker_color='crimson',   nbinsx=50))
            fig.add_vline(x=ne.mean(), line_dash='dash', line_color='steelblue',
                          annotation_text=f'Normal {ne.mean():.3f}')
            fig.add_vline(x=ae.mean(), line_dash='dash', line_color='crimson',
                          annotation_text=f'Anomaly {ae.mean():.3f}')
            fig.update_layout(barmode='overlay', height=380,
                              xaxis_title='Relative Energy (eV/atom)', yaxis_title='Count',
                              title='Phan bo Relative Energy')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 5. ISOLATION FOREST SCORE ────────────────────────────────────────────
    if 'isolation_forest_score' in results_df.columns:
        st.markdown("### 🌲 Isolation Forest Score")
        st.caption("Score cang am -> cau truc cang de bi co lap -> cang bat thuong.")
        ns  = results_df.loc[results_df['is_isolation_forest_anomaly']==0, 'isolation_forest_score']
        as_ = results_df.loc[results_df['is_isolation_forest_anomaly']==1, 'isolation_forest_score']
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=ns,  name='Normal',  opacity=0.55,
                                   marker_color='steelblue', nbinsx=60))
        fig.add_trace(go.Histogram(x=as_, name='Anomaly', opacity=0.55,
                                   marker_color='crimson',   nbinsx=60))
        fig.add_vline(x=ns.mean(),  line_dash='dash', line_color='steelblue',
                      annotation_text=f'Normal {ns.mean():.3f}')
        fig.add_vline(x=as_.mean(), line_dash='dash', line_color='crimson',
                      annotation_text=f'Anomaly {as_.mean():.3f}')
        fig.update_layout(barmode='overlay', height=360,
                          xaxis_title='Isolation Forest Score', yaxis_title='Count',
                          title='Isolation Forest Score Distribution')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

    # ── 6. OVERLAP / JACCARD ─────────────────────────────────────────────────
    st.markdown("### 🔗 Overlap giữa các phương pháp")
    method_pairs = [
        ('is_hdbscan_noise',            'HDBSCAN Noise'),
        ('is_low_probability',          'Low Probability'),
        ('is_isolation_forest_anomaly', 'Isolation Forest'),
    ]
    labels = [m[1] for m in method_pairs]
    import numpy as _np
    jac = _np.zeros((3, 3))
    for i, (c1, _) in enumerate(method_pairs):
        for j, (c2, _) in enumerate(method_pairs):
            s1 = set(results_df.index[results_df[c1]==1])
            s2 = set(results_df.index[results_df[c2]==1])
            jac[i, j] = len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0
    fig = go.Figure(go.Heatmap(
        z=jac, x=labels, y=labels,
        colorscale='YlOrRd', zmin=0, zmax=1,
        text=[[f'{v:.3f}' for v in row] for row in jac],
        texttemplate='%{text}', textfont=dict(size=14),
    ))
    fig.update_layout(title='Jaccard Similarity giua cac phuong phap', height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # ── 7. DANH SACH ANOMALIES ───────────────────────────────────────────────
    st.markdown("### 📋 Chi tiết Consensus Anomalies")
    if details_df is not None and len(details_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            vote_filter = st.multiselect("Loc vote count:", [2, 3], default=[2, 3])
        with col2:
            n_show = st.slider("So hang hien thi:", 10, min(100, len(details_df)), 20, 10)
        filtered = details_df[details_df['anomaly_vote_count'].isin(vote_filter)]
        st.markdown(f"**{len(filtered):,} anomalies** (vote in {vote_filter})")
        st.dataframe(filtered.head(n_show), use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", filtered.to_csv(index=False),
                           "consensus_anomalies.csv", "text/csv")
    else:
        st.info("Khong co consensus anomalies")

    st.markdown("---")

    # ── 8. KHUYEN NGHI ───────────────────────────────────────────────────────
    st.markdown("### 💡 Khuyến nghị sử dụng")
    c1, c2, c3 = st.columns(3)
    c1.success("**Phan tich sau**\n\n`is_anomaly_all`\n\nCa 3 phuong phap dong y\n\nHigh precision")
    c2.warning("**Loc du lieu (khuyen nghi)**\n\n`is_anomaly_consensus`\n\n>=2/3 phuong phap\n\nCan bang precision/recall")
    c3.info("**Kham pha**\n\n`is_anomaly_any`\n\n>=1 phuong phap\n\nHigh recall")



# ============================================================================
# ENERGY PREDICTION — DATA LOADING & RENDER
# ============================================================================

@st.cache_data
def load_energy_results():
    try:
        lb   = pd.read_csv('carbon24_energy_results/leaderboard.csv')
        pred = pd.read_csv('carbon24_energy_results/predictions_test.csv')
        fi   = pd.read_csv('carbon24_energy_results/feature_importance.csv', index_col=0)
        return lb, pred, fi
    except Exception as e:
        st.error(f"Lỗi load energy results: {e}")
        return None, None, None


def render_energy_prediction_tab():
    st.markdown('<div class="sub-header">⚡ Dự đoán Năng lượng</div>', unsafe_allow_html=True)

    st.info(
        "**Model Leaderboard** — 4 mô hình dự đoán `relative_energy` (eV/atom) "
        "sử dụng 27 features: structural + cluster labels + scientific labels từ Ground-Truth Labeling."
    )

    lb, pred_df, fi_df = load_energy_results()
    if lb is None:
        st.warning("Chưa có kết quả. Vui lòng chạy:")
        st.code("python carbon24_energy_prediction.py")
        return

    # ── LEADERBOARD METRICS ──────────────────────────────────────────────────
    st.markdown("### 🏆 Model Leaderboard")

    best = lb.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🥇 Best Model",  best["Model"])
    c2.metric("Test RMSE",      f"{best['test_RMSE']:.5f} eV/atom")
    c3.metric("Test R²",        f"{best['test_R2']:.4f}")
    c4.metric("Test MAPE",      f"{best['test_MAPE%']:.2f}%")

    # Leaderboard table
    display_cols = ["Model", "test_RMSE", "test_MAE", "test_R2", "test_MAPE%",
                    "val_RMSE", "val_R2", "Train_time_s"]
    display_cols = [c for c in display_cols if c in lb.columns]

    def highlight_best(s):
        if s.name in ["test_RMSE", "test_MAE", "val_RMSE"]:
            return ["background-color: #d4edda" if v == s.min() else "" for v in s]
        if s.name in ["test_R2", "val_R2"]:
            return ["background-color: #d4edda" if v == s.max() else "" for v in s]
        return [""] * len(s)

    st.dataframe(
        lb[display_cols].style.apply(highlight_best).format({
            "test_RMSE": "{:.6f}", "test_MAE": "{:.6f}", "test_R2": "{:.4f}",
            "test_MAPE%": "{:.2f}", "val_RMSE": "{:.6f}", "val_R2": "{:.4f}",
            "Train_time_s": "{:.2f}s",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # ── TABS ─────────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "📊 Leaderboard Charts",
        "🎯 Predictions",
        "🌟 Feature Importance",
        "📉 Residuals",
    ])

    # ── TAB 1: LEADERBOARD CHARTS ────────────────────────────────────────────
    with t1:
        col1, col2 = st.columns(2)

        with col1:
            # RMSE comparison
            fig = go.Figure()
            colors_lb = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
            for i, (split, dash) in enumerate([("train", "dot"), ("val", "dash"), ("test", "solid")]):
                col_name = f"{split}_RMSE"
                if col_name not in lb.columns:
                    continue
                fig.add_trace(go.Bar(
                    name=split.capitalize(),
                    x=lb["Model"], y=lb[col_name],
                    marker_color=["#3498db", "#f39c12", "#e74c3c"][i],
                    opacity=0.8,
                    text=[f"{v:.5f}" for v in lb[col_name]],
                    textposition="outside",
                ))
            fig.update_layout(
                barmode="group", title="RMSE: Train / Val / Test",
                yaxis_title="RMSE (eV/atom)", height=420,
                xaxis_tickangle=-15,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # R2 comparison
            fig = go.Figure()
            for split, color in [("val", "#f39c12"), ("test", "#e74c3c")]:
                col_name = f"{split}_R2"
                if col_name not in lb.columns:
                    continue
                fig.add_trace(go.Bar(
                    name=split.capitalize(),
                    x=lb["Model"], y=lb[col_name],
                    marker_color=color, opacity=0.8,
                    text=[f"{v:.4f}" for v in lb[col_name]],
                    textposition="outside",
                ))
            fig.add_hline(y=0.95, line_dash="dash", line_color="green",
                          annotation_text="R²=0.95 threshold")
            fig.update_layout(
                barmode="group", title="R² Score: Val / Test",
                yaxis_title="R²", height=420,
                xaxis_tickangle=-15, yaxis_range=[0.6, 1.02],
            )
            st.plotly_chart(fig, use_container_width=True)

        # Model tier explanation
        st.markdown("**Giải thích các tầng mô hình:**")
        tier_data = {
            "Tầng": ["Baseline", "Trung cấp", "Cao cấp 1", "Cao cấp 2"],
            "Mô hình": ["Ridge Regression", "Random Forest", "LightGBM", "CatBoost"],
            "Đặc điểm": [
                "Linear, fast, interpretable — baseline reference",
                "Ensemble trees, robust, feature importance",
                "Gradient boosting, fast, handles large data",
                "Tối ưu cho categorical features (scientific_label, crystal_system)",
            ],
        }
        st.dataframe(pd.DataFrame(tier_data), use_container_width=True, hide_index=True)

    # ── TAB 2: PREDICTIONS ───────────────────────────────────────────────────
    with t2:
        st.markdown("#### Predicted vs Actual (Test Set)")

        # Chon model
        pred_model_cols = [c for c in pred_df.columns if c.startswith("pred_") and c != "pred_best"]
        model_display   = {c: c.replace("pred_", "").replace("_", " ") for c in pred_model_cols}

        sel_col = st.selectbox(
            "Chọn mô hình:",
            pred_model_cols,
            format_func=lambda x: model_display[x],
        )

        y_true = pred_df["relative_energy"].values
        y_pred = pred_df[sel_col].values
        r2   = float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_true, y=y_pred, mode="markers",
                marker=dict(size=4, color="#3498db", opacity=0.4),
                name="Predictions",
                hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>",
            ))
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            fig.add_trace(go.Scatter(
                x=lims, y=lims, mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Perfect prediction",
            ))
            fig.update_layout(
                title=f"{model_display[sel_col]} — R²={r2:.4f}  RMSE={rmse:.5f}",
                xaxis_title="Actual Relative Energy (eV/atom)",
                yaxis_title="Predicted Relative Energy (eV/atom)",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("R²",   f"{r2:.4f}")
            st.metric("RMSE", f"{rmse:.5f} eV/atom")
            mae = float(np.mean(np.abs(y_true - y_pred)))
            st.metric("MAE",  f"{mae:.5f} eV/atom")

            # Error by cluster
            if "kmeans_label" in pred_df.columns:
                st.markdown("**Error by K-means cluster:**")
                err_col = sel_col.replace("pred_", "err_")
                if err_col in pred_df.columns:
                    err_by_cluster = pred_df.groupby("kmeans_label")[err_col].agg(
                        lambda x: np.sqrt(np.mean(x**2))
                    ).round(5).to_frame("RMSE")
                    st.dataframe(err_by_cluster, use_container_width=True)

        # Predictions table
        st.markdown("**Sample predictions (first 50):**")
        show_cols = ["material_id", "relative_energy", sel_col,
                     "kmeans_label", "scientific_label", "crystal_system"]
        show_cols = [c for c in show_cols if c in pred_df.columns]
        st.dataframe(pred_df[show_cols].head(50), use_container_width=True, hide_index=True)

        # Download
        st.download_button(
            "📥 Download All Predictions (CSV)",
            pred_df.to_csv(index=False),
            "energy_predictions.csv", "text/csv",
        )

    # ── TAB 3: FEATURE IMPORTANCE ────────────────────────────────────────────
    with t3:
        st.markdown("#### Feature Importance")

        if fi_df is not None and not fi_df.empty:
            model_fi_cols = fi_df.columns.tolist()
            sel_fi = st.selectbox("Chọn mô hình:", model_fi_cols, key="fi_sel")

            top_n = st.slider("Top N features:", 10, min(30, len(fi_df)), 20)

            fi_series = fi_df[sel_fi].abs().sort_values(ascending=False).head(top_n)

            fig = go.Figure(go.Bar(
                x=fi_series.values[::-1],
                y=fi_series.index[::-1],
                orientation="h",
                marker_color="#3498db",
                opacity=0.8,
            ))
            fig.update_layout(
                title=f"Top {top_n} Feature Importance — {sel_fi}",
                xaxis_title="Importance",
                height=max(400, top_n * 22),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature groups
            st.markdown("**Feature groups:**")
            struct_fi = fi_df[sel_fi].reindex([f for f in [
                "num_atoms","a","b","c","alpha","beta","gamma",
                "volume","volume_per_atom","b_over_a","c_over_a","angle_deviation",
                "mean_bond_length","std_bond_length","min_bond_length","max_bond_length",
                "std_coordination","min_coordination","max_coordination",
            ] if f in fi_df.index]).abs().sum()
            cluster_fi = fi_df[sel_fi].reindex([f for f in [
                "kmeans_cluster","gmm_cluster","hdbscan_probability","pca1","pca2",
            ] if f in fi_df.index]).abs().sum()
            cat_fi = fi_df[sel_fi].reindex([f for f in [
                "crystal_system_enc","space_group_symbol_enc","scientific_label_enc",
            ] if f in fi_df.index]).abs().sum()
            total = struct_fi + cluster_fi + cat_fi
            if total > 0:
                group_df = pd.DataFrame({
                    "Feature Group": ["Structural (19)", "Cluster Labels (5)", "Categorical (3)"],
                    "Total Importance": [struct_fi, cluster_fi, cat_fi],
                    "Percentage": [f"{v/total*100:.1f}%" for v in [struct_fi, cluster_fi, cat_fi]],
                })
                st.dataframe(group_df, use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance không có sẵn")

    # ── TAB 4: RESIDUALS ─────────────────────────────────────────────────────
    with t4:
        st.markdown("#### Residual Analysis")

        sel_res = st.selectbox(
            "Chọn mô hình:",
            pred_model_cols,
            format_func=lambda x: model_display[x],
            key="res_sel",
        )

        y_true = pred_df["relative_energy"].values
        y_pred = pred_df[sel_res].values
        residuals = y_true - y_pred

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_pred, y=residuals, mode="markers",
                marker=dict(size=4, color="#e74c3c", opacity=0.4),
                name="Residuals",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=2)
            fig.update_layout(
                title="Residuals vs Predicted",
                xaxis_title="Predicted (eV/atom)",
                yaxis_title="Residual (eV/atom)",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=residuals, nbinsx=60,
                marker_color="#e74c3c", opacity=0.7,
                name="Residuals",
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            fig.add_vline(x=residuals.mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Mean={residuals.mean():.4f}")
            fig.update_layout(
                title=f"Residual Distribution (std={residuals.std():.4f})",
                xaxis_title="Residual (eV/atom)",
                yaxis_title="Count",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Residuals by scientific label
        if "scientific_label" in pred_df.columns:
            st.markdown("**RMSE by Scientific Label:**")
            err_col = sel_res.replace("pred_", "err_")
            if err_col in pred_df.columns:
                rmse_by_sci = pred_df.groupby("scientific_label")[err_col].agg(
                    lambda x: np.sqrt(np.mean(x**2))
                ).sort_values(ascending=False).round(5).to_frame("RMSE")
                st.dataframe(rmse_by_sci, use_container_width=True)


# ============================================================================
