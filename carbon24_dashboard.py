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
st.markdown('<div class="main-header"> Carbon-24 Data Mining Dashboard</div>', unsafe_allow_html=True)
st.markdown("**Phân cụm, Phát hiện dị biệt và Dự đoán năng lượng cấu trúc Carbon-24**")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Carbon-24", use_container_width=True)
    st.markdown("###  Navigation")
    
    page = st.radio(
        "Chọn trang:",
        [" Tổng quan", " Khảo sát dữ liệu",
         " Phân cụm K-means", " Phân cụm GMM",
         " Phân cụm Hierarchical", " Phân cụm HDBSCAN",
         " So sánh thuật toán", " Phát hiện dị biệt",
         " Pipeline 3 Tầng", " Ground-Truth Labeling", " Dự đoán năng lượng"]
    )
    
    st.markdown("---")
    st.markdown("###  Thông tin")
    st.info("""
    **Dự án:** Khai thác dữ liệu Carbon-24
    
    **Phương pháp:**
    - K-means Clustering
    - Isolation Forest
    - Regression Models
    
    **Dataset:** 10,153 samples
    """)

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
# PAGE: TỔNG QUAN
# ============================================================================
if page == " Tổng quan":
    st.markdown('<div class="sub-header"> Tổng Quan Dự Án</div>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(" Tổng số mẫu", f"{len(df):,}")
    
    with col2:
        if feature_info:
            st.metric(" Features", len(feature_info['numeric_features']))
        else:
            st.metric(" Features", len(df.columns))
    
    with col3:
        if has_clusters:
            n_clusters = df['cluster'].nunique()
            st.metric(" Số clusters", n_clusters)
        else:
            st.metric(" Số clusters", "Chưa phân cụm")
    
    with col4:
        if 'crystal_system' in df.columns:
            st.metric(" Crystal systems", df['crystal_system'].nunique())
        else:
            st.metric(" Crystal systems", "N/A")
    
    st.markdown("---")
    
    # Project workflow
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("###  Quy Trình Xử Lý")
        
        workflow = """
        ```mermaid
        graph TD
            A[Raw Data<br/>10,153 samples, 41 features] --> B[Preprocessing]
            B --> C[Feature Engineering<br/>+4 features]
            C --> D[Feature Selection<br/>-14 features, 31 remaining]
            D --> E[Normalization<br/>StandardScaler]
            E --> F[K-means Clustering<br/>k=3 clusters]
            E --> G[Anomaly Detection<br/>Coming Soon]
            E --> H[Energy Prediction<br/>Coming Soon]
        ```
        """
        st.markdown(workflow)
    
    with col2:
        st.markdown("###  Tiến Độ")
        
        progress_data = {
            'Bước': ['Preprocessing', 'Feature Selection', 'Clustering', 'Anomaly Detection', 'Prediction'],
            'Hoàn thành': [100, 100, 100 if has_clusters else 0, 0, 0]
        }
        progress_df = pd.DataFrame(progress_data)
        
        fig = px.bar(progress_df, x='Hoàn thành', y='Bước', orientation='h',
                     color='Hoàn thành', color_continuous_scale='Blues',
                     range_x=[0, 100])
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Dataset preview
    st.markdown("###  Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# PAGE: KHẢO SÁT DỮ LIỆU
# ============================================================================
elif page == " Khảo sát dữ liệu":
    st.markdown('<div class="sub-header"> Khảo Sát Dữ Liệu</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([" Phân phối", " Tương quan", " Crystal Systems"])
    
    with tab1:
        st.markdown("#### Phân Phối Features")
        
        if feature_info:
            numeric_features = feature_info['numeric_features']
        else:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        selected_feature = st.selectbox("Chọn feature:", numeric_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=selected_feature, nbins=50,
                             title=f"Histogram: {selected_feature}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=selected_feature,
                        title=f"Box Plot: {selected_feature}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("#### Thống Kê Mô Tả")
        stats = df[selected_feature].describe()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Mean", f"{stats['mean']:.4f}")
        col2.metric("Std", f"{stats['std']:.4f}")
        col3.metric("Min", f"{stats['min']:.4f}")
        col4.metric("Max", f"{stats['max']:.4f}")
        col5.metric("Median", f"{stats['50%']:.4f}")
    
    with tab2:
        st.markdown("#### Ma Trận Tương Quan")
        
        if feature_info:
            numeric_features = feature_info['numeric_features'][:15]  # Top 15 features
        else:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()[:15]
        
        corr_matrix = df[numeric_features].corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns,
                       color_continuous_scale='RdBu_r',
                       zmin=-1, zmax=1)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if 'crystal_system' in df.columns:
            st.markdown("#### Phân Bố Crystal Systems")
            
            crystal_counts = df['crystal_system'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(x=crystal_counts.index, y=crystal_counts.values,
                           labels={'x': 'Crystal System', 'y': 'Count'},
                           title="Crystal System Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(values=crystal_counts.values, names=crystal_counts.index,
                           title="Crystal System Proportion")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.markdown("#### Chi Tiết")
            crystal_df = pd.DataFrame({
                'Crystal System': crystal_counts.index,
                'Count': crystal_counts.values,
                'Percentage': (crystal_counts.values / len(df) * 100).round(2)
            })
            st.dataframe(crystal_df, use_container_width=True)
        else:
            st.warning("Không có thông tin crystal system trong dataset")

# ============================================================================
# PAGE: PHÂN CỤM K-MEANS
# ============================================================================
elif page == " Phân cụm K-means":
    st.markdown('<div class="sub-header"> Phân Cụm K-means</div>', unsafe_allow_html=True)
    
    if not has_clusters:
        st.warning("Chưa có kết quả phân cụm. Vui lòng chạy notebook K-means trước.")
        st.info("Chạy: `carbon24-kmeans-clustering.ipynb`")
    else:
        # Clustering metrics
        if clustering_report:
            st.markdown("####  Clustering Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = clustering_report['metrics']
            col1.metric("Silhouette Score", f"{metrics['silhouette_score']:.4f}")
            col2.metric("Davies-Bouldin", f"{metrics['davies_bouldin_index']:.4f}")
            col3.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.2f}")
            col4.metric("Inertia", f"{metrics['inertia']:.2f}")
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs([" Cluster Overview", " Visualization", " Analysis"])
        
        with tab1:
            st.markdown("#### Phân Bố Clusters")
            
            cluster_counts = df['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           labels={'x': 'Cluster', 'y': 'Count'},
                           title="Cluster Sizes")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                           title="Cluster Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### PCA Visualization")
            
            # Choose 2D or 3D
            viz_type = st.radio("Visualization type:", ["2D", "3D"], horizontal=True)
            
            if viz_type == "2D":
                if 'pca1' in df.columns and 'pca2' in df.columns:
                    color_by = st.radio("Color by:", ["Cluster", "Relative Energy"], horizontal=True, key="2d_color")
                    
                    if color_by == "Cluster":
                        fig = px.scatter(df, x='pca1', y='pca2', color='cluster',
                                       title="PCA 2D: Clusters",
                                       labels={'pca1': 'PC1', 'pca2': 'PC2'},
                                       color_continuous_scale='viridis')
                    else:
                        if 'relative_energy' in df.columns:
                            fig = px.scatter(df, x='pca1', y='pca2', color='relative_energy',
                                           title="PCA 2D: Relative Energy",
                                           labels={'pca1': 'PC1', 'pca2': 'PC2'},
                                           color_continuous_scale='viridis')
                        else:
                            st.warning("Không có thông tin relative_energy")
                            fig = None
                    
                    if fig:
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Chưa có PCA 2D components. Vui lòng chạy lại notebook.")
            
            else:  # 3D
                if 'pca1_3d' in df.columns and 'pca2_3d' in df.columns and 'pca3_3d' in df.columns:
                    color_by = st.radio("Color by:", ["Cluster", "Relative Energy", "Crystal System"], 
                                       horizontal=True, key="3d_color")
                    
                    if color_by == "Cluster":
                        fig = px.scatter_3d(df, x='pca1_3d', y='pca2_3d', z='pca3_3d',
                                          color='cluster',
                                          title="PCA 3D: Clusters",
                                          labels={'pca1_3d': 'PC1', 'pca2_3d': 'PC2', 'pca3_3d': 'PC3'},
                                          color_continuous_scale='viridis',
                                          opacity=0.7)
                    elif color_by == "Relative Energy":
                        if 'relative_energy' in df.columns:
                            fig = px.scatter_3d(df, x='pca1_3d', y='pca2_3d', z='pca3_3d',
                                              color='relative_energy',
                                              title="PCA 3D: Relative Energy",
                                              labels={'pca1_3d': 'PC1', 'pca2_3d': 'PC2', 'pca3_3d': 'PC3'},
                                              color_continuous_scale='viridis',
                                              opacity=0.7)
                        else:
                            st.warning("Không có thông tin relative_energy")
                            fig = None
                    else:  # Crystal System
                        if 'crystal_system' in df.columns:
                            fig = px.scatter_3d(df, x='pca1_3d', y='pca2_3d', z='pca3_3d',
                                              color='crystal_system',
                                              title="PCA 3D: Crystal Systems",
                                              labels={'pca1_3d': 'PC1', 'pca2_3d': 'PC2', 'pca3_3d': 'PC3'},
                                              opacity=0.7)
                        else:
                            st.warning("Không có thông tin crystal_system")
                            fig = None
                    
                    if fig:
                        fig.update_traces(marker=dict(size=3, line=dict(width=0.5, color='white')))
                        fig.update_layout(
                            height=700,
                            scene=dict(
                                camera=dict(
                                    eye=dict(x=1.5, y=1.5, z=1.3)
                                )
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add controls info
                        with st.expander("🎮 3D Controls"):
                            st.markdown("""
                            - **Rotate**: Click and drag
                            - **Zoom**: Scroll or pinch
                            - **Pan**: Right-click and drag
                            - **Reset**: Double-click
                            """)
                else:
                    st.warning("Chưa có PCA 3D components. Chạy script: `python add_pca_3d_to_notebook.py`")
        
        with tab3:
            st.markdown("#### Phân Tích Theo Cluster")
            
            selected_cluster = st.selectbox("Chọn cluster:", sorted(df['cluster'].unique()))
            
            cluster_data = df[df['cluster'] == selected_cluster]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Số lượng mẫu", len(cluster_data))
                st.metric("Tỷ lệ", f"{len(cluster_data)/len(df)*100:.2f}%")
            
            with col2:
                if 'relative_energy' in df.columns:
                    st.metric("Mean Energy", f"{cluster_data['relative_energy'].mean():.4f}")
                    st.metric("Std Energy", f"{cluster_data['relative_energy'].std():.4f}")
            
            # Feature comparison
            if feature_info:
                st.markdown("#### So Sánh Features")
                
                feature_to_compare = st.selectbox("Chọn feature:", 
                                                 feature_info['numeric_features'])
                
                fig = px.box(df, x='cluster', y=feature_to_compare,
                           title=f"{feature_to_compare} by Cluster")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PAGE: PHÂN CỤM GMM
# ============================================================================
elif page == " Phân cụm GMM":
    st.markdown('<div class="sub-header">🎲 Phân Cụm GMM (Gaussian Mixture Model)</div>', unsafe_allow_html=True)
    
    if gmm_df is None or gmm_report is None:
        st.warning("⚠️ Chưa có kết quả phân cụm GMM. Vui lòng chạy notebook GMM trước.")
        st.info("📝 Chạy: `carbon24-gmm-clustering.ipynb`")
    else:
        # GMM metrics
        st.markdown("####  GMM Clustering Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = gmm_report['metrics']
        col1.metric("Silhouette Score", f"{metrics['silhouette_score']:.4f}")
        col2.metric("Davies-Bouldin", f"{metrics['davies_bouldin_index']:.4f}")
        col3.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.2f}")
        col4.metric("Số Clusters", gmm_report['optimal_n_components'])
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs([" Cluster Overview", " Visualization", " Uncertainty Analysis", " Cluster Profiles"])
        
        with tab1:
            st.markdown("#### Phân Bố Clusters")
            
            cluster_counts = gmm_df['GMM_Cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           labels={'x': 'Cluster', 'y': 'Count'},
                           title="GMM Cluster Sizes")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                           title="GMM Cluster Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # AIC/BIC scores
            st.markdown("#### Model Selection: AIC & BIC Scores")
            st.info(f"""
            **Optimal number of clusters:** {gmm_report['optimal_n_components']}
            
            - **AIC (Akaike Information Criterion):** {gmm_report['metrics']['aic']:.2f}
            - **BIC (Bayesian Information Criterion):** {gmm_report['metrics']['bic']:.2f}
            
            Số clusters được chọn dựa trên BIC thấp nhất.
            """)
        
        with tab2:
            st.markdown("#### PCA Visualization")
            
            if 'PCA1' in gmm_df.columns and 'PCA2' in gmm_df.columns:
                color_by = st.radio("Color by:", ["Cluster", "Max Probability", "Relative Energy"], 
                                   horizontal=True, key="gmm_color")
                
                if color_by == "Cluster":
                    fig = px.scatter(gmm_df, x='PCA1', y='PCA2', color='GMM_Cluster',
                                   title="GMM PCA 2D: Clusters",
                                   labels={'PCA1': 'PC1', 'PCA2': 'PC2'},
                                   color_continuous_scale='viridis')
                elif color_by == "Max Probability":
                    fig = px.scatter(gmm_df, x='PCA1', y='PCA2', color='Max_Probability',
                                   title="GMM PCA 2D: Assignment Probability",
                                   labels={'PCA1': 'PC1', 'PCA2': 'PC2'},
                                   color_continuous_scale='RdYlGn')
                else:
                    if 'relative_energy' in gmm_df.columns:
                        fig = px.scatter(gmm_df, x='PCA1', y='PCA2', color='relative_energy',
                                       title="GMM PCA 2D: Relative Energy",
                                       labels={'PCA1': 'PC1', 'PCA2': 'PC2'},
                                       color_continuous_scale='viridis')
                    else:
                        st.warning("Không có thông tin relative_energy")
                        fig = None
                
                if fig:
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Chưa có PCA components trong kết quả GMM")
        
        with tab3:
            st.markdown("#### Phân Tích Độ Không Chắc Chắn (Uncertainty)")
            
            st.info("""
            GMM là mô hình xác suất, mỗi điểm dữ liệu có xác suất thuộc về mỗi cluster.
            **Max Probability** cho biết độ tin cậy của việc gán cluster.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram of max probabilities
                fig = px.histogram(gmm_df, x='Max_Probability', nbins=50,
                                 title="Distribution of Max Probabilities")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot by cluster
                fig = px.box(gmm_df, x='GMM_Cluster', y='Max_Probability',
                           title="Max Probability by Cluster")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Uncertain samples
            threshold = st.slider("Ngưỡng xác suất (samples dưới ngưỡng = uncertain):", 
                                 0.5, 1.0, 0.7, 0.05)
            uncertain_samples = gmm_df[gmm_df['Max_Probability'] < threshold]
            
            st.markdown(f"#### Samples Không Chắc Chắn (Probability < {threshold})")
            st.metric("Số lượng", f"{len(uncertain_samples):,} ({len(uncertain_samples)/len(gmm_df)*100:.2f}%)")
            
            if len(uncertain_samples) > 0:
                st.dataframe(uncertain_samples.head(10), use_container_width=True)
        
        with tab4:
            st.markdown("#### Cluster Profiles")
            
            if gmm_cluster_profile is not None:
                st.dataframe(gmm_cluster_profile, use_container_width=True)
                
                # Energy analysis by cluster
                if 'relative_energy' in gmm_df.columns:
                    st.markdown("#### Relative Energy by Cluster")
                    
                    fig = px.box(gmm_df, x='GMM_Cluster', y='relative_energy',
                               title="Relative Energy Distribution by GMM Cluster")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    energy_stats = gmm_df.groupby('GMM_Cluster')['relative_energy'].agg(['mean', 'std', 'min', 'max'])
                    st.dataframe(energy_stats, use_container_width=True)

# ============================================================================
# PAGE: PHÂN CỤM HIERARCHICAL
# ============================================================================
elif page == " Phân cụm Hierarchical":
    st.markdown('<div class="sub-header"> Phân Cụm Hierarchical (Agglomerative)</div>', unsafe_allow_html=True)
    
    if hierarchical_df is None:
        st.warning(" Chưa có kết quả phân cụm Hierarchical. Vui lòng chạy notebook Hierarchical trước.")
        st.info(" Chạy notebook hoặc script tạo kết quả Hierarchical clustering")
    else:
        # Hierarchical metrics
        st.markdown("####  Hierarchical Clustering Info")
        
        col1, col2, col3 = st.columns(3)
        
        n_clusters = hierarchical_df['cluster_hierarchical'].nunique()
        col1.metric("Số Clusters", n_clusters)
        col2.metric("Tổng Samples", f"{len(hierarchical_df):,}")
        col3.metric("Linkage Method", "Ward")
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs([" Cluster Overview", " Visualization", " Cluster Interpretation"])
        
        with tab1:
            st.markdown("#### Phân Bố Clusters")
            
            cluster_counts = hierarchical_df['cluster_hierarchical'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           labels={'x': 'Cluster', 'y': 'Count'},
                           title="Hierarchical Cluster Sizes")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart
                fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                           title="Hierarchical Cluster Distribution")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Hierarchical Clustering (Ward Linkage):**
            - Phương pháp bottom-up: bắt đầu với mỗi điểm là 1 cluster
            - Ward linkage: minimize variance khi merge clusters
            - Tạo ra dendrogram thể hiện cấu trúc phân cấp
            """)
        
        with tab2:
            st.markdown("#### PCA Visualization")
            
            if 'PC1' in hierarchical_df.columns and 'PC2' in hierarchical_df.columns:
                color_by = st.radio("Color by:", ["Cluster", "Relative Energy"], 
                                   horizontal=True, key="hier_color")
                
                if color_by == "Cluster":
                    fig = px.scatter(hierarchical_df, x='PC1', y='PC2', 
                                   color='cluster_hierarchical',
                                   title="Hierarchical PCA 2D: Clusters",
                                   labels={'PC1': 'PC1', 'PC2': 'PC2'},
                                   color_continuous_scale='viridis')
                else:
                    if 'relative_energy' in hierarchical_df.columns:
                        fig = px.scatter(hierarchical_df, x='PC1', y='PC2', 
                                       color='relative_energy',
                                       title="Hierarchical PCA 2D: Relative Energy",
                                       labels={'PC1': 'PC1', 'PC2': 'PC2'},
                                       color_continuous_scale='viridis')
                    else:
                        st.warning("Không có thông tin relative_energy")
                        fig = None
                
                if fig:
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Chưa có PCA components trong kết quả Hierarchical")
        
        with tab3:
            st.markdown("#### Cluster Interpretation")
            
            if hierarchical_interpretation is not None:
                st.dataframe(hierarchical_interpretation, use_container_width=True)
            
            # Energy analysis by cluster
            if 'relative_energy' in hierarchical_df.columns:
                st.markdown("#### Relative Energy by Cluster")
                
                fig = px.box(hierarchical_df, x='cluster_hierarchical', y='relative_energy',
                           title="Relative Energy Distribution by Hierarchical Cluster")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                energy_stats = hierarchical_df.groupby('cluster_hierarchical')['relative_energy'].agg(['mean', 'std', 'min', 'max', 'count'])
                st.dataframe(energy_stats, use_container_width=True)
            
            # Cluster analysis
            st.markdown("#### Phân Tích Theo Cluster")
            
            selected_cluster = st.selectbox("Chọn cluster:", 
                                           sorted(hierarchical_df['cluster_hierarchical'].unique()),
                                           key="hier_cluster_select")
            
            cluster_data = hierarchical_df[hierarchical_df['cluster_hierarchical'] == selected_cluster]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Số lượng mẫu", len(cluster_data))
                st.metric("Tỷ lệ", f"{len(cluster_data)/len(hierarchical_df)*100:.2f}%")
            
            with col2:
                if 'relative_energy' in hierarchical_df.columns:
                    st.metric("Mean Energy", f"{cluster_data['relative_energy'].mean():.4f}")
                    st.metric("Std Energy", f"{cluster_data['relative_energy'].std():.4f}")

# ============================================================================
# PAGE: PHÂN CỤM HDBSCAN
# ============================================================================
elif page == " Phân cụm HDBSCAN":
    st.markdown('<div class="sub-header"> Phân Cụm HDBSCAN (Density-Based)</div>', unsafe_allow_html=True)
    
    if hdbscan_df is None:
        st.warning(" Chưa có kết quả phân cụm HDBSCAN. Vui lòng chạy notebook HDBSCAN trước.")
        st.info(" Chạy: `HDBSCAN.ipynb`")
    else:
        # HDBSCAN metrics
        st.markdown("####  HDBSCAN Clustering Info")
        
        col1, col2, col3, col4 = st.columns(4)
        
        n_clusters = len([c for c in hdbscan_df['hdbscan_cluster'].unique() if c != -1])
        n_noise = len(hdbscan_df[hdbscan_df['hdbscan_cluster'] == -1])
        
        col1.metric("Số Clusters", n_clusters)
        col2.metric("Tổng Samples", f"{len(hdbscan_df):,}")
        col3.metric("Noise Points", f"{n_noise:,}")
        col4.metric("Noise %", f"{n_noise/len(hdbscan_df)*100:.2f}%")
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs([" Cluster Overview", " Visualization", " Noise Analysis", " Cluster Profiles"])
        
        with tab1:
            st.markdown("#### Phân Bố Clusters")
            
            # Separate noise and clusters
            clusters_only = hdbscan_df[hdbscan_df['hdbscan_cluster'] != -1]
            cluster_counts = clusters_only['hdbscan_cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart (excluding noise)
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           labels={'x': 'Cluster', 'y': 'Count'},
                           title="HDBSCAN Cluster Sizes (Excluding Noise)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart with noise
                all_counts = hdbscan_df['hdbscan_cluster'].value_counts()
                labels = ['Noise' if x == -1 else f'Cluster {x}' for x in all_counts.index]
                
                fig = px.pie(values=all_counts.values, names=labels,
                           title="HDBSCAN Distribution (Including Noise)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **HDBSCAN (Hierarchical Density-Based Spatial Clustering):**
            - Tự động tìm số clusters dựa trên mật độ
            - Xác định noise points (outliers) - label = -1
            - Không yêu cầu chỉ định số clusters trước
            - Tốt cho clusters có mật độ khác nhau
            """)
        
        with tab2:
            st.markdown("#### PCA Visualization")
            
            if 'pca1' in hdbscan_df.columns and 'pca2' in hdbscan_df.columns:
                color_by = st.radio("Color by:", ["Cluster", "Membership Probability", "Relative Energy"], 
                                   horizontal=True, key="hdbscan_color")
                
                # Create a copy for visualization
                viz_df = hdbscan_df.copy()
                viz_df['cluster_label'] = viz_df['hdbscan_cluster'].apply(
                    lambda x: 'Noise' if x == -1 else f'Cluster {x}'
                )
                
                if color_by == "Cluster":
                    fig = px.scatter(viz_df, x='pca1', y='pca2', 
                                   color='cluster_label',
                                   title="HDBSCAN PCA 2D: Clusters",
                                   labels={'pca1': 'PC1', 'pca2': 'PC2'},
                                   category_orders={'cluster_label': sorted(viz_df['cluster_label'].unique())})
                elif color_by == "Membership Probability":
                    if 'hdbscan_probability' in hdbscan_df.columns:
                        fig = px.scatter(viz_df, x='pca1', y='pca2', 
                                       color='hdbscan_probability',
                                       title="HDBSCAN PCA 2D: Membership Probability",
                                       labels={'pca1': 'PC1', 'pca2': 'PC2'},
                                       color_continuous_scale='RdYlGn')
                    else:
                        st.warning("Không có thông tin membership_probability")
                        fig = None
                else:
                    if 'relative_energy' in hdbscan_df.columns:
                        fig = px.scatter(viz_df, x='pca1', y='pca2', 
                                       color='relative_energy',
                                       title="HDBSCAN PCA 2D: Relative Energy",
                                       labels={'pca1': 'PC1', 'pca2': 'PC2'},
                                       color_continuous_scale='viridis')
                    else:
                        st.warning("Không có thông tin relative_energy")
                        fig = None
                
                if fig:
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Chưa có PCA components trong kết quả HDBSCAN")
        
        with tab3:
            st.markdown("#### Phân Tích Noise Points")
            
            noise_data = hdbscan_df[hdbscan_df['hdbscan_cluster'] == -1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Tổng Noise Points", f"{len(noise_data):,}")
                st.metric("Tỷ lệ Noise", f"{len(noise_data)/len(hdbscan_df)*100:.2f}%")
            
            with col2:
                if 'relative_energy' in noise_data.columns:
                    st.metric("Mean Energy (Noise)", f"{noise_data['relative_energy'].mean():.4f}")
                    st.metric("Std Energy (Noise)", f"{noise_data['relative_energy'].std():.4f}")
            
            # Noise outliers table
            if hdbscan_noise_outliers is not None and len(hdbscan_noise_outliers) > 0:
                st.markdown("#### Top Noise/Outlier Samples")
                st.dataframe(hdbscan_noise_outliers.head(20), use_container_width=True)
            
            # Energy comparison: Noise vs Clusters
            if 'relative_energy' in hdbscan_df.columns:
                st.markdown("#### Energy Comparison: Noise vs Clusters")
                
                viz_df = hdbscan_df.copy()
                viz_df['type'] = viz_df['hdbscan_cluster'].apply(
                    lambda x: 'Noise' if x == -1 else 'Cluster'
                )
                
                fig = px.box(viz_df, x='type', y='relative_energy',
                           title="Relative Energy: Noise vs Clusters")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("#### Cluster Profiles")
            
            if hdbscan_cluster_profile is not None:
                # Exclude noise from profile
                profile_display = hdbscan_cluster_profile[hdbscan_cluster_profile['hdbscan_cluster'] != -1]
                st.dataframe(profile_display, use_container_width=True)
            
            # Energy summary
            if hdbscan_energy_summary is not None:
                st.markdown("#### Energy Summary by Cluster")
                st.dataframe(hdbscan_energy_summary, use_container_width=True)
            
            # Cluster analysis (excluding noise)
            clusters_only = hdbscan_df[hdbscan_df['hdbscan_cluster'] != -1]
            
            if len(clusters_only) > 0:
                st.markdown("#### Phân Tích Theo Cluster")
                
                available_clusters = sorted(clusters_only['hdbscan_cluster'].unique())
                selected_cluster = st.selectbox("Chọn cluster:", available_clusters,
                                               key="hdbscan_cluster_select")
                
                cluster_data = hdbscan_df[hdbscan_df['hdbscan_cluster'] == selected_cluster]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Số lượng mẫu", len(cluster_data))
                    st.metric("Tỷ lệ", f"{len(cluster_data)/len(hdbscan_df)*100:.2f}%")
                
                with col2:
                    if 'relative_energy' in hdbscan_df.columns:
                        st.metric("Mean Energy", f"{cluster_data['relative_energy'].mean():.4f}")
                        st.metric("Std Energy", f"{cluster_data['relative_energy'].std():.4f}")
                
                # Membership probability distribution
                if 'hdbscan_probability' in cluster_data.columns:
                    st.markdown("##### Membership Probability Distribution")
                    fig = px.histogram(cluster_data, x='hdbscan_probability', nbins=30,
                                     title=f"Membership Probability - Cluster {selected_cluster}")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: SO SÁNH THUẬT TOÁN
# ============================================================================
elif page == " So sánh thuật toán":
    st.markdown('<div class="sub-header"> So Sánh Các Thuật Toán Clustering</div>', unsafe_allow_html=True)
    
    if methods_overview is None or quality_metrics is None or method_ranking is None:
        st.warning(" Chưa có kết quả so sánh thuật toán. Vui lòng chạy notebook so sánh trước.")
        st.info(" Chạy: `carbon24-clustering-comparison-evaluation.ipynb`")
    else:
        # Overview metrics
        st.markdown("####  Tổng Quan So Sánh")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(" Số thuật toán", len(methods_overview))
        
        with col2:
            best_method = method_ranking.iloc[0]['Method']
            st.metric(" Thuật toán tốt nhất", best_method)
        
        with col3:
            avg_clusters = methods_overview['Clusters'].mean()
            st.metric(" Trung bình clusters", f"{avg_clusters:.1f}")
        
        with col4:
            total_noise = methods_overview['Noise Points'].sum()
            st.metric(" Tổng noise points", f"{total_noise:,}")
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            " Xếp hạng", 
            " Chỉ số chất lượng", 
            " Chi tiết thuật toán",
            " So sánh trực quan"
        ])
        
        with tab1:
            st.markdown("####  Bảng Xếp Hạng Thuật Toán")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Ranking table with styling
                ranking_display = method_ranking.copy()
                ranking_display['Medal'] = ['1', '2', '3', '4'][:len(ranking_display)]
                ranking_display = ranking_display[['Medal', 'Rank', 'Method', 'Total_Score']]
                
                st.dataframe(
                    ranking_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Medal": st.column_config.TextColumn("", width="small"),
                        "Rank": st.column_config.NumberColumn("Hạng", width="small"),
                        "Method": st.column_config.TextColumn("Thuật toán", width="medium"),
                        "Total_Score": st.column_config.NumberColumn("Điểm tổng", width="small")
                    }
                )
                
                # Explanation
                with st.expander(" Cách tính điểm"):
                    st.markdown("""
                    **Hệ thống chấm điểm:**
                    - Mỗi chỉ số được xếp hạng từ 1-4
                    - Điểm = 5 - Hạng (1st=4đ, 2nd=3đ, 3rd=2đ, 4th=1đ)
                    - Tổng điểm = Tổng điểm của 3 chỉ số
                    
                    **Các chỉ số đánh giá:**
                    1. Silhouette Score (cao hơn = tốt hơn)
                    2. Davies-Bouldin Index (thấp hơn = tốt hơn)
                    3. Calinski-Harabasz Score (cao hơn = tốt hơn)
                    """)
            
            with col2:
                # Bar chart of scores
                fig = px.bar(
                    method_ranking,
                    x='Method',
                    y='Total_Score',
                    title='Điểm Tổng Các Thuật Toán',
                    labels={'Total_Score': 'Điểm tổng', 'Method': 'Thuật toán'},
                    color='Total_Score',
                    color_continuous_scale='Blues',
                    text='Total_Score'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Radar chart for top 2 methods
                st.markdown("#####  So Sánh Top 2 Thuật Toán")
                
                top_2_methods = method_ranking.head(2)['Method'].tolist()
                top_2_metrics = quality_metrics[quality_metrics['Method'].isin(top_2_methods)]
                
                fig = go.Figure()
                
                for _, row in top_2_metrics.iterrows():
                    # Normalize metrics to 0-1 scale for radar chart
                    silhouette_norm = row['Silhouette']
                    db_norm = 1 - (row['Davies-Bouldin'] / quality_metrics['Davies-Bouldin'].max())
                    ch_norm = row['Calinski-Harabasz'] / quality_metrics['Calinski-Harabasz'].max()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=[silhouette_norm, db_norm, ch_norm],
                        theta=['Silhouette', 'Davies-Bouldin<br>(inverted)', 'Calinski-Harabasz'],
                        fill='toself',
                        name=row['Method']
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("####  Chỉ Số Chất Lượng Chi Tiết")
            
            # Display metrics table
            metrics_display = quality_metrics.copy()
            metrics_display = metrics_display.round(4)
            
            st.dataframe(
                metrics_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Method": st.column_config.TextColumn("Thuật toán", width="medium"),
                    "Silhouette": st.column_config.NumberColumn("Silhouette ↑", format="%.4f"),
                    "Davies-Bouldin": st.column_config.NumberColumn("Davies-Bouldin ↓", format="%.4f"),
                    "Calinski-Harabasz": st.column_config.NumberColumn("Calinski-Harabasz ↑", format="%.2f"),
                    "N_Clusters": st.column_config.NumberColumn("Số Clusters", format="%d")
                }
            )
            
            st.markdown("---")
            
            # Individual metric comparisons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### Silhouette Score")
                fig = px.bar(
                    quality_metrics.sort_values('Silhouette', ascending=False),
                    x='Method',
                    y='Silhouette',
                    color='Silhouette',
                    color_continuous_scale='Greens',
                    text='Silhouette'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("↑ Cao hơn = Tốt hơn | Đo độ gắn kết cluster")
            
            with col2:
                st.markdown("##### Davies-Bouldin Index")
                fig = px.bar(
                    quality_metrics.sort_values('Davies-Bouldin'),
                    x='Method',
                    y='Davies-Bouldin',
                    color='Davies-Bouldin',
                    color_continuous_scale='Reds_r',
                    text='Davies-Bouldin'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("↓ Thấp hơn = Tốt hơn | Đo độ tách biệt cluster")
            
            with col3:
                st.markdown("##### Calinski-Harabasz Score")
                fig = px.bar(
                    quality_metrics.sort_values('Calinski-Harabasz', ascending=False),
                    x='Method',
                    y='Calinski-Harabasz',
                    color='Calinski-Harabasz',
                    color_continuous_scale='Blues',
                    text='Calinski-Harabasz'
                )
                fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("↑ Cao hơn = Tốt hơn | Đo tỷ lệ phân tán")
            
            # Metrics explanation
            with st.expander(" Giải Thích Các Chỉ Số"):
                st.markdown("""
                ### Silhouette Score (Hệ số Silhouette)
                - **Phạm vi:** -1 đến 1
                - **Ý nghĩa:** Đo độ tương đồng của một điểm với cluster của nó so với các cluster khác
                - **Giá trị tốt:** Gần 1 (cluster rõ ràng, tách biệt tốt)
                - **Giá trị xấu:** Gần 0 hoặc âm (cluster chồng chéo)
                
                ### Davies-Bouldin Index
                - **Phạm vi:** 0 đến ∞
                - **Ý nghĩa:** Đo tỷ lệ giữa khoảng cách trong cluster và giữa các cluster
                - **Giá trị tốt:** Gần 0 (cluster compact và tách biệt)
                - **Giá trị xấu:** Cao (cluster rải rác hoặc chồng chéo)
                
                ### Calinski-Harabasz Score (Variance Ratio Criterion)
                - **Phạm vi:** 0 đến ∞
                - **Ý nghĩa:** Tỷ lệ giữa phương sai giữa các cluster và trong cluster
                - **Giá trị tốt:** Cao (cluster rõ ràng và tách biệt)
                - **Giá trị xấu:** Thấp (cluster không rõ ràng)
                """)
        
        with tab3:
            st.markdown("####  Chi Tiết Từng Thuật Toán")
            
            # Method selector
            selected_method = st.selectbox(
                "Chọn thuật toán để xem chi tiết:",
                methods_overview['Method'].tolist()
            )
            
            method_overview = methods_overview[methods_overview['Method'] == selected_method].iloc[0]
            method_quality = quality_metrics[quality_metrics['Method'] == selected_method].iloc[0]
            method_rank = method_ranking[method_ranking['Method'] == selected_method].iloc[0]
            
            # Display method details
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### {selected_method}")
                st.markdown(f"**Xếp hạng:** #{method_rank['Rank']} / {len(method_ranking)}")
                st.markdown(f"**Điểm tổng:** {method_rank['Total_Score']}/12")
                
                st.markdown("---")
                
                st.markdown("####  Thông Tin Cơ Bản")
                st.metric("Số mẫu", f"{method_overview['Samples']:,}")
                st.metric("Số clusters", method_overview['Clusters'])
                st.metric("Noise points", f"{method_overview['Noise Points']:,}")
                st.metric("Tỷ lệ noise", method_overview['Noise %'])
                
                st.markdown("---")
                
                st.markdown("####  Chỉ Số Chất Lượng")
                st.metric("Silhouette", f"{method_quality['Silhouette']:.4f}")
                st.metric("Davies-Bouldin", f"{method_quality['Davies-Bouldin']:.4f}")
                st.metric("Calinski-Harabasz", f"{method_quality['Calinski-Harabasz']:.2f}")
            
            with col2:
                # Comparison with other methods
                st.markdown("####  So Sánh Với Các Thuật Toán Khác")
                
                # Create comparison dataframe
                comparison_data = []
                for metric in ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']:
                    selected_value = method_quality[metric]
                    avg_value = quality_metrics[metric].mean()
                    best_value = quality_metrics[metric].max() if metric != 'Davies-Bouldin' else quality_metrics[metric].min()
                    
                    comparison_data.append({
                        'Metric': metric,
                        'Giá trị': selected_value,
                        'Trung bình': avg_value,
                        'Tốt nhất': best_value
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Bar chart comparison
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Giá trị hiện tại',
                    x=comparison_df['Metric'],
                    y=comparison_df['Giá trị'],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Bar(
                    name='Trung bình',
                    x=comparison_df['Metric'],
                    y=comparison_df['Trung bình'],
                    marker_color='orange'
                ))
                
                fig.add_trace(go.Bar(
                    name='Tốt nhất',
                    x=comparison_df['Metric'],
                    y=comparison_df['Tốt nhất'],
                    marker_color='green'
                ))
                
                fig.update_layout(
                    barmode='group',
                    height=400,
                    title=f'So Sánh {selected_method} với Các Thuật Toán Khác'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Method characteristics
                st.markdown("####  Đặc Điểm Thuật Toán")
                
                method_info = {
                    'K-means': {
                        'icon': '',
                        'pros': ['Nhanh và hiệu quả', 'Dễ hiểu và implement', 'Hoạt động tốt với cluster hình cầu'],
                        'cons': ['Cần chỉ định số cluster trước', 'Nhạy cảm với outliers', 'Giả định cluster có kích thước tương đương'],
                        'use_case': 'Phù hợp khi biết trước số cluster và data có phân bố đều'
                    },
                    'GMM': {
                        'icon': '',
                        'pros': ['Mô hình xác suất', 'Cluster có thể chồng chéo', 'Linh hoạt với hình dạng cluster'],
                        'cons': ['Chậm hơn K-means', 'Cần nhiều data', 'Có thể overfit'],
                        'use_case': 'Phù hợp khi cluster có phân bố Gaussian và có thể chồng chéo'
                    },
                    'Hierarchical': {
                        'icon': '',
                        'pros': ['Không cần chỉ định số cluster', 'Tạo dendrogram trực quan', 'Deterministic'],
                        'cons': ['Chậm với data lớn (O(n³))', 'Nhạy cảm với noise', 'Không thể undo merge'],
                        'use_case': 'Phù hợp khi cần phân tích cấu trúc phân cấp của data'
                    },
                    'HDBSCAN': {
                        'icon': '',
                        'pros': ['Tự động tìm số cluster', 'Xử lý noise tốt', 'Tìm cluster với mật độ khác nhau'],
                        'cons': ['Nhiều hyperparameters', 'Khó tune', 'Có thể tạo nhiều noise points'],
                        'use_case': 'Phù hợp khi có outliers và cluster có mật độ khác nhau'
                    }
                }
                
                if selected_method in method_info:
                    info = method_info[selected_method]
                    
                    st.markdown(f"### {info['icon']} {selected_method}")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("** Ưu điểm:**")
                        for pro in info['pros']:
                            st.markdown(f"- {pro}")
                    
                    with col_b:
                        st.markdown("** Nhược điểm:**")
                        for con in info['cons']:
                            st.markdown(f"- {con}")
                    
                    st.info(f"** Khi nào dùng:** {info['use_case']}")
        
        with tab4:
            st.markdown("####  So Sánh Trực Quan")
            
            # Multi-metric comparison
            st.markdown("#####  So Sánh Đa Chỉ Số")
            
            # Normalize metrics for comparison
            metrics_normalized = quality_metrics.copy()
            metrics_normalized['Silhouette_norm'] = metrics_normalized['Silhouette']
            metrics_normalized['DB_norm'] = 1 - (metrics_normalized['Davies-Bouldin'] / metrics_normalized['Davies-Bouldin'].max())
            metrics_normalized['CH_norm'] = metrics_normalized['Calinski-Harabasz'] / metrics_normalized['Calinski-Harabasz'].max()
            
            # Parallel coordinates plot
            fig = go.Figure(data=
                go.Parcoords(
                    line=dict(
                        color=metrics_normalized.index,
                        colorscale='Viridis',
                        showscale=True
                    ),
                    dimensions=[
                        dict(label='Silhouette', values=metrics_normalized['Silhouette_norm']),
                        dict(label='Davies-Bouldin<br>(inverted)', values=metrics_normalized['DB_norm']),
                        dict(label='Calinski-Harabasz', values=metrics_normalized['CH_norm']),
                        dict(label='N_Clusters', values=metrics_normalized['N_Clusters'])
                    ],
                    labelfont=dict(size=12),
                    tickfont=dict(size=10)
                )
            )
            
            fig.update_layout(
                title='Parallel Coordinates: So Sánh Tất Cả Chỉ Số (Normalized)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add method labels
            method_labels = "  |  ".join([f"{i+1}. {m}" for i, m in enumerate(quality_metrics['Method'])])
            st.caption(f"**Thuật toán:** {method_labels}")
            
            st.markdown("---")
            
            # Cluster count vs metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#####  Số Clusters vs Silhouette Score")
                fig = px.scatter(
                    quality_metrics,
                    x='N_Clusters',
                    y='Silhouette',
                    size='Calinski-Harabasz',
                    color='Method',
                    text='Method',
                    size_max=30
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#####  Noise Points Distribution")
                fig = px.bar(
                    methods_overview,
                    x='Method',
                    y='Noise Points',
                    color='Noise Points',
                    color_continuous_scale='Reds',
                    text='Noise Points'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of all metrics
            st.markdown("#####  Heatmap: Tất Cả Chỉ Số")
            
            heatmap_data = quality_metrics.set_index('Method')[['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']]
            
            # Normalize for better visualization
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            heatmap_normalized = pd.DataFrame(
                scaler.fit_transform(heatmap_data),
                index=heatmap_data.index,
                columns=heatmap_data.columns
            )
            
            fig = px.imshow(
                heatmap_normalized.T,
                labels=dict(x="Thuật toán", y="Chỉ số", color="Giá trị (normalized)"),
                x=heatmap_normalized.index,
                y=heatmap_normalized.columns,
                color_continuous_scale='RdYlGn',
                aspect='auto',
                text_auto='.2f'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Summary and recommendations
        st.markdown("###  Tổng Kết & Khuyến Nghị")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            ** Thuật toán được khuyến nghị: {method_ranking.iloc[0]['Method']}**
            
            Dựa trên tổng hợp các chỉ số đánh giá, {method_ranking.iloc[0]['Method']} cho kết quả tốt nhất với:
            - Điểm tổng: {method_ranking.iloc[0]['Total_Score']}/12
            - Cân bằng tốt giữa các chỉ số chất lượng
            - Phù hợp với đặc điểm của dataset Carbon-24
            """)
        
        with col2:
            st.info("""
            ** Lưu ý khi chọn thuật toán:**
            
            - **K-means/Hierarchical**: Tốt cho cluster rõ ràng, không có noise
            - **HDBSCAN**: Tốt khi có outliers và cluster mật độ khác nhau
            - **GMM**: Tốt khi cần mô hình xác suất và cluster chồng chéo
            
            Nên xem xét cả yêu cầu cụ thể của bài toán và tài nguyên tính toán.
            """)

# ============================================================================
# PAGE: PHÁT HIỆN DỊ BIỆT
# ============================================================================
elif page == " Phát hiện dị biệt":
    render_anomaly_detection_tab()

# ============================================================================
# PAGE: PIPELINE 3 TẦNG
# ============================================================================
elif page == " Pipeline 3 Tầng":
    render_pipeline_tab()

# ============================================================================
# PAGE: GROUND-TRUTH LABELING
# ============================================================================
elif page == " Ground-Truth Labeling":
    render_ground_truth_tab()

# ============================================================================
# PAGE: DỰ ĐOÁN NĂNG LƯỢNG
# ============================================================================
elif page == " Dự đoán năng lượng":
    render_energy_prediction_tab()
    
    st.markdown("""
    ### Mô hình sẽ áp dụng:
    
    1. **Linear Regression**
       - Baseline model
       - Dễ interpret
    
    2. **Random Forest Regressor**
       - Ensemble method
       - Feature importance
    
    3. **Gradient Boosting**
       - XGBoost / LightGBM
       - High performance
    
    4. **Neural Network**
       - Deep learning approach
       - Complex patterns
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p> Carbon-24 Data Mining Dashboard | Developed with Streamlit</p>
</div>
""", unsafe_allow_html=True)




# ============================================================================
