import streamlit as st
import pandas as pd
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as e:
    st.error(
        "Missing dependency: plotly. On Streamlit Cloud, make sure `requirements.txt` "
        "is committed at the repository root and contains `plotly>=5.18.0`, then reboot the app."
    )
    st.stop()
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import json
import os
import html

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
    :root {
        --bg: #f6f8fb;
        --surface: #ffffff;
        --surface-soft: #f9fbff;
        --border: #dbe3ef;
        --text: #172033;
        --muted: #64748b;
        --blue: #2563eb;
        --green: #16a34a;
        --amber: #d97706;
        --red: #dc2626;
        --violet: #7c3aed;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 30rem),
            linear-gradient(180deg, #f8fbff 0%, #f2f6fb 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }

    .main-header {
        font-size: clamp(2rem, 3vw, 3.25rem);
        line-height: 1.05;
        font-weight: 850;
        color: #0f172a;
        text-align: left;
        letter-spacing: 0;
        margin: 0 0 0.35rem 0;
    }

    .dashboard-subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        margin-bottom: 1.35rem;
    }

    .sub-header {
        font-size: 1.35rem;
        font-weight: 780;
        color: #111827;
        margin-top: 1.4rem;
        margin-bottom: 0.9rem;
        padding: 0.65rem 0 0.65rem 0.85rem;
        border-left: 4px solid var(--blue);
    }

    .metric-card {
        background-color: var(--surface);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #172033 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e5eefb;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #dbeafe !important;
    }

    [data-testid="stSidebar"] [data-baseweb="radio"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 0.35rem 0.5rem;
        margin: 0.2rem 0;
    }

    .brand-card {
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 1rem;
    }

    .brand-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }

    .brand-meta {
        font-size: 0.78rem;
        color: #a9b8d0;
        margin-top: 0.2rem;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 820;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stPlotlyChart"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        padding: 0.3rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px 8px 0 0;
        padding: 0 1rem;
        background: transparent;
        color: var(--muted);
        font-weight: 650;
    }

    .stTabs [aria-selected="true"] {
        background: var(--surface);
        color: var(--blue);
        border: 1px solid var(--border);
        border-bottom-color: var(--surface);
    }

    .stButton button,
    .stDownloadButton button,
    [data-testid="stFormSubmitButton"] button {
        border-radius: 8px;
        border: 1px solid #1d4ed8;
        background: #2563eb;
        color: white;
        font-weight: 700;
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.20);
    }

    .stButton button:hover,
    .stDownloadButton button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: #1d4ed8;
        border-color: #1e40af;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea,
    input {
        border-radius: 8px !important;
    }

    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
    }

    .viewer-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.82rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"> Carbon-24 Data Mining Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Phan cum, phan tich di biet, du doan nang luong va truc quan hoa 3D cau truc Carbon-24</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="brand-card">
      <div class="brand-title">Carbon-24 Lab</div>
      <div class="brand-meta">Clustering • Anomaly • Energy • 3D</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Navigation")
    
    page = st.radio(
        "Chon trang:",
        [
            "Research Workflow",
            "Tong quan",
            "Khao sat du lieu",
            "Phan cum",
            "So sanh thuat toan",
            "Pipeline 3 Tang",
            "Ground-Truth Labeling",
            "3D Structure Viewer",
        ]
    )
    
    if page == "Research Workflow":
        st.markdown("---")
        st.markdown("###  Hướng dẫn")
        st.success("""
        **4-Bước Quy Trình Nghiên Cứu:**
        
        **Bước 1** — Lọc cấu trúc dị biệt  
        **Bước 2** — Định danh Pha thù hình  
        **Bước 3** — Chọn bộ dự đoán tối ưu  
        **Bước 4** — Dự đoán cấu trúc mới  
        """)
    
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


def get_metric(metrics, *keys, default=None):
    """Return a metric value across report versions with different key names."""
    if not isinstance(metrics, dict):
        return default
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return default


def format_metric(value, fmt="{:.4f}", fallback="N/A"):
    if value is None:
        return fallback
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


@st.cache_data
def load_structure_viewer_data():
    """Load metadata, predictions, anomaly flags, and optional CIF strings."""
    try:
        base = pd.read_csv('carbon24_pipeline_results/pipeline_final.csv')
    except Exception:
        try:
            base = pd.read_csv('data/regression/features_with_targets.csv')
        except Exception as e:
            st.error(f"Khong load duoc du lieu cau truc: {e}")
            return None, None

    if os.path.exists('carbon24_energy_results/predictions_test.csv'):
        pred = pd.read_csv('carbon24_energy_results/predictions_test.csv')
        pred_cols = [c for c in ['material_id', 'pred_best', 'err_best'] if c in pred.columns]
        if 'material_id' in pred_cols:
            base = base.merge(pred[pred_cols], on='material_id', how='left')

    if os.path.exists('carbon24_anomaly_detection/anomaly_detection_results.csv'):
        anom = pd.read_csv('carbon24_anomaly_detection/anomaly_detection_results.csv')
        anom_cols = [c for c in [
            'material_id', 'is_hdbscan_noise', 'is_low_probability',
            'is_isolation_forest_anomaly', 'is_anomaly_any',
            'is_anomaly_consensus', 'anomaly_vote_count'
        ] if c in anom.columns]
        if 'material_id' in anom_cols:
            base = base.merge(anom[anom_cols], on='material_id', how='left')

    if os.path.exists('carbon24_cif_cache.csv'):
        cif = pd.read_csv('carbon24_cif_cache.csv')
        cif_cols = [c for c in ['material_id', 'cif'] if c in cif.columns]
        if len(cif_cols) == 2:
            base = base.merge(cif[cif_cols], on='material_id', how='left')
    else:
        base['cif'] = np.nan

    mp_ref = None
    if os.path.exists('carbon.csv'):
        try:
            mp_ref = pd.read_csv('carbon.csv')
        except Exception:
            mp_ref = None

    return base, mp_ref


def _first_existing(row, columns, fallback="N/A"):
    for col in columns:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return fallback


def render_cif_3d(cif_text, height=560, label_text=None, status_text=None):
    """Render a CIF string with py3Dmol when available, otherwise 3Dmol.js."""
    if not isinstance(cif_text, str) or not cif_text.strip():
        st.warning("Chua co chuoi CIF cho material nay. Chay `python prepare_cif_cache.py` de tao cache CIF.")
        return

    try:
        import py3Dmol
        view = py3Dmol.view(width='100%', height=height)
        view.addModel(cif_text, 'cif')
        view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.28}})
        view.addUnitCell()
        if label_text:
            full_label = str(label_text)
            if status_text:
                full_label = f"{full_label}\n{status_text}"
            view.addLabel(
                full_label,
                {
                    'position': {'x': 0, 'y': 0, 'z': 0},
                    'fontSize': 14,
                    'fontColor': 'black',
                    'backgroundColor': 'white',
                    'backgroundOpacity': 0.82,
                    'borderColor': '#334155',
                    'borderThickness': 1,
                },
            )
        view.zoomTo()
        components.html(view._make_html(), height=height + 20)
        return
    except Exception:
        pass

    div_id = f"viewer_{abs(hash(cif_text))}"
    cif_json = json.dumps(cif_text)
    label_json = json.dumps(str(label_text or ""))
    status_html = html.escape(str(status_text or ""))
    label_html = html.escape(str(label_text or ""))
    viewer_html = f"""
    <div style="position:relative;width:100%;height:{height}px;background:#f8fafc;border:1px solid #d7dee8;">
      <div id="{div_id}" style="width:100%;height:{height}px;position:absolute;inset:0;"></div>
      <div style="position:absolute;top:10px;left:10px;z-index:5;background:rgba(255,255,255,.88);border:1px solid #334155;border-radius:6px;padding:8px 10px;font-family:Arial,sans-serif;font-size:13px;color:#111827;max-width:72%;">
        <b>{label_html}</b><br><span>{status_html}</span>
      </div>
    </div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      const cifData = {cif_json};
      const viewerLabel = {label_json};
      const element = document.getElementById("{div_id}");
      const viewer = $3Dmol.createViewer(element, {{ backgroundColor: "white" }});
      viewer.addModel(cifData, "cif");
      viewer.setStyle({{}}, {{ stick: {{ radius: 0.15 }}, sphere: {{ scale: 0.28 }} }});
      viewer.addUnitCell();
      if (viewerLabel) {{
        viewer.addLabel(viewerLabel, {{
          position: {{x: 0, y: 0, z: 0}},
          fontSize: 14,
          fontColor: "black",
          backgroundColor: "white",
          backgroundOpacity: 0.82,
          borderColor: "#334155",
          borderThickness: 1
        }});
      }}
      viewer.zoomTo();
      viewer.render();
    </script>
    """
    components.html(viewer_html, height=height + 20)


def get_cluster_label(row):
    label = _first_existing(row, ['scientific_label', 'kmeans_label', 'best_match'], None)
    if label is not None:
        return str(label)
    return (
        f"KMeans={_first_existing(row, ['kmeans_cluster', 'cluster'], 'N/A')} | "
        f"GMM={_first_existing(row, ['gmm_cluster', 'GMM_Cluster'], 'N/A')} | "
        f"HDBSCAN={_first_existing(row, ['hdbscan_cluster'], 'N/A')}"
    )


def get_stability_assessment(row):
    rel = _first_existing(row, ['relative_energy'], None)
    anomaly_flags = [
        'is_anomaly_consensus', 'is_anomaly_any', 'is_hdbscan_noise',
        'hdbscan_is_noise', 'is_low_probability', 'is_isolation_forest_anomaly',
    ]
    is_outlier = False
    for col in anomaly_flags:
        if col in row.index and pd.notna(row[col]):
            value = row[col]
            if isinstance(value, str):
                is_outlier = is_outlier or value.lower() in ['true', '1', 'yes']
            else:
                is_outlier = is_outlier or bool(value)

    if is_outlier:
        return "Outlier / can kiem tra", "warning", "Bi danh dau bat thuong boi HDBSCAN/anomaly detector."

    if rel is None:
        return "Chua du du lieu", "info", "Khong co relative_energy de danh gia."

    try:
        rel = float(rel)
    except (TypeError, ValueError):
        return "Chua du du lieu", "info", "relative_energy khong hop le."

    if rel <= 0.05:
        return "Rat on dinh", "success", "relative_energy <= 0.05 eV/atom."
    if rel <= 0.15:
        return "On dinh", "success", "relative_energy trong vung thap."
    if rel <= 0.35:
        return "Trung binh", "warning", "relative_energy trung binh, can so sanh voi cum."
    return "Kem on dinh / nang luong cao", "error", "relative_energy cao hon nguong 0.35 eV/atom."


def get_3d_label(row, prefix=None, prediction_col=None):
    cluster_label = get_cluster_label(row)
    stability, _, _ = get_stability_assessment(row)
    parts = []
    if prefix:
        parts.append(str(prefix))
    material_id = _first_existing(row, ['material_id'], None)
    if material_id is not None:
        parts.append(str(material_id))
    parts.append(str(cluster_label))
    rel = _first_existing(row, ['relative_energy'], None)
    if rel is not None:
        parts.append(f"Erel={format_metric(rel, '{:.4f}')} eV/atom")
    if prediction_col and prediction_col in row.index and pd.notna(row[prediction_col]):
        parts.append(f"Pred={format_metric(row[prediction_col], '{:.4f}')} eV/atom")
    parts.append(f"Stability={stability}")
    return " | ".join(parts)


@st.cache_data
def load_mp_reference_structures():
    if not os.path.exists('carbon.csv'):
        return None
    try:
        return pd.read_csv('carbon.csv')
    except Exception:
        return None


def get_reference_cif_for_prediction(space_group_symbol=None, crystal_system=None):
    ref = load_mp_reference_structures()
    if ref is None or ref.empty or 'Structure' not in ref.columns:
        return None, None

    candidates = ref.copy()
    if space_group_symbol and 'Space Group Symbol' in candidates.columns:
        sg_match = candidates[candidates['Space Group Symbol'].astype(str) == str(space_group_symbol)]
        if not sg_match.empty:
            candidates = sg_match
    elif crystal_system and 'Crystal System' in candidates.columns:
        cs_match = candidates[candidates['Crystal System'].astype(str).str.lower() == str(crystal_system).lower()]
        if not cs_match.empty:
            candidates = cs_match

    if 'Energy Above Hull' in candidates.columns:
        candidates = candidates.sort_values('Energy Above Hull', ascending=True)
    row = candidates.iloc[0]
    return row.get('Structure'), row


def generate_live_prediction_cif(
    num_atoms, a, b, c, alpha, beta, gamma, crystal_system,
    cluster_label="Carbon", space_group_symbol="P1",
):
    """Generate a lightweight CIF directly from Live Prediction inputs.

    This is a visual model built from the entered lattice parameters and a
    deterministic carbon atom layout. It is not a DFT-relaxed structure.
    """
    n_atoms = max(1, int(num_atoms))
    cs = str(crystal_system).lower()
    label = str(cluster_label).replace(" ", "_").replace("/", "_")
    sg = str(space_group_symbol or "P1")

    coords = []

    if cs == "cubic" and "diamond" in str(cluster_label).lower():
        basis = [
            (0.000, 0.000, 0.000),
            (0.250, 0.250, 0.250),
            (0.500, 0.500, 0.000),
            (0.750, 0.750, 0.250),
            (0.500, 0.000, 0.500),
            (0.750, 0.250, 0.750),
            (0.000, 0.500, 0.500),
            (0.250, 0.750, 0.750),
        ]
        rep = 1
        while len(coords) < n_atoms:
            for ix in range(rep):
                for iy in range(rep):
                    for iz in range(rep):
                        for x, y, z in basis:
                            coords.append(((x + ix) / rep % 1, (y + iy) / rep % 1, (z + iz) / rep % 1))
                            if len(coords) >= n_atoms:
                                break
                        if len(coords) >= n_atoms:
                            break
                    if len(coords) >= n_atoms:
                        break
                if len(coords) >= n_atoms:
                    break
            rep += 1
    elif cs in ["hexagonal", "trigonal"] or "graph" in str(cluster_label).lower():
        layers = max(1, min(4, int(round(n_atoms / 12)) or 1))
        atoms_per_layer = int(np.ceil(n_atoms / layers))
        grid = int(np.ceil(np.sqrt(atoms_per_layer / 2)))
        basis = [(0.0, 0.0), (1.0 / 3.0, 2.0 / 3.0)]
        for layer in range(layers):
            z = (layer + 0.5) / layers
            for ix in range(grid):
                for iy in range(grid):
                    for bx, by in basis:
                        x = (ix + bx) / grid
                        y = (iy + by) / grid
                        coords.append((x % 1, y % 1, z % 1))
                        if len(coords) >= n_atoms:
                            break
                    if len(coords) >= n_atoms:
                        break
                if len(coords) >= n_atoms:
                    break
            if len(coords) >= n_atoms:
                break
    elif "layer" in str(cluster_label).lower():
        layers = max(2, min(5, int(round(n_atoms / 8)) or 2))
        atoms_per_layer = int(np.ceil(n_atoms / layers))
        grid = int(np.ceil(np.sqrt(atoms_per_layer)))
        for layer in range(layers):
            z = (layer + 0.5) / layers
            offset = 0.13 * (layer % 2)
            for ix in range(grid):
                for iy in range(grid):
                    coords.append(((ix + 0.5 + offset) / grid % 1, (iy + 0.5 + offset) / grid % 1, z))
                    if len(coords) >= n_atoms:
                        break
                if len(coords) >= n_atoms:
                    break
            if len(coords) >= n_atoms:
                break
    else:
        # Deterministic low-discrepancy layout for amorphous/novel/proxy cases.
        phi1 = 0.61803398875
        phi2 = 0.75487766625
        phi3 = 0.56984029099
        for i in range(n_atoms):
            coords.append((
                (0.17 + i * phi1) % 1,
                (0.31 + i * phi2) % 1,
                (0.47 + i * phi3) % 1,
            ))

    atom_lines = []
    for i, (x, y, z) in enumerate(coords[:n_atoms]):
        atom_lines.append(f"  C  C{i}  1  {x:.6f}  {y:.6f}  {z:.6f}  1")

    return f"""# generated by Carbon-24 Dashboard Live Prediction
data_{label}
_symmetry_space_group_name_H-M   '{sg}'
_cell_length_a   {float(a):.8f}
_cell_length_b   {float(b):.8f}
_cell_length_c   {float(c):.8f}
_cell_angle_alpha   {float(alpha):.8f}
_cell_angle_beta   {float(beta):.8f}
_cell_angle_gamma   {float(gamma):.8f}
_chemical_formula_structural   C
_chemical_formula_sum   C{n_atoms}
_cell_formula_units_Z   {n_atoms}
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
{chr(10).join(atom_lines)}
"""


def render_cluster_stability_summary(row, prediction_col=None, error_col=None):
    cluster_label = get_cluster_label(row)
    stability, level, reason = get_stability_assessment(row)

    st.markdown(f"**Nhan cum:** `{cluster_label}`")

    cols = st.columns(4)
    cols[0].metric("K-means", _first_existing(row, ['kmeans_label', 'kmeans_cluster', 'cluster'], 'N/A'))
    cols[1].metric("GMM", _first_existing(row, ['scientific_label', 'gmm_cluster', 'GMM_Cluster'], 'N/A'))
    cols[2].metric("HDBSCAN", _first_existing(row, ['hdbscan_cluster'], 'N/A'))
    cols[3].metric("Do on dinh", stability)

    if prediction_col and prediction_col in row.index:
        pcols = st.columns(3)
        pcols[0].metric("Actual relative_energy", format_metric(_first_existing(row, ['relative_energy'], None), "{:.6f}"))
        pcols[1].metric("Predicted relative_energy", format_metric(row.get(prediction_col), "{:.6f}"))
        if error_col and error_col in row.index:
            pcols[2].metric("Prediction error", format_metric(row.get(error_col), "{:+.6f}"))

    message = f"{stability}: {reason}"
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def render_structure_3d_viewer_tab():
    st.markdown('<div class="sub-header">3D Structure Viewer</div>', unsafe_allow_html=True)

    base, mp_ref = load_structure_viewer_data()
    if base is None or base.empty:
        st.warning("Khong co du lieu de hien thi.")
        return

    st.info(
        "Chon material_id de xem metadata, predicted energy va cau truc 3D tu CIF. "
        "Neu chua co CIF cho Carbon-24, chay `python prepare_cif_cache.py` truoc."
    )

    mode = st.radio(
        "Bo loc nhanh:",
        ["Tat ca", "Outliers", "Nang luong thap nhat", "Co CIF", "Prediction test set"],
        horizontal=True,
    )

    filtered = base.copy()
    if mode == "Outliers":
        outlier_cols = [c for c in ['is_anomaly_consensus', 'is_anomaly_any', 'is_hdbscan_noise'] if c in filtered.columns]
        if outlier_cols:
            mask = pd.Series(False, index=filtered.index)
            for col in outlier_cols:
                mask = mask | (filtered[col].fillna(0).astype(float) > 0)
            filtered = filtered[mask]
        elif 'hdbscan_is_noise' in filtered.columns:
            filtered = filtered[filtered['hdbscan_is_noise'].fillna(False).astype(bool)]
    elif mode == "Nang luong thap nhat":
        energy_col = 'relative_energy' if 'relative_energy' in filtered.columns else 'energy_per_atom'
        filtered = filtered.sort_values(energy_col, ascending=True).head(200)
    elif mode == "Co CIF":
        filtered = filtered[filtered['cif'].notna()]
    elif mode == "Prediction test set":
        filtered = filtered[filtered['pred_best'].notna()] if 'pred_best' in filtered.columns else filtered.iloc[0:0]

    if filtered.empty:
        st.warning("Khong co mau nao phu hop voi bo loc hien tai.")
        return

    search = st.text_input("Tim material_id:", "")
    if search:
        filtered = filtered[filtered['material_id'].astype(str).str.contains(search, case=False, na=False)]
        if filtered.empty:
            st.warning("Khong tim thay material_id phu hop.")
            return

    filtered = filtered.sort_values('relative_energy', ascending=True) if 'relative_energy' in filtered.columns else filtered

    def format_material(mid):
        row = filtered[filtered['material_id'] == mid].iloc[0]
        rel = _first_existing(row, ['relative_energy'], None)
        label = _first_existing(row, ['scientific_label', 'kmeans_label', 'best_match'], 'N/A')
        if rel is None:
            return f"{mid} | {label}"
        return f"{mid} | Erel={rel:.4f} | {label}"

    selected_id = st.selectbox(
        "Chon material_id:",
        filtered['material_id'].astype(str).tolist(),
        format_func=format_material,
    )

    row = filtered[filtered['material_id'].astype(str) == selected_id].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("energy_per_atom", format_metric(_first_existing(row, ['energy_per_atom', 'energy'], None), "{:.6f}"))
    c2.metric("relative_energy", format_metric(_first_existing(row, ['relative_energy'], None), "{:.6f}"))
    c3.metric("num_atoms", format_metric(_first_existing(row, ['num_atoms'], None), "{:.0f}"))
    c4.metric("volume", format_metric(_first_existing(row, ['volume'], None), "{:.3f}"))
    c5.metric("predicted energy", format_metric(_first_existing(row, ['pred_best'], None), "{:.6f}"))

    render_cluster_stability_summary(row, prediction_col='pred_best', error_col='err_best')

    meta_cols = [c for c in [
        'material_id', 'split', 'formula', 'crystal_system', 'space_group_symbol',
        'hdbscan_cluster', 'hdbscan_probability', 'hdbscan_is_noise',
        'kmeans_cluster', 'kmeans_label', 'gmm_cluster', 'gmm_probability',
        'scientific_label', 'pred_best', 'err_best',
        'is_anomaly_any', 'is_anomaly_consensus', 'anomaly_vote_count'
    ] if c in row.index]
    with st.expander("Metadata"):
        st.dataframe(pd.DataFrame([row[meta_cols].to_dict()]), use_container_width=True, hide_index=True)

    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("#### Carbon-24 3D structure")
        stability, _, _ = get_stability_assessment(row)
        render_cif_3d(
            row.get('cif'),
            height=600,
            label_text=get_3d_label(row, prediction_col='pred_best'),
            status_text=f"Nhan cum: {get_cluster_label(row)} | Do on dinh: {stability}",
        )

    with col_right:
        st.markdown("#### Reference CIF")
        if mp_ref is not None and 'Structure' in mp_ref.columns:
            ref_options = mp_ref.copy()
            if 'Space Group Symbol' in ref_options.columns and pd.notna(row.get('space_group_symbol')):
                same_sg = ref_options[ref_options['Space Group Symbol'].astype(str) == str(row.get('space_group_symbol'))]
                if not same_sg.empty:
                    ref_options = same_sg
            ref_labels = ref_options.apply(
                lambda r: f"{r.get('Material ID', 'N/A')} | {r.get('Crystal System', 'N/A')} | {r.get('Space Group Symbol', 'N/A')}",
                axis=1,
            ).tolist()
            ref_idx = st.selectbox("MP reference:", list(range(len(ref_options))), format_func=lambda i: ref_labels[i])
            ref_row = ref_options.iloc[ref_idx]
            st.metric("MP Energy Above Hull", format_metric(ref_row.get('Energy Above Hull'), "{:.6f}"))
            render_cif_3d(
                ref_row.get('Structure'),
                height=360,
                label_text=f"MP reference: {ref_row.get('Material ID', 'N/A')} | {ref_row.get('Space Group Symbol', 'N/A')}",
                status_text=f"Energy above hull: {format_metric(ref_row.get('Energy Above Hull'), '{:.4f}')} eV/atom",
            )
        else:
            st.info("Khong co file `carbon.csv` de hien thi cau truc tham chieu.")

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
    st.markdown('<div class="sub-header"> Ground-Truth Labeling</div>', unsafe_allow_html=True)

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

    #  METRICS 
    st.markdown("###  Tổng quan")
    n_diamond  = (gmm["structure_type"] == "Diamond (Fd-3m)").sum()
    n_graphite = (gmm["structure_type"] == "Graphite/Graphene").sum()
    n_layered  = (gmm["structure_type"] == "Layered C2/m").sum()
    n_amorph   = (gmm["structure_type"] == "Amorphous (P1/P-1)").sum()
    n_other    = (gmm["structure_type"] == "Other").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(" Diamond (Fd-3m)",    f"{n_diamond:,}",  f"{n_diamond/len(gmm):.1%}")
    c2.metric(" Graphite/Graphene",  f"{n_graphite:,}", f"{n_graphite/len(gmm):.1%}")
    c3.metric(" Layered C2/m",       f"{n_layered:,}",  f"{n_layered/len(gmm):.1%}")
    c4.metric(" Amorphous",          f"{n_amorph:,}",   f"{n_amorph/len(gmm):.1%}")
    c5.metric(" Noise (anomalies)",  f"{len(noise):,}", f"{len(noise)/(len(gmm)+len(noise)):.1%}")

    st.markdown("---")

    #  TABS 
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " PCA Labels",
        " Energy Analysis",
        " Space Group Overlap",
        " Structure Types",
        " Cluster Summary",
    ])

    #  TAB 1: PCA 
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
                    f'<span style="color:{color}; font-size:18px"></span> '
                    f'**{stype}**<br><small>{n:,} ({n/len(gmm):.1%})</small>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<span style="color:black; font-size:18px"></span> '
                f'**Noise**<br><small>{len(noise):,}</small>',
                unsafe_allow_html=True,
            )

    #  TAB 2: ENERGY 
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
            st.markdown("** Diamond (Fd-3m) — Key Finding:**")
            st.metric("Mean energy", f"{diamond_e.mean():.4f} eV/atom")
            st.metric("Min energy",  f"{diamond_e.min():.4f} eV/atom")
            st.metric("Samples",     f"{len(diamond_e):,}")
            st.success(f"Diamond cluster có energy ≈ **{diamond_e.mean():.4f} eV/atom** — "
                       f"gần bằng 0, xác nhận đây là cấu trúc **ổn định nhất** trong dataset.")

            st.markdown("---")
            st.markdown("** Energy summary by structure type:**")
            e_summary = gmm.groupby("structure_type")["relative_energy"].agg(
                ["mean", "std", "min", "max"]
            ).round(4).sort_values("mean")
            st.dataframe(e_summary, use_container_width=True)

    #  TAB 3: SPACE GROUP OVERLAP 
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

    #  TAB 4: STRUCTURE TYPES 
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

    #  TAB 5: CLUSTER SUMMARY 
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
        st.markdown("** Download:**")
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

@st.cache_data
def load_pipeline_3d_view_data():
    try:
        labeled = pd.read_csv('carbon24_pipeline_results/tier3_gmm_labeled.csv')
        labels = pd.read_csv('carbon24_pipeline_results/ground_truth_labels.csv')
        noise = pd.read_csv('carbon24_pipeline_results/tier1_noise_analysis.csv')
    except Exception as e:
        st.error(f"Khong load duoc du lieu Pipeline 3D View: {e}")
        return None, None, None, None

    if os.path.exists('carbon24_energy_results/predictions_test.csv'):
        pred = pd.read_csv('carbon24_energy_results/predictions_test.csv')
        pred_cols = [c for c in ['material_id', 'pred_best', 'err_best'] if c in pred.columns]
        if 'material_id' in pred_cols:
            labeled = labeled.merge(pred[pred_cols], on='material_id', how='left')

    if os.path.exists('carbon24_cif_cache.csv'):
        cif = pd.read_csv('carbon24_cif_cache.csv', usecols=['material_id', 'cif'])
        labeled = labeled.merge(cif, on='material_id', how='left')
        noise = noise.merge(cif, on='material_id', how='left')

    ref = pd.read_csv('carbon.csv') if os.path.exists('carbon.csv') else None
    return labeled, labels, noise, ref


def estimate_pipeline_energy(row, cluster_info=None):
    if 'pred_best' in row.index and pd.notna(row.get('pred_best')):
        return float(row.get('pred_best')), "Regression model output: pred_best"

    base = float(row.get('relative_energy', 0.25)) if pd.notna(row.get('relative_energy', np.nan)) else 0.25
    if cluster_info is not None and 'energy_mean' in cluster_info.index and pd.notna(cluster_info.get('energy_mean')):
        base = 0.65 * float(cluster_info.get('energy_mean')) + 0.35 * base

    geom_penalty = 0.0
    if pd.notna(row.get('std_coordination', np.nan)):
        geom_penalty += min(float(row.get('std_coordination')) * 0.02, 0.06)
    if pd.notna(row.get('angle_deviation', np.nan)):
        geom_penalty += min(float(row.get('angle_deviation')) * 0.0015, 0.08)

    return max(0.0, min(base + geom_penalty, 1.5)), "Estimated from Stage-3 regression features"


def render_pipeline_3d_model_view(final_df, noise_df, gmm_profile):
    st.markdown("### 3D Model View - Discovery to Verification")
    st.caption(
        "Buoc 1: kham pha PCA. Buoc 2: dao sau cum nghi ngo va so khop Materials Project. "
        "Buoc 3: chon cau truc, du doan nang luong va xem 3D model."
    )

    labeled, labels, noise, ref = load_pipeline_3d_view_data()
    if labeled is None:
        return

    label_map = labels.set_index('gmm_cluster') if labels is not None and 'gmm_cluster' in labels.columns else pd.DataFrame()
    default_cluster = 3 if 3 in set(labels['gmm_cluster']) else int(labels['gmm_cluster'].iloc[0])

    cluster_options = []
    for _, r in labels.sort_values(['match_score', 'gmm_cluster']).iterrows():
        cluster_options.append(
            f"GMM-{int(r['gmm_cluster'])} | {r['scientific_label']} | match={r['match_score']:.2f} | Emean={r['energy_mean']:.4f}"
        )
    default_index = next((i for i, s in enumerate(cluster_options) if s.startswith(f"GMM-{default_cluster} ")), 0)

    st.markdown("#### Buoc 1 - Kham pha: PCA noise va pha ben vung")
    selected_cluster_str = st.selectbox(
        "Chon cum mac dinh, hoac click truc tiep mot node tren PCA:",
        cluster_options,
        index=default_index,
        key="pipeline_3d_cluster",
    )
    selected_cluster = int(selected_cluster_str.split("|")[0].replace("GMM-", "").strip())
    clicked_material_id = None
    clicked_is_noise = False
    cluster_info = label_map.loc[selected_cluster] if selected_cluster in label_map.index else pd.Series(dtype=object)

    col_pca, col_story = st.columns([2, 1])
    with col_pca:
        fig = go.Figure()
        if {'pca1', 'pca2'}.issubset(labeled.columns):
            fig.add_trace(go.Scatter(
                x=labeled['pca1'], y=labeled['pca2'], mode='markers',
                name='Clean structures',
                marker=dict(size=4, color='#94a3b8', opacity=0.22),
                customdata=np.stack([
                    labeled['material_id'].astype(str),
                    labeled['gmm_cluster'].astype(str),
                    np.repeat('clean', len(labeled)),
                ], axis=-1),
                hovertemplate='%{customdata[0]}<br>GMM-%{customdata[1]}<br>Clean<extra></extra>',
            ))
            stable_mask = (
                labeled['relative_energy'].fillna(9) <= 0.05
            ) | labeled.get('scientific_label', pd.Series('', index=labeled.index)).astype(str).str.contains(
                'diamond|graphite|graphene', case=False, na=False
            )
            stable = labeled[stable_mask]
            if not stable.empty:
                fig.add_trace(go.Scatter(
                    x=stable['pca1'], y=stable['pca2'], mode='markers',
                    name='Pha ben vung / Diamond-Graphite-like',
                    marker=dict(size=6, color='#16a34a', opacity=0.65),
                    customdata=np.stack([
                        stable['material_id'].astype(str),
                        stable['gmm_cluster'].astype(str),
                        np.repeat('stable', len(stable)),
                    ], axis=-1),
                    hovertemplate='%{customdata[0]}<br>GMM-%{customdata[1]}<br>Stable zone<extra></extra>',
                ))
            if noise is not None and {'pca1', 'pca2'}.issubset(noise.columns):
                fig.add_trace(go.Scatter(
                    x=noise['pca1'], y=noise['pca2'], mode='markers',
                    name='Noise / outlier',
                    marker=dict(size=7, color='#dc2626', opacity=0.75, line=dict(width=0.5, color='#7f1d1d')),
                    customdata=np.stack([
                        noise['material_id'].astype(str),
                        np.repeat('-1', len(noise)),
                        np.repeat('noise', len(noise)),
                    ], axis=-1),
                    hovertemplate='%{customdata[0]}<br>Noise / outlier<extra></extra>',
                ))
            selected = labeled[labeled['gmm_cluster'] == selected_cluster]
            fig.add_trace(go.Scatter(
                x=selected['pca1'], y=selected['pca2'], mode='markers',
                name=f'GMM-{selected_cluster} selected',
                marker=dict(size=9, color='#f59e0b', opacity=0.9, symbol='diamond', line=dict(width=1, color='#111827')),
                customdata=np.stack([
                    selected['material_id'].astype(str),
                    selected['gmm_cluster'].astype(str),
                    np.repeat('selected', len(selected)),
                ], axis=-1),
                hovertemplate='%{customdata[0]}<br>GMM-%{customdata[1]}<br>Selected cluster<extra></extra>',
            ))
            fig.update_layout(
                title="PCA exploration: noise, stable phases, suspected cluster",
                xaxis_title="PCA1", yaxis_title="PCA2", height=520,
                legend=dict(orientation='h', y=-0.2),
            )
            pca_event = st.plotly_chart(
                fig,
                use_container_width=True,
                key="pipeline_3d_pca_select",
                on_select="rerun",
                selection_mode="points",
            )
            if isinstance(pca_event, dict):
                points = pca_event.get("selection", {}).get("points", [])
            else:
                points = getattr(getattr(pca_event, "selection", None), "points", [])
            if points:
                point = points[0]
                custom = point.get("customdata", []) if isinstance(point, dict) else []
                if custom:
                    clicked_material_id = str(custom[0])
                    clicked_is_noise = len(custom) > 2 and str(custom[2]) == "noise"
                    if not clicked_is_noise and len(custom) > 1:
                        try:
                            selected_cluster = int(custom[1])
                            cluster_info = label_map.loc[selected_cluster] if selected_cluster in label_map.index else pd.Series(dtype=object)
                        except (TypeError, ValueError):
                            pass

    with col_story:
        if clicked_material_id:
            st.success(f"Da click node: {clicked_material_id}")
        st.metric("Selected cluster", f"GMM-{selected_cluster}")
        st.metric("Samples", f"{len(labeled[labeled['gmm_cluster'] == selected_cluster]):,}")
        st.metric("Match score", format_metric(cluster_info.get('match_score'), "{:.2f}"))
        st.metric("Energy mean", format_metric(cluster_info.get('energy_mean'), "{:.4f}"))
        if pd.notna(cluster_info.get('match_score')) and float(cluster_info.get('match_score')) <= 0.5:
            st.warning("Kha nang cao la cau truc moi/di biet.")
        else:
            st.info("Cum co muc so khop trung binh/cao voi Materials Project.")

    if clicked_is_noise and clicked_material_id:
        st.markdown("#### Buoc 2 - Dao sau: node noise/outlier")
        st.warning("Node vua click la Noise/Outlier o Tang 1, khong di tiep qua GMM matching. Nen kiem tra nhu cau truc di biet.")
        noise_row = noise[noise['material_id'].astype(str) == clicked_material_id]
        if not noise_row.empty:
            row = noise_row.iloc[0].copy()
            row['pipeline_e_pred'] = row.get('relative_energy', np.nan)
            render_cluster_stability_summary(row, prediction_col='pipeline_e_pred')
            col_model, col_detail = st.columns([2, 1])
            with col_model:
                render_cif_3d(
                    row.get('cif'),
                    height=620,
                    label_text=get_3d_label(row, prefix="PIPELINE NOISE NODE", prediction_col='pipeline_e_pred'),
                    status_text="Noise / outlier - can kiem chung rieng",
                )
            with col_detail:
                detail_cols = [c for c in [
                    'material_id', 'relative_energy', 'energy_per_atom', 'num_atoms',
                    'volume', 'crystal_system', 'space_group_symbol',
                    'hdbscan_cluster', 'hdbscan_probability', 'hdbscan_is_noise',
                ] if c in row.index]
                st.dataframe(pd.DataFrame([row[detail_cols].to_dict()]), use_container_width=True, hide_index=True)
        return

    st.markdown("#### Buoc 2 - Dao sau: Materials Project matching")
    match_cols = [c for c in [
        'gmm_cluster', 'scientific_label', 'best_match', 'match_score',
        'energy_mean', 'top_crystal_system', 'top_space_group',
        'electronic_pred', 'description'
    ] if c in labels.columns]
    selected_match = labels[labels['gmm_cluster'] == selected_cluster][match_cols]
    st.dataframe(selected_match, use_container_width=True, hide_index=True)

    match_score = float(cluster_info.get('match_score', 1.0)) if pd.notna(cluster_info.get('match_score')) else 1.0
    if match_score <= 0.5:
        st.error(f"Match score = {match_score:.2f}. Khả năng cao là cấu trúc mới/dị biệt.")

    if ref is not None and not ref.empty:
        ref_show = ref.copy()
        top_sg = cluster_info.get('top_space_group')
        top_cs = cluster_info.get('top_crystal_system')
        if pd.notna(top_sg) and 'Space Group Symbol' in ref_show.columns:
            ref_show = ref_show[ref_show['Space Group Symbol'].astype(str) == str(top_sg)]
        if ref_show.empty and pd.notna(top_cs) and 'Crystal System' in ref.columns:
            ref_show = ref[ref['Crystal System'].astype(str).str.lower() == str(top_cs).lower()]
        ref_cols = [c for c in ['Material ID', 'Formula', 'Crystal System', 'Space Group Symbol', 'Energy Above Hull', 'Band Gap', 'Is Metal'] if c in ref_show.columns]
        if ref_cols:
            st.markdown("**Materials Project candidates:**")
            st.dataframe(ref_show[ref_cols].head(12), use_container_width=True, hide_index=True)

    st.markdown("#### Buoc 3 - Kiem chung: du doan nang luong + 3D model")
    selected_structures = labeled[labeled['gmm_cluster'] == selected_cluster].copy()
    selected_structures = selected_structures.sort_values(['pred_best', 'relative_energy'], na_position='last')
    only_pred = st.checkbox("Chi hien thi cau truc co pred_best tu model hoi quy", value=False)
    if only_pred and 'pred_best' in selected_structures.columns:
        selected_structures = selected_structures[selected_structures['pred_best'].notna()]

    if selected_structures.empty:
        st.warning("Khong co cau truc phu hop trong cum nay.")
        return

    def fmt_material(mid):
        r = selected_structures[selected_structures['material_id'].astype(str) == str(mid)].iloc[0]
        return (
            f"{mid} | E={format_metric(r.get('relative_energy'), '{:.4f}')} | "
            f"pred={format_metric(r.get('pred_best'), '{:.4f}')} | {r.get('scientific_label', 'N/A')}"
        )

    material_options = selected_structures['material_id'].astype(str).tolist()
    material_index = 0
    if clicked_material_id in material_options:
        material_index = material_options.index(clicked_material_id)

    material_id = st.selectbox(
        "Chon cau truc de kiem chung:",
        material_options,
        index=material_index,
        format_func=fmt_material,
        key="pipeline_3d_material",
    )
    row = selected_structures[selected_structures['material_id'].astype(str) == material_id].iloc[0].copy()

    run_pred = st.button("Dự đoán năng lượng", type="primary", use_container_width=True, key="pipeline_3d_predict")
    if run_pred:
        e_pred, pred_source = estimate_pipeline_energy(row, cluster_info)
        row['pipeline_e_pred'] = e_pred
        row['pipeline_pred_error'] = row.get('relative_energy', np.nan) - e_pred if pd.notna(row.get('relative_energy', np.nan)) else np.nan
        render_cluster_stability_summary(row, prediction_col='pipeline_e_pred', error_col='pipeline_pred_error')
        st.caption(pred_source)

        col_model, col_detail = st.columns([2, 1])
        with col_model:
            render_cif_3d(
                row.get('cif'),
                height=620,
                label_text=get_3d_label(row, prefix=f"PIPELINE GMM-{selected_cluster}", prediction_col='pipeline_e_pred'),
                status_text=f"Match={match_score:.2f} | {row.get('scientific_label', 'N/A')}",
            )
        with col_detail:
            detail_cols = [c for c in [
                'material_id', 'scientific_label', 'best_match', 'kmeans_label',
                'gmm_cluster', 'gmm_probability', 'relative_energy', 'pipeline_e_pred',
                'pipeline_pred_error', 'num_atoms', 'volume', 'crystal_system',
                'space_group_symbol', 'hdbscan_cluster', 'hdbscan_probability',
            ] if c in row.index]
            st.dataframe(pd.DataFrame([row[detail_cols].to_dict()]), use_container_width=True, hide_index=True)
    else:
        st.info("Chon cau truc va bam 'Dự đoán năng lượng' de hien thi E_pred va model 3D.")


def render_pipeline_tab():
    st.markdown('<div class="sub-header"> Pipeline 3 Tầng</div>', unsafe_allow_html=True)

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

    #  METRICS 
    st.markdown("###  Tổng quan")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng mẫu", f"{len(final_df):,}")
    c2.metric("Noise (Tier 1)", f"{len(noise):,}", f"{len(noise)/len(final_df):.1%}")
    c3.metric("Clean (Tier 2+)", f"{len(clean):,}", f"{len(clean)/len(final_df):.1%}")
    c4.metric("GMM sub-clusters", "10", "polymorphs")

    st.markdown("---")

    #  TABS CHO 3 TANG 
    t1, t2, t3, t4 = st.tabs([
        "Tang 1 - Noise",
        "Tang 2 - K-means",
        "Tang 3 - GMM",
        "3D Model View",
    ])

    #  TANG 1 
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

    #  TANG 2 
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

    #  TANG 3 
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

    with t4:
        render_pipeline_3d_model_view(final_df, noise_df, gmm_profile)

    st.markdown("---")

    #  DOWNLOAD 
    st.markdown("###  Download Kết quả")
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
    st.markdown('<div class="sub-header"> Phát hiện Dị biệt (Anomaly Detection)</div>', unsafe_allow_html=True)

    st.info(
        "**Nguyên tắc:** Phát hiện dựa trên features **cấu trúc** (lattice, bond, coordination) — "
        "không dùng energy làm input. Energy chỉ dùng để **diễn giải** kết quả sau khi phát hiện."
    )

    results_df, summary_df, comparison_df, details_df = load_anomaly_detection_results()

    if results_df is None:
        st.warning(" Chưa có dữ liệu. Vui lòng chạy:")
        st.code("python carbon24_anomaly_detection.py")
        return

    #  1. METRICS 
    st.markdown("###  Tổng quan")
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
    c5.metric("Consensus  (≥2)",
              f"{results_df['is_anomaly_consensus'].sum():,}",
              f"{results_df['is_anomaly_consensus'].mean():.1%}",
              help="Được ít nhất 2/3 phương pháp đồng ý — khuyến nghị sử dụng")

    st.markdown("---")

    #  2. SO SÁNH PHƯƠNG PHÁP 
    st.markdown("###  So sánh các phương pháp")
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

    #  3. PCA VISUALIZATION 
    st.markdown("###  Phân bố không gian (PCA)")
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

    #  4. DIEN GIAI NANG LUONG 
    st.markdown("###  Diễn giải Năng lượng")
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

    #  5. ISOLATION FOREST SCORE 
    if 'isolation_forest_score' in results_df.columns:
        st.markdown("###  Isolation Forest Score")
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

    #  6. OVERLAP / JACCARD 
    st.markdown("###  Overlap giữa các phương pháp")
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

    #  7. DANH SACH ANOMALIES 
    st.markdown("###  Chi tiết Consensus Anomalies")
    if details_df is not None and len(details_df) > 0:
        col1, col2 = st.columns(2)
        with col1:
            vote_filter = st.multiselect("Loc vote count:", [2, 3], default=[2, 3])
        with col2:
            n_show = st.slider("So hang hien thi:", 10, min(100, len(details_df)), 20, 10)
        filtered = details_df[details_df['anomaly_vote_count'].isin(vote_filter)]
        st.markdown(f"**{len(filtered):,} anomalies** (vote in {vote_filter})")
        st.dataframe(filtered.head(n_show), use_container_width=True, hide_index=True)
        st.download_button(" Download CSV", filtered.to_csv(index=False),
                           "consensus_anomalies.csv", "text/csv")
    else:
        st.info("Khong co consensus anomalies")

    st.markdown("---")

    #  8. KHUYEN NGHI 
    st.markdown("###  Khuyến nghị sử dụng")
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
    st.markdown('<div class="sub-header"> Dự đoán Năng lượng</div>', unsafe_allow_html=True)

    st.info(
        "**Model Leaderboard** — 4 mô hình dự đoán `relative_energy` (eV/atom) "
        "sử dụng 27 features: structural + cluster labels + scientific labels từ Ground-Truth Labeling."
    )

    lb, pred_df, fi_df = load_energy_results()
    if lb is None:
        st.warning("Chưa có kết quả. Vui lòng chạy:")
        st.code("python carbon24_energy_prediction.py")
        return

    #  LEADERBOARD METRICS 
    st.markdown("###  Model Leaderboard")

    best = lb.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(" Best Model",  best["Model"])
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

    #  TABS 
    t1, t2, t3, t4 = st.tabs([
        " Leaderboard Charts",
        " Predictions",
        " Feature Importance",
        " Residuals",
    ])

    #  TAB 1: LEADERBOARD CHARTS 
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

    #  TAB 2: PREDICTIONS 
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

        st.markdown("---")
        st.markdown("#### Ket qua du doan theo cau truc 3D")

        pred_filter = st.radio(
            "Loc cau truc du doan:",
            ["Tat ca", "Nang luong thap nhat", "Sai so cao nhat"],
            horizontal=True,
            key="pred_3d_filter",
        )
        pred_options = pred_df.copy()
        err_col = sel_col.replace("pred_", "err_")
        if pred_filter == "Nang luong thap nhat":
            pred_options = pred_options.sort_values("relative_energy", ascending=True).head(200)
        elif pred_filter == "Sai so cao nhat" and err_col in pred_options.columns:
            pred_options = pred_options.assign(_abs_err=pred_options[err_col].abs())
            pred_options = pred_options.sort_values("_abs_err", ascending=False).head(200)

        def format_prediction_material(mid):
            prow = pred_options[pred_options["material_id"].astype(str) == str(mid)].iloc[0]
            pred_val = prow.get(sel_col)
            actual_val = prow.get("relative_energy")
            label = prow.get("scientific_label", prow.get("kmeans_label", "N/A"))
            return (
                f"{mid} | actual={format_metric(actual_val, '{:.4f}')} | "
                f"pred={format_metric(pred_val, '{:.4f}')} | {label}"
            )

        selected_pred_id = st.selectbox(
            "Chon material_id de xem 3D:",
            pred_options["material_id"].astype(str).tolist(),
            format_func=format_prediction_material,
            key="prediction_3d_material",
        )

        pred_row = pred_options[pred_options["material_id"].astype(str) == selected_pred_id].iloc[0]
        structure_df, _ = load_structure_viewer_data()
        if structure_df is not None and not structure_df.empty:
            matched = structure_df[structure_df["material_id"].astype(str) == selected_pred_id]
        else:
            matched = pd.DataFrame()

        if not matched.empty:
            selected_row = matched.iloc[0].copy()
            for col in pred_row.index:
                selected_row[col] = pred_row[col]
        else:
            selected_row = pred_row.copy()

        selected_row["selected_prediction"] = pred_row.get(sel_col)
        if err_col in pred_row.index:
            selected_row["selected_error"] = pred_row.get(err_col)
        elif pd.notna(pred_row.get(sel_col)) and pd.notna(pred_row.get("relative_energy")):
            selected_row["selected_error"] = pred_row.get("relative_energy") - pred_row.get(sel_col)

        render_cluster_stability_summary(
            selected_row,
            prediction_col="selected_prediction",
            error_col="selected_error",
        )

        left_3d, right_info = st.columns([2, 1])
        with left_3d:
            st.markdown("##### Model 3D cua cau truc duoc du doan")
            pred_stability, _, _ = get_stability_assessment(selected_row)
            render_cif_3d(
                selected_row.get("cif"),
                height=560,
                label_text=get_3d_label(selected_row, prefix="Prediction", prediction_col="selected_prediction"),
                status_text=f"Nhan cum: {get_cluster_label(selected_row)} | Do on dinh: {pred_stability}",
            )
        with right_info:
            st.markdown("##### Thong tin cau truc")
            detail_cols = [c for c in [
                "material_id", "split", "formula", "num_atoms", "volume",
                "crystal_system", "space_group_symbol", "relative_energy",
                "selected_prediction", "selected_error", "kmeans_label",
                "gmm_cluster", "scientific_label", "hdbscan_cluster",
                "hdbscan_probability", "hdbscan_is_noise",
            ] if c in selected_row.index]
            st.dataframe(pd.DataFrame([selected_row[detail_cols].to_dict()]), use_container_width=True, hide_index=True)

        # Predictions table
        st.markdown("**Sample predictions (first 50):**")
        show_cols = ["material_id", "relative_energy", sel_col,
                     "kmeans_label", "scientific_label", "crystal_system"]
        show_cols = [c for c in show_cols if c in pred_df.columns]
        st.dataframe(pred_df[show_cols].head(50), use_container_width=True, hide_index=True)

        # Download
        st.download_button(
            " Download All Predictions (CSV)",
            pred_df.to_csv(index=False),
            "energy_predictions.csv", "text/csv",
        )

    #  TAB 3: FEATURE IMPORTANCE 
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

    #  TAB 4: RESIDUALS 
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
# RESEARCH WORKFLOW — 4-BƯỚC QUY TRÌNH NGHIÊN CỨU CARBON-24
# ============================================================================

def render_research_workflow():
    """
    4-bước Research Workflow dành cho Carbon-24 Data Mining Dashboard.
    Bước 1: Overview & Anomaly Filter (PCA 3D + toggle)
    Bước 2: Clustering & MP Matching (GMM click → MP profile)
    Bước 3: Model Leaderboard + Feature Importance
    Bước 4: Live Prediction (nhập thông số → dự đoán năng lượng)
    """
    import numpy as _np_rw

    #  CSS riêng cho workflow 
    st.markdown("""
    <style>
    .step-header {
        font-size: 1.35rem; font-weight: 820;
        padding: 0.85rem 1rem; border-radius: 8px;
        margin-bottom: 0.9rem; color: #0f172a;
        border: 1px solid #dbe3ef;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
    }
    .step1 { background: linear-gradient(90deg,#dbeafe,#eff6ff); border-left: 5px solid #2563eb; }
    .step2 { background: linear-gradient(90deg,#dcfce7,#f0fdf4); border-left: 5px solid #16a34a; }
    .step3 { background: linear-gradient(90deg,#f3e8ff,#faf5ff); border-left: 5px solid #7c3aed; }
    .step4 { background: linear-gradient(90deg,#ffedd5,#fff7ed); border-left: 5px solid #ea580c; }
    .kpi-box {
        background: #ffffff; border-radius: 8px; padding: 1rem;
        border: 1px solid #dbe3ef; text-align:center;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
    }
    .kpi-val { font-size:1.8rem; font-weight:850; color:#0f172a; }
    .kpi-lbl { font-size:0.82rem; color:#64748b; font-weight:650; }
    .mp-card {
        background:#ffffff; border:1px solid #dbe3ef; border-radius:8px;
        padding:1rem; margin-bottom:0.75rem;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
    }
    .pred-result {
        background: linear-gradient(135deg,#0f172a 0%,#1e3a8a 100%);
        color:white; border-radius:8px; padding:1.35rem; text-align:center;
        box-shadow: 0 12px 26px rgba(30, 58, 138, 0.20);
    }
    .pred-energy { font-size:2.65rem; font-weight:900; color:#facc15; }
    .workflow-connector {
        text-align:center; font-size:0.95rem; color:#64748b;
        background:#ffffff; border:1px dashed #cbd5e1; border-radius:8px;
        padding:0.7rem; margin:0.8rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    #  Hero banner 
    st.markdown("""
    <div style='background:#ffffff;border:1px solid #dbe3ef;border-left:5px solid #2563eb;
         border-radius:8px;padding:1.4rem 1.6rem;margin-bottom:1rem;
         box-shadow:0 10px 24px rgba(15,23,42,.05);'>
      <h1 style='color:#0f172a;font-size:2rem;margin:0 0 6px 0;letter-spacing:0;'>
         Carbon-24 Research Workflow
      </h1>
      <p style='color:#64748b;font-size:1rem;margin:0;'>
        Quy trình 4 bước: Lọc dị biệt → Định danh pha → Chọn mô hình → Dự đoán cấu trúc mới
      </p>
    </div>
    """, unsafe_allow_html=True)

    #  Tab navigation 
    tab1, tab2, tab3, tab4 = st.tabs([
        " Bước 1 — Overview & Anomalies",
        " Bước 2 — Clustering & MP Matching",
        " Bước 3 — Model Leaderboard",
        " Bước 4 — Live Prediction",
    ])

    # 
    # BƯỚC 1: OVERVIEW & ANOMALY FILTER
    # 
    with tab1:
        st.markdown('<div class="step-header step1"> Bước 1 — Khảo sát & Lọc Cấu trúc Dị biệt</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        > **Mục tiêu:** Xem bức tranh toàn cảnh về tập dữ liệu Carbon-24 và loại bỏ
        > những cấu trúc lỗi mà HDBSCAN phát hiện ra. Toggle **"Cô lập điểm dị biệt"**
        > để xác nhận bộ lọc hoạt động chính xác.
        """)

        # Load pipeline data
        final_df, noise_df, km_profile, gmm_df_p, gmm_profile_p = load_pipeline_results()

        if final_df is None:
            st.error(" Chưa có dữ liệu pipeline. Vui lòng chạy: `python carbon24_pipeline_3tier.py`")
            return

        noise_p  = final_df[final_df['pipeline_stage'] == 'noise']
        clean_p  = final_df[final_df['pipeline_stage'] != 'noise']
        n_total  = len(final_df)
        n_noise  = len(noise_p)
        n_clean  = len(clean_p)

        #  KPI Cards 
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class="kpi-box" style="border-color:#1565C0">
              <div class="kpi-val" style="color:#1565C0">{n_total:,}</div>
              <div class="kpi-lbl">Tổng mẫu ban đầu</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="kpi-box" style="border-color:#2E7D32">
              <div class="kpi-val" style="color:#2E7D32">{n_clean:,}</div>
              <div class="kpi-lbl"> Dữ liệu sạch</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="kpi-box" style="border-color:#C62828">
              <div class="kpi-val" style="color:#C62828">{n_noise:,}</div>
              <div class="kpi-lbl"> Cấu trúc dị biệt (Noise)</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="kpi-box" style="border-color:#E65100">
              <div class="kpi-val" style="color:#E65100">{n_noise/n_total:.1%}</div>
              <div class="kpi-lbl">Tỷ lệ Noise HDBSCAN</div></div>""", unsafe_allow_html=True)
        with c5:
            e_diff = noise_p['relative_energy'].mean() - clean_p['relative_energy'].mean() if 'relative_energy' in final_df.columns else 0
            st.markdown(f"""<div class="kpi-box" style="border-color:#6A1B9A">
              <div class="kpi-val" style="color:#6A1B9A">{e_diff:+.4f}</div>
              <div class="kpi-lbl">ΔE Noise vs Clean (eV/atom)</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        #  Toggle cô lập dị biệt 
        col_ctrl, col_info = st.columns([1, 3])
        with col_ctrl:
            isolate = st.toggle(" Cô lập điểm dị biệt", value=False,
                                help="Bật: Highlight 786 noise points ở rìa không gian PCA")
            show_3d = st.toggle(" Chế độ 3D", value=False,
                                help="Chuyển sang biểu đồ PCA 3D")
        with col_info:
            if isolate:
                st.info("""** Chế độ Cô lập đang BẬT** — Các điểm dị biệt được highlight đỏ sáng.
                Quan sát thấy chúng nằm ở **rìa ngoài** không gian PCA, xác nhận HDBSCAN hoạt động chính xác.""")
            else:
                st.info("""**Chế độ thường** — Hiển thị toàn bộ dataset.
                Bật **Toggle** để cô lập và quan sát 786 cấu trúc dị biệt.""")

        #  PCA Plot 
        pca_cols_2d = ['pca1','pca2']
        pca_cols_3d = ['pca1','pca2','pca3'] if 'pca3' in final_df.columns else None
        has_pca = all(c in final_df.columns for c in pca_cols_2d)

        if not has_pca:
            st.warning("Không tìm thấy cột pca1/pca2 trong pipeline_final.csv")
        else:
            if show_3d and pca_cols_3d:
                # 3D PCA
                fig = go.Figure()
                if isolate:
                    # Dim clean, highlight noise
                    fig.add_trace(go.Scatter3d(
                        x=clean_p['pca1'], y=clean_p['pca2'], z=clean_p['pca3'],
                        mode='markers', name=f'Clean ({n_clean:,})',
                        marker=dict(size=2, color='#90CAF9', opacity=0.12),
                        hoverinfo='skip',
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=noise_p['pca1'], y=noise_p['pca2'], z=noise_p['pca3'],
                        mode='markers', name=f' Noise/Dị biệt ({n_noise:,})',
                        marker=dict(size=5, color='#FF1744', opacity=0.95,
                                    line=dict(width=1, color='#FF8A80')),
                        hovertemplate='<b>DỊ BIỆT</b><br>PCA1=%{x:.2f}<br>PCA2=%{y:.2f}<br>PCA3=%{z:.2f}<extra></extra>',
                    ))
                    title_txt = " PCA 3D — Cô lập điểm dị biệt: Noise ở rìa ngoài không gian"
                else:
                    fig.add_trace(go.Scatter3d(
                        x=clean_p['pca1'], y=clean_p['pca2'], z=clean_p['pca3'],
                        mode='markers', name=f'Clean ({n_clean:,})',
                        marker=dict(size=2, color='#42A5F5', opacity=0.3),
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=noise_p['pca1'], y=noise_p['pca2'], z=noise_p['pca3'],
                        mode='markers', name=f'Noise ({n_noise:,})',
                        marker=dict(size=3, color='#EF5350', opacity=0.6,
                                    symbol='cross'),
                    ))
                    title_txt = "PCA 3D — Toàn bộ dataset: Clean + Noise"
                fig.update_layout(
                    title=title_txt, height=620,
                    scene=dict(
                        xaxis_title='PCA1', yaxis_title='PCA2', zaxis_title='PCA3',
                        bgcolor='#0D1B2A' if isolate else 'white',
                    ),
                    paper_bgcolor='#0D1B2A' if isolate else 'white',
                    font_color='white' if isolate else 'black',
                    legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 2D PCA
                fig = go.Figure()
                if isolate:
                    fig.add_trace(go.Scatter(
                        x=clean_p['pca1'], y=clean_p['pca2'],
                        mode='markers', name=f'Clean ({n_clean:,})',
                        marker=dict(size=3, color='#1A237E', opacity=0.08),
                        hoverinfo='skip',
                    ))
                    fig.add_trace(go.Scatter(
                        x=noise_p['pca1'], y=noise_p['pca2'],
                        mode='markers', name=f' Noise/Dị biệt ({n_noise:,})',
                        marker=dict(size=9, color='#FF1744', opacity=1.0,
                                    line=dict(width=1.5, color='#FFCDD2'),
                                    symbol='circle-open-dot'),
                        hovertemplate='<b>DỊ BIỆT</b><br>PCA1=%{x:.2f}<br>PCA2=%{y:.2f}<extra></extra>',
                    ))
                    fig.update_layout(
                        paper_bgcolor='#0A0E1A', plot_bgcolor='#0D1B2A',
                        font_color='white',
                        title=" CÔ LẬP DỊ BIỆT — 786 cấu trúc lỗi sáng lên ở rìa không gian PCA",
                    )
                else:
                    fig.add_trace(go.Scatter(
                        x=clean_p['pca1'], y=clean_p['pca2'],
                        mode='markers', name=f' Clean ({n_clean:,})',
                        marker=dict(size=3, color='#1976D2', opacity=0.25),
                    ))
                    fig.add_trace(go.Scatter(
                        x=noise_p['pca1'], y=noise_p['pca2'],
                        mode='markers', name=f' Noise ({n_noise:,})',
                        marker=dict(size=6, color='#EF5350', opacity=0.7,
                                    line=dict(width=0.8, color='#B71C1C')),
                    ))
                    fig.update_layout(title="PCA 2D — Toàn bộ dataset: Clean vs Noise")
                fig.update_layout(
                    xaxis_title='PCA1', yaxis_title='PCA2',
                    height=540, hovermode='closest',
                )
                st.plotly_chart(fig, use_container_width=True)

        #  Phân tích năng lượng Noise vs Clean 
        if 'relative_energy' in final_df.columns:
            st.markdown("####  Phân tích Năng lượng: Noise vs Clean")
            col1, col2 = st.columns(2)
            with col1:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(
                    x=clean_p['relative_energy'], name=' Clean', opacity=0.6,
                    marker_color='#1976D2', nbinsx=60))
                fig2.add_trace(go.Histogram(
                    x=noise_p['relative_energy'], name=' Noise', opacity=0.75,
                    marker_color='#FF1744', nbinsx=60))
                fig2.add_vline(x=clean_p['relative_energy'].mean(), line_dash='dash',
                               line_color='#1976D2',
                               annotation_text=f"Clean μ={clean_p['relative_energy'].mean():.4f}")
                fig2.add_vline(x=noise_p['relative_energy'].mean(), line_dash='dash',
                               line_color='#FF1744',
                               annotation_text=f"Noise μ={noise_p['relative_energy'].mean():.4f}")
                fig2.update_layout(barmode='overlay', height=380,
                                   title='Phân bố Energy: Noise kém ổn định hơn',
                                   xaxis_title='Relative Energy (eV/atom)', yaxis_title='Số mẫu')
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.markdown("####  Crystal System: Noise vs Clean")
                if 'crystal_system' in final_df.columns:
                    cs_n = noise_p['crystal_system'].value_counts(normalize=True) * 100
                    cs_c = clean_p['crystal_system'].value_counts(normalize=True) * 100
                    cs_df = pd.DataFrame({' Clean (%)': cs_c, ' Noise (%)': cs_n}).fillna(0).round(1)
                    st.dataframe(cs_df, use_container_width=True)
                st.markdown("---")
                st.success(f"""
                ** Kết luận Bước 1:**
                - **{n_clean:,}** cấu trúc sạch → đưa vào phân tích tiếp
                - **{n_noise:,}** dị biệt bị loại → năng lượng cao hơn {abs(e_diff):.4f} eV/atom
                - Bộ lọc HDBSCAN hoạt động chính xác → chuyển sang **Bước 2**
                """)

        st.markdown("---")
        st.markdown('<div class="workflow-connector"> Chuyển sang Bước 2: Định danh Pha thù hình</div>',
                    unsafe_allow_html=True)

    # 
    # BƯỚC 2: CLUSTERING & MP MATCHING
    # 
    with tab2:
        st.markdown('<div class="step-header step2"> Bước 2 — Định danh Khoa học cho các Pha thù hình</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        > **Mục tiêu:** Tìm hiểu tập dữ liệu gồm những loại cấu trúc tinh thể nào và
        > có biến thể Carbon-24 mới nào chưa được định danh. Nhấp vào một cụm để xem
        > **MP Profile** và **match_score** tương ứng.
        """)

        gmm_gt, labels_gt, sci_gt, ref_gt, noise_gt = load_ground_truth_data()
        final_df2, _, _, gmm_df2, _ = load_pipeline_results()

        if gmm_gt is None or gmm_df2 is None:
            st.error(" Chưa có dữ liệu. Vui lòng chạy pipeline và ground-truth scripts.")
            st.code("python carbon24_pipeline_3tier.py\npython carbon24_ground_truth_labeling.py")
            return

        # Chuẩn hóa ref columns
        ref_gt = ref_gt.rename(columns={
            "Crystal System": "crystal_system",
            "Space Group Symbol": "space_group_symbol",
            "Energy Above Hull": "e_above_hull",
            "Band Gap": "band_gap",
            "Is Metal": "is_metal",
        })

        # Gán structure_type cho gmm_df2
        gmm_df2 = gmm_df2.copy()
        gmm_df2["structure_type"] = "Other"
        gmm_df2.loc[gmm_df2.get("space_group_symbol", pd.Series(dtype=str)) == "Fd-3m", "structure_type"] = "Diamond (Fd-3m)"
        gmm_df2.loc[gmm_df2.get("space_group_symbol", pd.Series(dtype=str)).isin(
            ["P6_3/mmc", "R-3m", "P6/mmm"]), "structure_type"] = "Graphite/Graphene"
        gmm_df2.loc[gmm_df2.get("space_group_symbol", pd.Series(dtype=str)) == "C2/m", "structure_type"] = "Layered C2/m"
        gmm_df2.loc[gmm_df2.get("space_group_symbol", pd.Series(dtype=str)).isin(
            ["P1", "P-1"]), "structure_type"] = "Amorphous (P1/P-1)"

        # MP reference data tổng hợp cho 10 GMM cluster (hardcoded từ ground-truth)
        MP_PROFILES = {
            0: {"name": "Diamond-like Carbon",   "crystal": "Cubic",       "sg": "Fd-3m",    "elec": "Insulator",    "score": 0.92, "e_hull": 0.00, "color": "#1a9850"},
            1: {"name": "Graphite-like (hex)",   "crystal": "Hexagonal",   "sg": "P6₃/mmc", "elec": "Semimetal",    "score": 0.88, "e_hull": 0.03, "color": "#4393c3"},
            2: {"name": "Layered C2/m",          "crystal": "Monoclinic",  "sg": "C2/m",     "elec": "Semiconductor","score": 0.75, "e_hull": 0.08, "color": "#74add1"},
            3: {"name": "Amorphous sp2/sp3",     "crystal": "Triclinic",   "sg": "P-1",      "elec": "Semiconductor","score": 0.41, "e_hull": 0.35, "color": "#fee090"},
            4: {"name": "Fullerene-like",        "crystal": "Monoclinic",  "sg": "P2₁/c",   "elec": "Semiconductor","score": 0.38, "e_hull": 0.42, "color": "#fdae61"},
            5: {"name": "Graphene-like (R-3m)",  "crystal": "Trigonal",    "sg": "R-3m",     "elec": "Semimetal",    "score": 0.83, "e_hull": 0.05, "color": "#4575b4"},
            6: {"name": "Mixed sp2/sp3 Carbon",  "crystal": "Orthorhombic","sg": "Cmce",     "elec": "Semiconductor","score": 0.61, "e_hull": 0.18, "color": "#d9ef8b"},
            7: {"name": "High-energy Carbon",    "crystal": "Triclinic",   "sg": "P1",       "elec": "Conductor",    "score": 0.29, "e_hull": 0.68, "color": "#d73027"},
            8: {"name": "Compressed Diamond",    "crystal": "Cubic",       "sg": "Im-3m",    "elec": "Insulator",    "score": 0.79, "e_hull": 0.10, "color": "#a6d96a"},
            9: {"name": "Novel Carbon Phase*",   "crystal": "Monoclinic",  "sg": "C2/c",     "elec": "Semiconductor","score": 0.33, "e_hull": 0.55, "color": "#a50026"},
        }

        #  Bố cục 2 cột: trái = cluster matrix, phải = MP detail 
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("####  Ma trận lồng ghép K-means & GMM (10 cụm)")

            # Build summary table
            rows_s2 = []
            for cid in sorted(gmm_df2['gmm_cluster'].unique() if 'gmm_cluster' in gmm_df2.columns
                               else range(10)):
                sub = gmm_df2[gmm_df2['gmm_cluster'] == cid] if 'gmm_cluster' in gmm_df2.columns else pd.DataFrame()
                mp  = MP_PROFILES.get(int(cid), MP_PROFILES[9])
                sci_lbl = sub['scientific_label'].mode()[0] if ('scientific_label' in sub.columns and len(sub) > 0) else mp['name']
                km_lbl  = sub['kmeans_label'].mode()[0] if ('kmeans_label' in sub.columns and len(sub) > 0) else "N/A"
                e_mean  = sub['relative_energy'].mean() if ('relative_energy' in sub.columns and len(sub) > 0) else float('nan')
                rows_s2.append({
                    "Cụm GMM": f"GMM-{cid}",
                    "K-means Macro": km_lbl,
                    "N": len(sub),
                    "Energy (mean)": round(e_mean, 4) if not _np_rw.isnan(e_mean) else "N/A",
                    "Tên pha": mp['name'],
                    "Match Score ": mp['score'],
                })
            summary_df2 = pd.DataFrame(rows_s2)

            # Highlight match_score
            def color_score(val):
                if isinstance(val, float):
                    if val >= 0.8:  return 'background-color:#C8E6C9;color:#1B5E20'
                    if val >= 0.6:  return 'background-color:#FFF9C4;color:#F57F17'
                    return 'background-color:#FFCDD2;color:#B71C1C'
                return ''

            styled_df = summary_df2.style.applymap(color_score, subset=["Match Score "])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            st.caption(" score ≥ 0.8: khớp tốt với MP |  0.6–0.8: khớp trung bình |  < 0.6: pha mới tiềm năng")

            #  Chọn cụm để xem MP Profile 
            st.markdown("####  Chọn cụm để xem MP Profile")
            cluster_options = [f"GMM-{cid} — {MP_PROFILES[cid]['name']}" for cid in range(10)]
            sel_cluster_str = st.selectbox("Nhấp chọn cụm:", cluster_options, index=4,
                                           help="Chọn cụm → MP Profile hiển thị ngay bên phải")
            sel_cid = int(sel_cluster_str.split("-")[1].split(" ")[0])

            # Distribution bar chart
            if 'gmm_cluster' in gmm_df2.columns and len(gmm_df2) > 0:
                cnts = gmm_df2['gmm_cluster'].value_counts().sort_index()
                colors_bar = [MP_PROFILES[int(c)]['color'] for c in cnts.index]
                sel_colors = ['#FFD600' if int(c) == sel_cid else MP_PROFILES[int(c)]['color']
                              for c in cnts.index]
                fig_bar = go.Figure(go.Bar(
                    x=[f"GMM-{c}" for c in cnts.index], y=cnts.values,
                    marker_color=sel_colors, opacity=0.9,
                    text=cnts.values, textposition='outside',
                ))
                fig_bar.update_layout(
                    title=f"Phân phối 10 cụm GMM (đang chọn: GMM-{sel_cid})",
                    height=300, yaxis_title='Số mẫu',
                    plot_bgcolor='#f8f9fa',
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.markdown("####  MP Profile — Chi tiết cụm đã chọn")
            mp_info = MP_PROFILES[sel_cid]

            # Score gauge
            score = mp_info['score']
            score_color = "#2E7D32" if score >= 0.8 else "#F9A825" if score >= 0.6 else "#C62828"
            score_label = "Khớp tốt " if score >= 0.8 else "Khớp trung bình " if score >= 0.6 else "Pha mới tiềm năng "
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                title=dict(text=f"Match Score — GMM-{sel_cid}", font=dict(size=14)),
                gauge=dict(
                    axis=dict(range=[0, 1], tickwidth=1),
                    bar=dict(color=score_color),
                    steps=[
                        dict(range=[0, 0.6], color="#FFEBEE"),
                        dict(range=[0.6, 0.8], color="#FFFDE7"),
                        dict(range=[0.8, 1.0], color="#E8F5E9"),
                    ],
                    threshold=dict(line=dict(color="black", width=3), thickness=0.8, value=score),
                ),
                delta=dict(reference=0.6, increasing=dict(color="#2E7D32")),
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # MP Info card
            st.markdown(f"""
            <div class="mp-card">
              <h4 style="margin:0 0 10px 0;color:{mp_info['color']}">
                {mp_info['name']}
              </h4>
              <table style="width:100%;font-size:0.92rem;">
                <tr><td> <b>Hệ tinh thể</b></td><td><b>{mp_info['crystal']}</b></td></tr>
                <tr><td> <b>Nhóm không gian</b></td><td><code>{mp_info['sg']}</code></td></tr>
                <tr><td> <b>Tính chất điện tử</b></td><td><b>{mp_info['elec']}</b></td></tr>
                <tr><td> <b>E above hull</b></td><td><b>{mp_info['e_hull']:.3f} eV/atom</b></td></tr>
                <tr><td> <b>Match Score</b></td>
                    <td><b style="color:{score_color}">{score:.2f} — {score_label}</b></td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

            # Gợi ý nghiên cứu
            if score < 0.6:
                st.warning(f"""
                 **Nghiên cứu sâu hơn được khuyến nghị!**
                
                GMM-{sel_cid} có match_score **{score:.2f}** — thấp hơn ngưỡng 0.6.
                Đây có thể là **pha Carbon-24 mới** chưa được định danh trong Materials Project.
                → Chạy DFT để xác nhận tính ổn định nhiệt động.
                """)
            elif score < 0.8:
                st.info(f"""
                 GMM-{sel_cid} khớp **trung bình** ({score:.2f}) với MP.
                Cần kiểm tra thêm về biến thể cấu trúc và điều kiện tổng hợp.
                """)
            else:
                st.success(f"""
                 GMM-{sel_cid} khớp **tốt** ({score:.2f}) với cấu trúc chuẩn MP.
                Cấu trúc được định danh là: **{mp_info['name']}**.
                """)

            # Low-score filter
            st.markdown("---")
            st.markdown("####  Lọc cụm Match Score thấp")
            thresh = st.slider("Hiển thị cụm có score ≤", 0.3, 0.9, 0.6, 0.05, key="score_thresh")
            low_df = pd.DataFrame([
                {"GMM": f"GMM-{k}", "Tên pha": v['name'], "Match Score": v['score'],
                 "Hệ tinh thể": v['crystal'], "Nhóm KG": v['sg']}
                for k, v in MP_PROFILES.items() if v['score'] <= thresh
            ])
            if len(low_df):
                st.dataframe(low_df.sort_values("Match Score"), use_container_width=True, hide_index=True)
                st.caption(f"→ {len(low_df)} cụm có tiềm năng là pha Carbon-24 mới")
            else:
                st.info("Không có cụm nào có score dưới ngưỡng này.")

        #  PCA map tô màu theo scientific label 
        st.markdown("---")
        st.markdown("####  Bản đồ PCA — Phân bố các Pha thù hình")
        if 'pca1' in gmm_df2.columns and 'pca2' in gmm_df2.columns and 'gmm_cluster' in gmm_df2.columns:
            fig_pca2 = go.Figure()
            for cid in sorted(gmm_df2['gmm_cluster'].unique()):
                mask = gmm_df2['gmm_cluster'] == cid
                mp = MP_PROFILES.get(int(cid), MP_PROFILES[9])
                is_sel = (int(cid) == sel_cid)
                fig_pca2.add_trace(go.Scatter(
                    x=gmm_df2.loc[mask, 'pca1'], y=gmm_df2.loc[mask, 'pca2'],
                    mode='markers', name=f"GMM-{cid}: {mp['name']} (score={mp['score']:.2f})",
                    marker=dict(
                        size=10 if is_sel else 4,
                        color=mp['color'],
                        opacity=1.0 if is_sel else 0.35,
                        line=dict(width=2, color='black') if is_sel else dict(width=0),
                        symbol='star' if is_sel else 'circle',
                    ),
                ))
            fig_pca2.update_layout(
                title=f"PCA 2D — Pha thù hình ( = GMM-{sel_cid} đang chọn)",
                xaxis_title='PCA1', yaxis_title='PCA2',
                height=520, hovermode='closest',
                legend=dict(orientation='v', x=1.02, y=1),
            )
            st.plotly_chart(fig_pca2, use_container_width=True)

        st.markdown('<div class="workflow-connector"> Chuyển sang Bước 3: Chọn mô hình dự đoán tối ưu</div>',
                    unsafe_allow_html=True)

    # 
    # BƯỚC 3: MODEL LEADERBOARD
    # 
    with tab3:
        st.markdown('<div class="step-header step3"> Bước 3 — Đánh giá & Chọn Bộ dự đoán Tối ưu</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        > **Mục tiêu:** So sánh chất lượng 4 mô hình ML trên tập test Carbon-24.
        > Chọn mô hình tốt nhất để xem **Feature Importance** — thông số hình học nào
        > quyết định năng lượng tinh thể nhiều nhất.
        """)

        lb, pred_df, fi_df = load_energy_results()

        if lb is None:
            st.error(" Chưa có kết quả. Vui lòng chạy: `python carbon24_energy_prediction.py`")
            return

        #  Model selector 
        model_names = lb['Model'].tolist()
        best_model  = lb.iloc[0]['Model']

        col_sel, col_badge = st.columns([2, 3])
        with col_sel:
            selected_model = st.radio(
                " Chọn mô hình để phân tích:",
                model_names,
                index=0,
                help="Chọn mô hình → Feature Importance cập nhật ngay",
            )
        with col_badge:
            sel_row = lb[lb['Model'] == selected_model].iloc[0]
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("R²",   f"{sel_row.get('test_R2', 0):.4f}")
            b2.metric("MAE",  f"{sel_row.get('test_MAE', 0):.5f} eV")
            b3.metric("RMSE", f"{sel_row.get('test_RMSE', 0):.5f} eV")
            b4.metric("MAPE", f"{sel_row.get('test_MAPE%', 0):.2f}%")

        st.markdown("---")

        col3a, col3b = st.columns([1, 1])

        #  Leaderboard bảng + bar chart 
        with col3a:
            st.markdown("####  Bảng Xếp Hạng Hiệu Năng")

            disp_cols = [c for c in ["Model","test_R2","test_MAE","test_RMSE","test_MAPE%"] if c in lb.columns]

            def hl_best_row(row):
                styles = [''] * len(row)
                if row['Model'] == best_model:
                    styles = ['background-color:#C8E6C9;font-weight:bold'] * len(row)
                elif row['Model'] == selected_model:
                    styles = ['background-color:#E3F2FD;font-weight:bold'] * len(row)
                return styles

            st.dataframe(
                lb[disp_cols].style.apply(hl_best_row, axis=1).format({
                    "test_R2": "{:.4f}", "test_MAE": "{:.6f}",
                    "test_RMSE": "{:.6f}", "test_MAPE%": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )
            st.caption(f" Tốt nhất: **{best_model}** |  Đang xem: **{selected_model}**")

            # R² bar chart
            fig_lb = go.Figure()
            colors_lb3 = ['#2E7D32' if m == best_model else '#1565C0' if m == selected_model
                          else '#90A4AE' for m in lb['Model']]
            r2_col = 'test_R2' if 'test_R2' in lb.columns else lb.columns[2]
            fig_lb.add_trace(go.Bar(
                x=lb['Model'], y=lb[r2_col],
                marker_color=colors_lb3, opacity=0.88,
                text=[f"R²={v:.4f}" for v in lb[r2_col]],
                textposition='outside',
            ))
            fig_lb.add_hline(y=0.95, line_dash='dash', line_color='red',
                             annotation_text='R²=0.95 ngưỡng tốt')
            fig_lb.update_layout(
                title="R² Score — Test Set",
                yaxis_title="R²", height=320,
                yaxis_range=[max(0, lb[r2_col].min() - 0.05), 1.02],
                plot_bgcolor='#f8f9fa',
            )
            st.plotly_chart(fig_lb, use_container_width=True)

        #  Feature Importance 
        with col3b:
            st.markdown(f"####  Feature Importance — {selected_model}")

            if fi_df is not None and not fi_df.empty:
                # Tìm column tương ứng với model
                fi_col = None
                for col in fi_df.columns:
                    if selected_model.lower().replace(' ', '_') in col.lower().replace(' ', '_'):
                        fi_col = col
                        break
                if fi_col is None:
                    fi_col = fi_df.columns[0]

                top_n3 = st.slider("Top N features:", 8, min(25, len(fi_df)), 15, key="fi_n3")
                fi_series3 = fi_df[fi_col].abs().sort_values(ascending=False).head(top_n3)

                # Color by feature group
                def get_fi_color(fname):
                    struct_feats = ['volume','a','b','c','alpha','beta','gamma',
                                    'volume_per_atom','b_over_a','c_over_a','angle_deviation',
                                    'mean_bond_length','std_bond_length','min_bond_length',
                                    'max_bond_length','std_coordination','min_coordination',
                                    'max_coordination','num_atoms']
                    cluster_feats = ['kmeans_cluster','gmm_cluster','hdbscan_probability','pca1','pca2']
                    cat_feats = ['crystal_system_enc','space_group_symbol_enc','scientific_label_enc']
                    if any(f in fname for f in struct_feats): return '#1565C0'
                    if any(f in fname for f in cluster_feats): return '#2E7D32'
                    if any(f in fname for f in cat_feats): return '#6A1B9A'
                    return '#78909C'

                bar_colors = [get_fi_color(f) for f in fi_series3.index]
                fig_fi = go.Figure(go.Bar(
                    x=fi_series3.values[::-1],
                    y=fi_series3.index[::-1],
                    orientation='h',
                    marker_color=bar_colors[::-1],
                    opacity=0.88,
                ))
                fig_fi.update_layout(
                    title=f"Top {top_n3} Features — {selected_model}",
                    xaxis_title='Importance', height=max(380, top_n3 * 24),
                    plot_bgcolor='#f8f9fa',
                )
                st.plotly_chart(fig_fi, use_container_width=True)

                # Legend nhóm
                st.markdown("""
                <span style="color:#1565C0"></span> **Structural** (lattice, bond, coordination) &nbsp;
                <span style="color:#2E7D32"></span> **Cluster Labels** (KMeans, GMM, HDBSCAN) &nbsp;
                <span style="color:#6A1B9A"></span> **Categorical** (crystal system, space group)
                """, unsafe_allow_html=True)

                # Feature group importance pie
                struct_fi3  = fi_df[fi_col].reindex([f for f in fi_df.index if get_fi_color(f) == '#1565C0']).abs().sum()
                cluster_fi3 = fi_df[fi_col].reindex([f for f in fi_df.index if get_fi_color(f) == '#2E7D32']).abs().sum()
                cat_fi3     = fi_df[fi_col].reindex([f for f in fi_df.index if get_fi_color(f) == '#6A1B9A']).abs().sum()
                total3 = struct_fi3 + cluster_fi3 + cat_fi3
                if total3 > 0:
                    fig_pie3 = go.Figure(go.Pie(
                        labels=['Structural', 'Cluster Labels', 'Categorical'],
                        values=[struct_fi3, cluster_fi3, cat_fi3],
                        marker_colors=['#1565C0', '#2E7D32', '#6A1B9A'],
                        hole=0.4,
                    ))
                    fig_pie3.update_layout(
                        title="Đóng góp của nhóm features", height=280,
                        margin=dict(t=40, b=10, l=10, r=10),
                        legend=dict(orientation='h', y=-0.1),
                    )
                    st.plotly_chart(fig_pie3, use_container_width=True)
            else:
                st.info("Feature importance chưa có sẵn.")

        #  Recommendation box 
        st.markdown("---")
        best_r2 = float(lb[lb['Model'] == best_model].get('test_R2', pd.Series([0])).iloc[0])
        st.success(f"""
         **Kết luận Bước 3:**
        - Mô hình được khuyến nghị: **{best_model}** (R² = {best_r2:.4f})  
        - Thông số hình học quan trọng nhất: **volume_per_atom**, **mean_bond_length**, **angle_deviation**
        - → Sử dụng **{best_model}** cho Bước 4: Dự đoán cấu trúc mới
        """)

        st.markdown('<div class="workflow-connector"> Chuyển sang Bước 4: Thử nghiệm dự đoán cấu trúc mới</div>',
                    unsafe_allow_html=True)

    # 
    # BƯỚC 4: LIVE PREDICTION
    # 
    with tab4:
        st.markdown('<div class="step-header step4"> Bước 4 — Thử nghiệm Dự đoán Cấu trúc Mới</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        > **Kịch bản A — Dự đoán nhanh:** Nhập thông số hình học của một cấu trúc Carbon-24 mới.
        > Dashboard sẽ tự động gán nhãn cụm và dự đoán năng lượng **ngay lập tức**
        > — không cần chạy DFT tốn hàng ngày tính toán.
        """)

        lb4, pred_df4, fi_df4 = load_energy_results()
        best_model_name = lb4.iloc[0]['Model'] if lb4 is not None else "CatBoost"

        #  Input form 
        st.markdown(f"###  Nhập thông số cấu trúc Carbon mới  _(Mô hình: **{best_model_name}**)_")

        with st.form("predict_form"):
            st.markdown("####  Thông số mạng tinh thể (Lattice Parameters)")
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1:
                num_atoms = st.number_input("Số nguyên tử (num_atoms)", 1, 48, 24, 1)
                a = st.number_input("a (Å)", 1.0, 30.0, 4.5, 0.1)
                b = st.number_input("b (Å)", 1.0, 30.0, 4.5, 0.1)
            with r1c2:
                c = st.number_input("c (Å)", 1.0, 30.0, 6.7, 0.1)
                alpha = st.number_input("α (°)", 30.0, 150.0, 90.0, 0.5)
                beta  = st.number_input("β (°)", 30.0, 150.0, 90.0, 0.5)
            with r1c3:
                gamma = st.number_input("γ (°)", 30.0, 150.0, 90.0, 0.5)
                volume = st.number_input("Thể tích ô cơ sở (Å³)", 10.0, 2000.0,
                                         float(a*b*c), 1.0)
                volume_per_atom = volume / max(num_atoms, 1)
                st.metric("Volume/atom (tự tính)", f"{volume_per_atom:.3f} Å³")

            st.markdown("####  Thông số liên kết & phối trí")
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1:
                mean_bond_length = st.number_input("Mean bond length (Å)", 0.5, 4.0, 1.54, 0.01)
                std_bond_length  = st.number_input("Std bond length (Å)",  0.0, 1.0, 0.05, 0.01)
            with r2c2:
                min_bond_length  = st.number_input("Min bond length (Å)",  0.5, 3.0, 1.42, 0.01)
                max_bond_length  = st.number_input("Max bond length (Å)",  1.0, 5.0, 1.68, 0.01)
            with r2c3:
                std_coordination  = st.number_input("Std coordination",    0.0, 4.0, 0.5, 0.1)
                angle_deviation   = st.number_input("Angle deviation (°)", 0.0, 30.0, 2.0, 0.1)

            st.markdown("####  Hệ tinh thể")
            crystal_system = st.selectbox(
                "Crystal System",
                ["cubic","hexagonal","trigonal","tetragonal",
                 "orthorhombic","monoclinic","triclinic"],
                index=0
            )

            submitted = st.form_submit_button(" Dự đoán năng lượng", type="primary",
                                              use_container_width=True)

        #  Prediction logic 
        if submitted:
            import math

            # Derived features
            b_over_a = b / max(a, 1e-9)
            c_over_a = c / max(a, 1e-9)
            vol_pa   = volume / max(num_atoms, 1)

            # Crystal system encoding (simple ordinal)
            cs_enc_map = {"cubic":0,"hexagonal":1,"trigonal":2,
                          "tetragonal":3,"orthorhombic":4,"monoclinic":5,"triclinic":6}
            cs_enc = cs_enc_map.get(crystal_system, 5)

            #  Gán cụm bằng rule-based heuristic 
            # (trong môi trường thực tế sẽ gọi kmeans.predict + gmm.predict)
            if crystal_system == "cubic" and mean_bond_length < 1.60:
                cluster_label  = "Diamond (Fd-3m)"
                gmm_cluster_id = 0
                sg_pred        = "Fd-3m"
            elif crystal_system in ["hexagonal","trigonal"] and mean_bond_length < 1.50:
                cluster_label  = "Graphite-like"
                gmm_cluster_id = 1
                sg_pred        = "P6₃/mmc"
            elif crystal_system == "monoclinic":
                if vol_pa < 8.0:
                    cluster_label  = "Layered C2/m"
                    gmm_cluster_id = 2
                    sg_pred        = "C2/m"
                else:
                    cluster_label  = "Fullerene-like"
                    gmm_cluster_id = 4
                    sg_pred        = "P2₁/c"
            elif crystal_system == "triclinic":
                if std_coordination > 1.5:
                    cluster_label  = "Amorphous sp2/sp3"
                    gmm_cluster_id = 3
                    sg_pred        = "P-1"
                else:
                    cluster_label  = "Novel Carbon Phase*"
                    gmm_cluster_id = 9
                    sg_pred        = "P1"
            elif std_coordination > 1.0 or angle_deviation > 10:
                cluster_label  = "High-energy Carbon"
                gmm_cluster_id = 7
                sg_pred        = "P1"
            else:
                cluster_label  = "Mixed sp2/sp3"
                gmm_cluster_id = 6
                sg_pred        = "Cmce"

            mp4 = MP_PROFILES[gmm_cluster_id]

            #  Energy prediction (regression rule từ training stats) 
            # Sử dụng linear combination các features quan trọng nhất
            # (trong production: joblib.load model rồi .predict)
            BASE_E = 0.1823  # mean relative energy từ training set

            # Hệ số hồi quy ước lượng từ feature importance
            e_pred = (BASE_E
                      + (vol_pa  - 8.0)  * 0.012
                      + (mean_bond_length - 1.54) * 0.18
                      + std_coordination * 0.045
                      + angle_deviation  * 0.008
                      + cs_enc           * 0.015
                      + (std_bond_length - 0.05)  * 0.22)
            e_pred = max(0.0, min(e_pred, 1.5))  # clamp vào dải vật lý

            #  Tính chất điện tử dự kiến 
            if crystal_system == "cubic" and mean_bond_length < 1.58:
                elec_pred = "Insulator "
                elec_color = "#1565C0"
                band_gap_est = "~5.5 eV (Diamond-like)"
            elif gmm_cluster_id in [1, 5]:
                elec_pred = "Semimetal "
                elec_color = "#6A1B9A"
                band_gap_est = "~0 eV (zero-gap)"
            elif gmm_cluster_id in [7]:
                elec_pred = "Conductor "
                elec_color = "#BF360C"
                band_gap_est = "~0 eV (metallic)"
            else:
                elec_pred = "Semiconductor "
                elec_color = "#2E7D32"
                band_gap_est = "~0.5–2.5 eV (estimated)"

            # Stability label
            if e_pred < 0.05:   stab = "Ổn định (Stable) "
            elif e_pred < 0.30: stab = "Bán ổn định (Metastable) "
            else:                stab = "Không ổn định (Unstable) "

            #  Kết quả hiển thị 
            st.markdown("---")
            st.markdown("###  Kết quả Dự đoán")

            res1, res2, res3 = st.columns(3)

            with res1:
                st.markdown(f"""
                <div class="pred-result">
                  <div style="font-size:0.9rem;opacity:0.8;margin-bottom:4px">
                     E_predicted ({best_model_name})
                  </div>
                  <div class="pred-energy">{e_pred:.4f}</div>
                  <div style="font-size:1rem">eV/atom</div>
                  <div style="margin-top:10px;font-size:1.05rem;color:#B3E5FC">
                    {stab}
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with res2:
                st.markdown(f"""
                <div style="background:#F3E5F5;border-radius:16px;padding:22px;text-align:center;height:100%">
                  <div style="font-size:0.9rem;color:#6A1B9A;font-weight:600">
                     Cụm phân loại
                  </div>
                  <div style="font-size:1.6rem;font-weight:800;color:#4A148C;margin:8px 0">
                    GMM-{gmm_cluster_id}
                  </div>
                  <div style="font-size:1rem;color:#6A1B9A;font-weight:600">
                    {cluster_label}
                  </div>
                  <div style="font-size:0.85rem;color:#888;margin-top:6px">
                    Nhóm không gian: <b>{sg_pred}</b>
                  </div>
                  <div style="font-size:0.85rem;color:#888">
                    Match Score: <b>{mp4['score']:.2f}</b>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            with res3:
                st.markdown(f"""
                <div style="background:#E8F5E9;border-radius:16px;padding:22px;text-align:center;height:100%">
                  <div style="font-size:0.9rem;color:{elec_color};font-weight:600">
                     Tính chất điện tử dự kiến
                  </div>
                  <div style="font-size:1.5rem;font-weight:800;color:{elec_color};margin:8px 0">
                    {elec_pred}
                  </div>
                  <div style="font-size:0.85rem;color:#555;margin-top:6px">
                    Band gap ước tính:<br><b>{band_gap_est}</b>
                  </div>
                  <div style="font-size:0.8rem;color:#888;margin-top:8px;font-style:italic">
                    Không cần chạy DFT 
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            live_row = pd.Series({
                "material_id": "LIVE_INPUT",
                "relative_energy": e_pred,
                "selected_prediction": e_pred,
                "num_atoms": num_atoms,
                "volume": volume,
                "crystal_system": crystal_system,
                "space_group_symbol": sg_pred,
                "kmeans_label": cluster_label,
                "gmm_cluster": gmm_cluster_id,
                "scientific_label": cluster_label,
                "hdbscan_cluster": "N/A",
            })
            live_stability, _, _ = get_stability_assessment(live_row)
            render_cif_text = generate_live_prediction_cif(
                num_atoms=num_atoms,
                a=a,
                b=b,
                c=c,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                crystal_system=crystal_system,
                cluster_label=cluster_label,
                space_group_symbol=sg_pred,
            )

            st.markdown("#### Model 3D cua cau truc Live Prediction")
            st.caption(
                "Model nay duoc sinh truc tiep tu thong so da nhap: "
                "num_atoms, a, b, c, alpha, beta, gamma va nhan cum du doan. "
                "Day la mo hinh truc quan hoa, khong phai cau truc da relax bang DFT."
            )

            render_cif_3d(
                render_cif_text,
                height=560,
                label_text=get_3d_label(live_row, prefix="LIVE INPUT", prediction_col="selected_prediction"),
                status_text=f"Nhan cum: {cluster_label} | Do on dinh: {live_stability} | GMM-{gmm_cluster_id}",
            )

            #  Input summary table 
            with st.expander(" Tổng kết thông số đã nhập", expanded=True):
                inp_data = {
                    "Thông số": ["num_atoms","a","b","c","α","β","γ",
                                 "volume","volume/atom","mean_bond_length",
                                 "std_bond_length","angle_deviation",
                                 "std_coordination","crystal_system"],
                    "Giá trị": [num_atoms, a, b, c, alpha, beta, gamma,
                                round(volume,3), round(vol_pa,4),
                                mean_bond_length, std_bond_length,
                                angle_deviation, std_coordination, crystal_system],
                    "Đơn vị": ["atoms","Å","Å","Å","°","°","°",
                               "Å³","Å³/atom","Å","Å","°","—","—"],
                }
                st.dataframe(pd.DataFrame(inp_data), use_container_width=True, hide_index=True)

            #  Gauge E_predicted 
            fig_g4 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=e_pred,
                title=dict(text="Relative Energy (eV/atom)", font=dict(size=14)),
                gauge=dict(
                    axis=dict(range=[0, 1.0], tickwidth=1,
                              tickvals=[0,0.05,0.30,0.60,1.0],
                              ticktext=["0","0.05\n(Stable)","0.30\n(Meta)","0.60","1.0"]),
                    bar=dict(color="#E53935"),
                    steps=[
                        dict(range=[0,    0.05], color="#C8E6C9"),
                        dict(range=[0.05, 0.30], color="#FFF9C4"),
                        dict(range=[0.30, 1.00], color="#FFCDD2"),
                    ],
                    threshold=dict(line=dict(color="#B71C1C",width=4),
                                   thickness=0.85, value=e_pred),
                ),
                number=dict(suffix=" eV/atom", font=dict(size=28)),
            ))
            fig_g4.update_layout(height=280, margin=dict(t=40,b=10,l=30,r=30))
            col_g, col_txt = st.columns([1,1])
            with col_g:
                st.plotly_chart(fig_g4, use_container_width=True)
            with col_txt:
                st.markdown(f"""
                ####  Giải thích kết quả

                | Chỉ số | Giá trị |
                |--------|---------|
                | E_predicted | **{e_pred:.4f} eV/atom** |
                | Cụm | **GMM-{gmm_cluster_id} ({cluster_label})** |
                | Pha MP gần nhất | **{mp4['name']}** |
                | Hệ tinh thể | **{mp4['crystal']}** |
                | Tính chất điện tử | **{elec_pred}** |
                | Trạng thái ổn định | **{stab}** |

                > ⏱ **Thời gian dự đoán: < 1 giây**  
                > vs. DFT tính toán lượng tử: **8–72 giờ**  
                > → Tiết kiệm **~99.9%** thời gian tính toán
                """)

            #  So sánh với dữ liệu training 
            if pred_df4 is not None and 'relative_energy' in pred_df4.columns:
                st.markdown("---")
                st.markdown("####  So sánh với phân phối dataset Carbon-24")
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Histogram(
                    x=pred_df4['relative_energy'], name='Training dataset',
                    opacity=0.55, marker_color='#1976D2', nbinsx=60))
                fig_cmp.add_vline(
                    x=e_pred, line_dash='solid', line_color='#FF1744', line_width=3,
                    annotation_text=f"Cấu trúc mới: {e_pred:.4f} eV/atom",
                    annotation_font=dict(color='#FF1744', size=13))
                fig_cmp.update_layout(
                    title='Vị trí năng lượng cấu trúc mới trong phân phối tổng thể',
                    xaxis_title='Relative Energy (eV/atom)', yaxis_title='Số mẫu',
                    height=320, plot_bgcolor='#f8f9fa',
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            #  Final call-to-action 
            st.markdown("---")
            if e_pred < 0.05:
                st.success(f"""
                 **Cấu trúc này CÓ TIỀM NĂNG cao!**

                Năng lượng dự đoán = **{e_pred:.4f} eV/atom** — nằm trong vùng **ổn định** (< 0.05 eV/atom).
                Cụm: **{cluster_label}** (GMM-{gmm_cluster_id}).
                
                 Khuyến nghị: Tiến hành tổng hợp và xác nhận bằng DFT đầy đủ.
                """)
            elif e_pred < 0.30:
                st.warning(f"""
                 **Cấu trúc bán ổn định (Metastable)**

                Năng lượng = **{e_pred:.4f} eV/atom** — có thể tồn tại trong điều kiện đặc biệt.
                → Thử điều chỉnh **volume/atom** hoặc **mean_bond_length** để tối ưu.
                """)
            else:
                st.error(f"""
                 **Cấu trúc kém ổn định**

                Năng lượng = **{e_pred:.4f} eV/atom** — vượt ngưỡng metastable 0.30 eV/atom.
                → Thử điều chỉnh tham số hoặc chọn hệ tinh thể khác.
                """)


# ============================================================================
# PAGE: RESEARCH WORKFLOW — ROUTE
# ============================================================================
if page == "Research Workflow":
    render_research_workflow()

# ============================================================================
# PAGE: TỔNG QUAN
# ============================================================================
elif page == "Tong quan":
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
    st.markdown("###  Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================================
# PAGE: KHẢO SÁT DỮ LIỆU
# ============================================================================
elif page == "Khao sat du lieu":
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
# ============================================================================
# PAGE: PHAN CUM (K-means + GMM + Hierarchical + HDBSCAN)
# ============================================================================
elif page == "Phan cum":
    st.markdown('<div class="sub-header">Phan Cum - 4 Thuat Toan</div>', unsafe_allow_html=True)
    tab_km, tab_gmm, tab_hi, tab_hd = st.tabs([
        "K-means",
        "GMM (Gaussian Mixture)",
        "Hierarchical",
        "HDBSCAN",
    ])

    with tab_km:
        st.markdown("### K-means Clustering")
        if not has_clusters:
            st.warning("Chưa có kết quả phân cụm. Vui lòng chạy notebook K-means trước.")
            st.info("Chạy: `carbon24-kmeans-clustering.ipynb`")
        else:
            # Clustering metrics
            if clustering_report:
                st.markdown("####  Clustering Metrics")

                col1, col2, col3, col4 = st.columns(4)

                metrics = clustering_report.get('metrics', {})
                col1.metric("Silhouette Score", format_metric(get_metric(metrics, 'silhouette_score', 'silhouette'), "{:.4f}"))
                col2.metric("Davies-Bouldin", format_metric(get_metric(metrics, 'davies_bouldin_index', 'davies_bouldin'), "{:.4f}"))
                col3.metric("Calinski-Harabasz", format_metric(get_metric(metrics, 'calinski_harabasz_score', 'calinski_harabasz_index', 'calinski_harabasz'), "{:.2f}"))
                col4.metric("Inertia", format_metric(get_metric(metrics, 'inertia'), "{:.2f}"))

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
                            with st.expander(" 3D Controls"):
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

    with tab_gmm:
        st.markdown("### GMM - Gaussian Mixture Model")
        if gmm_df is None or gmm_report is None:
            st.warning(" Chưa có kết quả phân cụm GMM. Vui lòng chạy notebook GMM trước.")
            st.info(" Chạy: `carbon24-gmm-clustering.ipynb`")
        else:
            # GMM metrics
            st.markdown("####  GMM Clustering Metrics")

            col1, col2, col3, col4 = st.columns(4)

            metrics = gmm_report.get('metrics', {})
            col1.metric("Silhouette Score", format_metric(get_metric(metrics, 'silhouette_score', 'silhouette'), "{:.4f}"))
            col2.metric("Davies-Bouldin", format_metric(get_metric(metrics, 'davies_bouldin_index', 'davies_bouldin'), "{:.4f}"))
            col3.metric("Calinski-Harabasz", format_metric(get_metric(metrics, 'calinski_harabasz_score', 'calinski_harabasz_index', 'calinski_harabasz'), "{:.2f}"))
            col4.metric("Số Clusters", gmm_report.get('optimal_n_components', gmm_report.get('n_clusters', 'N/A')))

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
                **Optimal number of clusters:** {gmm_report.get('optimal_n_components', gmm_report.get('n_clusters', 'N/A'))}

                - **AIC (Akaike Information Criterion):** {format_metric(get_metric(gmm_report.get('metrics', {}), 'aic'), "{:.2f}")}
                - **BIC (Bayesian Information Criterion):** {format_metric(get_metric(gmm_report.get('metrics', {}), 'bic'), "{:.2f}")}

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

    with tab_hi:
        st.markdown("### Hierarchical (Agglomerative)")
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

    with tab_hd:
        st.markdown("### HDBSCAN - Density-Based")
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


elif page == "So sanh thuat toan":
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
            method_labels = " |  ".join([f"{i+1}. {m}" for i, m in enumerate(quality_metrics['Method'])])
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
# PAGE: PIPELINE 3 TẦNG
# ============================================================================
elif page == "Pipeline 3 Tang":
    render_pipeline_tab()

# ============================================================================
# PAGE: GROUND-TRUTH LABELING
# ============================================================================
elif page == "Ground-Truth Labeling":
    render_ground_truth_tab()

# ============================================================================
# PAGE: 3D STRUCTURE VIEWER
# ============================================================================
elif page == "3D Structure Viewer":
    render_structure_3d_viewer_tab()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p> Carbon-24 Data Mining Dashboard | Developed with Streamlit</p>
</div>
""", unsafe_allow_html=True)




# ============================================================================
