"""
Carbon-24 Interactive Dashboard
================================
Dashboard tương tác để khám phá kết quả phân tích stability và energy prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page config
st.set_page_config(
    page_title="Carbon-24 Stability Analysis Dashboard",
    page_icon="💎",
    layout="wide"
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
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">💎 Carbon-24 Stability Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    # Main dataset
    df = pd.read_csv('carbon24_feature_selected/carbon24_feature_selected_standard.csv')
    
    # Original data for energy_per_atom
    df_original = pd.read_csv('carbon24_features/carbon24_project/data/carbon24_features.csv')
    df_original['energy_per_atom'] = df_original['energy'] / df_original['num_atoms']
    df = df.merge(df_original[['row_index', 'energy_per_atom']], on='row_index', how='left')
    
    # Stability classification
    stability_df = pd.read_csv('carbon24_stability_analysis/cluster_stability_classification.csv')
    
    # Model comparison
    model_results = pd.read_csv('carbon24_stability_analysis/prediction_model_comparison.csv')
    
    # Predictions
    predictions = pd.read_csv('carbon24_stability_analysis/best_model_predictions.csv')
    
    # Feature importance
    try:
        feature_importance = pd.read_csv('carbon24_stability_analysis/feature_importance.csv')
    except:
        feature_importance = None
    
    # Features info
    with open('carbon24_feature_selected/selected_features.json', 'r') as f:
        feature_info = json.load(f)
    
    return df, stability_df, model_results, predictions, feature_importance, feature_info

try:
    df, stability_df, model_results, predictions, feature_importance, feature_info = load_data()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["Overview", "Stability Analysis", "Energy Prediction", "Cluster Explorer", "Data Explorer"]
    )
    
    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "Overview":
        st.markdown('<div class="sub-header">📋 Project Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Structures", f"{len(df):,}")
        with col2:
            st.metric("Clusters", len(stability_df))
        with col3:
            best_r2 = model_results['Test R²'].max()
            st.metric("Best Model R²", f"{best_r2:.4f}")
        with col4:
            best_mae = model_results['Test MAE'].min()
            st.metric("Best MAE", f"{best_mae:.4f} eV/atom")
        
        st.markdown("---")
        
        # Key findings
        st.markdown('<div class="sub-header">🔍 Key Findings</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Stability Classification")
            for _, row in stability_df.iterrows():
                cluster_count = (df['cluster'] == row['cluster']).sum()
                percentage = cluster_count / len(df) * 100
                
                if row['stability'] == 'Highly Stable':
                    emoji = "🟢"
                elif row['stability'] == 'Moderately Stable':
                    emoji = "🟡"
                else:
                    emoji = "🔴"
                
                st.markdown(f"""
                {emoji} **Cluster {row['cluster']}: {row['stability']}**
                - Structures: {cluster_count:,} ({percentage:.1f}%)
                - Mean Energy: {row['mean_energy']:.4f} eV/atom
                """)
        
        with col2:
            st.markdown("### Model Performance")
            best_model = model_results.loc[model_results['Test R²'].idxmax(), 'Model']
            st.markdown(f"**Best Model:** {best_model}")
            
            for _, row in model_results.iterrows():
                st.markdown(f"""
                **{row['Model']}**
                - Test R²: {row['Test R²']:.4f}
                - Test MAE: {row['Test MAE']:.4f} eV/atom
                - Time: {row['Time (s)']:.2f}s
                """)
        
        # Visualizations
        st.markdown("---")
        st.markdown('<div class="sub-header">📊 Quick Visualizations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            fig = px.pie(
                values=df['cluster'].value_counts().values,
                names=[f"Cluster {i}" for i in df['cluster'].value_counts().index],
                title="Cluster Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model comparison
            fig = px.bar(
                model_results,
                x='Model',
                y='Test R²',
                title="Model Performance Comparison",
                color='Test R²',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 2: STABILITY ANALYSIS
    # ========================================================================
    elif page == "Stability Analysis":
        st.markdown('<div class="sub-header">⚡ Stability Analysis</div>', unsafe_allow_html=True)
        
        # Stability metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            highly_stable = len(df[df['cluster'].isin(
                stability_df[stability_df['stability'] == 'Highly Stable']['cluster']
            )])
            st.metric("Highly Stable", f"{highly_stable:,}", 
                     f"{highly_stable/len(df)*100:.1f}%")
        
        with col2:
            moderately_stable = len(df[df['cluster'].isin(
                stability_df[stability_df['stability'] == 'Moderately Stable']['cluster']
            )])
            st.metric("Moderately Stable", f"{moderately_stable:,}",
                     f"{moderately_stable/len(df)*100:.1f}%")
        
        with col3:
            less_stable = len(df[df['cluster'].isin(
                stability_df[stability_df['stability'] == 'Less Stable']['cluster']
            )])
            st.metric("Less Stable", f"{less_stable:,}",
                     f"{less_stable/len(df)*100:.1f}%")
        
        st.markdown("---")
        
        # Relative energy distribution
        st.markdown("### Relative Energy Distribution by Cluster")
        
        fig = go.Figure()
        
        for i in range(len(stability_df)):
            cluster_data = df[df['cluster'] == i]['relative_energy']
            stability_label = stability_df[stability_df['cluster'] == i]['stability'].values[0]
            
            fig.add_trace(go.Histogram(
                x=cluster_data,
                name=f"Cluster {i}: {stability_label}",
                opacity=0.7,
                nbinsx=50
            ))
        
        fig.update_layout(
            barmode='overlay',
            xaxis_title="Relative Energy (eV/atom)",
            yaxis_title="Frequency",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot
        st.markdown("### Stability Comparison")
        
        fig = go.Figure()
        
        for i in range(len(stability_df)):
            cluster_data = df[df['cluster'] == i]['relative_energy']
            stability_label = stability_df[stability_df['cluster'] == i]['stability'].values[0]
            
            fig.add_trace(go.Box(
                y=cluster_data,
                name=f"Cluster {i}: {stability_label}",
                boxmean='sd'
            ))
        
        fig.update_layout(
            yaxis_title="Relative Energy (eV/atom)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics table
        st.markdown("### Detailed Statistics")
        
        stats_data = []
        for i in range(len(stability_df)):
            cluster_data = df[df['cluster'] == i]['relative_energy']
            stats_data.append({
                'Cluster': i,
                'Stability': stability_df[stability_df['cluster'] == i]['stability'].values[0],
                'Count': len(cluster_data),
                'Mean': cluster_data.mean(),
                'Median': cluster_data.median(),
                'Std': cluster_data.std(),
                'Min': cluster_data.min(),
                'Max': cluster_data.max()
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df.style.format({
            'Mean': '{:.4f}',
            'Median': '{:.4f}',
            'Std': '{:.4f}',
            'Min': '{:.4f}',
            'Max': '{:.4f}'
        }), use_container_width=True)
    
    # ========================================================================
    # PAGE 3: ENERGY PREDICTION
    # ========================================================================
    elif page == "Energy Prediction":
        st.markdown('<div class="sub-header">🤖 Energy Prediction Analysis</div>', unsafe_allow_html=True)
        
        # Model comparison
        st.markdown("### Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                model_results,
                x='Model',
                y=['Train R²', 'Test R²'],
                title="R² Score Comparison",
                barmode='group',
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                model_results,
                x='Model',
                y=['Train MAE', 'Test MAE'],
                title="MAE Comparison (eV/atom)",
                barmode='group',
                color_discrete_sequence=['#2ca02c', '#d62728']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Actual vs Predicted
        st.markdown("### Actual vs Predicted Energy")
        
        fig = px.scatter(
            predictions,
            x='actual',
            y='predicted',
            color='abs_error',
            title="Prediction Performance",
            labels={
                'actual': 'Actual Energy per Atom (eV/atom)',
                'predicted': 'Predicted Energy per Atom (eV/atom)',
                'abs_error': 'Absolute Error'
            },
            color_continuous_scale='Reds',
            opacity=0.6
        )
        
        # Add perfect prediction line
        min_val = min(predictions['actual'].min(), predictions['predicted'].min())
        max_val = max(predictions['actual'].max(), predictions['predicted'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Error analysis
        st.markdown("### Prediction Error Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                predictions,
                x='error',
                title="Error Distribution",
                nbins=50,
                labels={'error': 'Prediction Error (eV/atom)'}
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                predictions,
                x='predicted',
                y='error',
                title="Residual Plot",
                labels={
                    'predicted': 'Predicted Energy per Atom (eV/atom)',
                    'error': 'Residual (Actual - Predicted)'
                },
                opacity=0.5
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if feature_importance is not None:
            st.markdown("### Feature Importance")
            
            top_n = st.slider("Number of top features to display", 5, 20, 10)
            
            top_features = feature_importance.head(top_n)
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title=f"Top {top_n} Most Important Features",
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 4: CLUSTER EXPLORER
    # ========================================================================
    elif page == "Cluster Explorer":
        st.markdown('<div class="sub-header">🔍 Cluster Explorer</div>', unsafe_allow_html=True)
        
        # Cluster selection
        selected_cluster = st.selectbox(
            "Select Cluster to Explore:",
            range(len(stability_df)),
            format_func=lambda x: f"Cluster {x}: {stability_df[stability_df['cluster'] == x]['stability'].values[0]}"
        )
        
        cluster_data = df[df['cluster'] == selected_cluster]
        stability_label = stability_df[stability_df['cluster'] == selected_cluster]['stability'].values[0]
        
        # Cluster info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Structures", f"{len(cluster_data):,}")
        with col2:
            st.metric("Stability", stability_label)
        with col3:
            st.metric("Mean Energy", f"{cluster_data['relative_energy'].mean():.4f}")
        with col4:
            st.metric("Median Energy", f"{cluster_data['relative_energy'].median():.4f}")
        
        st.markdown("---")
        
        # Feature distributions
        st.markdown("### Feature Distributions")
        
        numeric_features = feature_info['numeric_features']
        selected_feature = st.selectbox("Select Feature:", numeric_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram for selected cluster
            fig = px.histogram(
                cluster_data,
                x=selected_feature,
                title=f"{selected_feature} Distribution in Cluster {selected_cluster}",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Comparison with all clusters
            fig = go.Figure()
            
            for i in range(len(stability_df)):
                cluster_i_data = df[df['cluster'] == i][selected_feature]
                fig.add_trace(go.Box(
                    y=cluster_i_data,
                    name=f"Cluster {i}",
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=f"{selected_feature} Comparison Across Clusters",
                yaxis_title=selected_feature
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("### Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_x = st.selectbox("X-axis:", numeric_features, key='x')
        with col2:
            feature_y = st.selectbox("Y-axis:", numeric_features, index=1, key='y')
        
        fig = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            color='cluster',
            title=f"{feature_x} vs {feature_y}",
            opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Highlight selected cluster
        fig.add_trace(go.Scatter(
            x=cluster_data[feature_x],
            y=cluster_data[feature_y],
            mode='markers',
            name=f'Cluster {selected_cluster} (highlighted)',
            marker=dict(size=8, color='red', symbol='star')
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 5: DATA EXPLORER
    # ========================================================================
    elif page == "Data Explorer":
        st.markdown('<div class="sub-header">📊 Data Explorer</div>', unsafe_allow_html=True)
        
        # Filters
        st.markdown("### Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cluster_filter = st.multiselect(
                "Select Clusters:",
                range(len(stability_df)),
                default=list(range(len(stability_df))),
                format_func=lambda x: f"Cluster {x}"
            )
        
        with col2:
            energy_range = st.slider(
                "Relative Energy Range:",
                float(df['relative_energy'].min()),
                float(df['relative_energy'].max()),
                (float(df['relative_energy'].min()), float(df['relative_energy'].max()))
            )
        
        with col3:
            num_atoms_range = st.slider(
                "Number of Atoms:",
                int(df['num_atoms'].min()),
                int(df['num_atoms'].max()),
                (int(df['num_atoms'].min()), int(df['num_atoms'].max()))
            )
        
        # Apply filters
        filtered_df = df[
            (df['cluster'].isin(cluster_filter)) &
            (df['relative_energy'] >= energy_range[0]) &
            (df['relative_energy'] <= energy_range[1]) &
            (df['num_atoms'] >= num_atoms_range[0]) &
            (df['num_atoms'] <= num_atoms_range[1])
        ]
        
        st.markdown(f"**Filtered Results:** {len(filtered_df):,} structures")
        
        # Display data
        st.markdown("### Data Table")
        
        display_columns = st.multiselect(
            "Select Columns to Display:",
            df.columns.tolist(),
            default=['cluster', 'num_atoms', 'relative_energy', 'energy_per_atom', 
                    'crystal_system', 'mean_bond_length', 'mean_coordination'][:7]
        )
        
        st.dataframe(filtered_df[display_columns], use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="carbon24_filtered_data.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.markdown("### Summary Statistics")
        
        st.dataframe(filtered_df[numeric_features].describe(), use_container_width=True)

except FileNotFoundError as e:
    st.error(f"""
    ⚠️ **Error: Required files not found!**
    
    Please run the analysis script first:
    ```
    python carbon24-stability-analysis.py
    ```
    
    Error details: {str(e)}
    """)
except Exception as e:
    st.error(f"""
    ⚠️ **An error occurred:**
    
    {str(e)}
    
    Please check that all required files are present and properly formatted.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Carbon-24 Stability Analysis Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
