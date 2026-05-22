"""
Script để sửa dashboard - di chuyển anomaly detection functions lên đúng vị trí
"""

print("="*80)
print("FIXING DASHBOARD - ANOMALY DETECTION")
print("="*80)

# Read dashboard
with open('carbon24_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the position to insert (after load_hdbscan_results function)
insert_marker = "def load_hdbscan_results():"
insert_pos = content.find(insert_marker)

if insert_pos == -1:
    print("❌ Không tìm thấy marker để insert")
    exit(1)

# Find the end of load_hdbscan_results function
# Look for the next function definition or page handling
next_func_pos = content.find("\n\n# =====", insert_pos + 100)
if next_func_pos == -1:
    next_func_pos = content.find("\n\nif page ==", insert_pos + 100)

insert_pos = next_func_pos

print(f"✅ Tìm thấy vị trí insert tại: {insert_pos}")

# The anomaly detection functions to insert
anomaly_functions = '''

# ============================================================================
# ANOMALY DETECTION DATA LOADING
# ============================================================================

@st.cache_data
def load_anomaly_detection_results():
    """Load anomaly detection results"""
    try:
        results_df = pd.read_csv('carbon24_anomaly_detection/anomaly_detection_results.csv')
        summary_df = pd.read_csv('carbon24_anomaly_detection/anomaly_summary.csv')
        comparison_df = pd.read_csv('carbon24_anomaly_detection/anomaly_method_comparison.csv')
        details_df = pd.read_csv('carbon24_anomaly_detection/anomaly_details.csv')
        
        return results_df, summary_df, comparison_df, details_df
    except Exception as e:
        st.error(f"Lỗi khi load anomaly detection results: {e}")
        return None, None, None, None


def render_anomaly_detection_tab():
    """Render Anomaly Detection analysis tab"""
    
    st.markdown('<div class="sub-header">🔍 Phát hiện Dị biệt (Anomaly Detection)</div>', unsafe_allow_html=True)
    
    # Load data
    results_df, summary_df, comparison_df, details_df = load_anomaly_detection_results()
    
    if results_df is None:
        st.warning("⚠️ Chưa có dữ liệu Anomaly Detection. Vui lòng chạy module trước!")
        st.code("python carbon24_anomaly_detection.py")
        return
    
    # Overview metrics
    st.markdown("### 📊 Tổng quan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tổng số mẫu",
            f"{len(results_df):,}",
            help="Tổng số cấu trúc Carbon-24"
        )
    
    with col2:
        n_consensus = results_df['is_anomaly_consensus'].sum()
        ratio_consensus = results_df['is_anomaly_consensus'].mean()
        st.metric(
            "Consensus Anomalies",
            f"{n_consensus:,}",
            f"{ratio_consensus:.2%}",
            help="Ít nhất 2/3 phương pháp đồng ý"
        )
    
    with col3:
        n_all = results_df['is_anomaly_all'].sum()
        ratio_all = results_df['is_anomaly_all'].mean()
        st.metric(
            "High Confidence",
            f"{n_all:,}",
            f"{ratio_all:.2%}",
            help="Cả 3 phương pháp đồng ý"
        )
    
    with col4:
        n_any = results_df['is_anomaly_any'].sum()
        ratio_any = results_df['is_anomaly_any'].mean()
        st.metric(
            "Any Method",
            f"{n_any:,}",
            f"{ratio_any:.2%}",
            help="Ít nhất 1 phương pháp phát hiện"
        )
    
    st.markdown("---")
    
    # Method comparison
    st.markdown("### 🔬 So sánh các phương pháp")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Bar chart comparison
        fig = go.Figure()
        
        methods = comparison_df['method'].values
        anomaly_ratios = comparison_df['anomaly_ratio'].values * 100
        n_anomalies = comparison_df['n_anomalies'].values
        
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
        
        fig.add_trace(go.Bar(
            y=methods,
            x=anomaly_ratios,
            orientation='h',
            text=[f'{ratio:.1f}% ({n:,})' for ratio, n in zip(anomaly_ratios, n_anomalies)],
            textposition='outside',
            marker=dict(color=colors[:len(methods)]),
            hovertemplate='<b>%{y}</b><br>Ratio: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Tỷ lệ Anomaly theo từng phương pháp",
            xaxis_title="Anomaly Ratio (%)",
            yaxis_title="",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Vote distribution
        vote_counts = results_df['anomaly_vote_count'].value_counts().sort_index()
        
        fig = go.Figure()
        
        colors_vote = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        
        fig.add_trace(go.Bar(
            x=vote_counts.index,
            y=vote_counts.values,
            text=[f'{v:,}<br>({v/len(results_df)*100:.1f}%)' for v in vote_counts.values],
            textposition='outside',
            marker=dict(color=colors_vote),
            hovertemplate='<b>%{x} methods</b><br>Count: %{y:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Phân bố số phương pháp đồng ý",
            xaxis_title="Số phương pháp phát hiện",
            yaxis_title="Số mẫu",
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Spatial distribution (PCA)
    st.markdown("### 🗺️ Phân bố không gian (PCA)")
    
    # Find PCA columns
    pca_cols = [c for c in results_df.columns if c.lower().startswith('pca')]
    
    if len(pca_cols) >= 2:
        method_options = {
            'HDBSCAN Noise': 'is_hdbscan_noise',
            'Low Probability': 'is_low_probability',
            'Isolation Forest': 'is_isolation_forest_anomaly',
            'Consensus (≥2)': 'is_anomaly_consensus',
            'All 3 methods': 'is_anomaly_all'
        }
        
        selected_method = st.selectbox(
            "Chọn phương pháp:",
            list(method_options.keys()),
            index=3  # Default to Consensus
        )
        
        method_col = method_options[selected_method]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Normal points
        normal_mask = results_df[method_col] == 0
        fig.add_trace(go.Scatter(
            x=results_df.loc[normal_mask, pca_cols[0]],
            y=results_df.loc[normal_mask, pca_cols[1]],
            mode='markers',
            name='Normal',
            marker=dict(
                size=5,
                color='lightblue',
                opacity=0.3,
                line=dict(width=0)
            ),
            hovertemplate='<b>Normal</b><br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>'
        ))
        
        # Anomaly points
        anomaly_mask = results_df[method_col] == 1
        fig.add_trace(go.Scatter(
            x=results_df.loc[anomaly_mask, pca_cols[0]],
            y=results_df.loc[anomaly_mask, pca_cols[1]],
            mode='markers',
            name='Anomaly',
            marker=dict(
                size=8,
                color='red',
                opacity=0.7,
                line=dict(width=1, color='darkred')
            ),
            hovertemplate='<b>Anomaly</b><br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<extra></extra>'
        ))
        
        n_anomalies = anomaly_mask.sum()
        anomaly_ratio = anomaly_mask.mean()
        
        fig.update_layout(
            title=f"{selected_method}: {n_anomalies:,} anomalies ({anomaly_ratio:.2%})",
            xaxis_title=pca_cols[0],
            yaxis_title=pca_cols[1],
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Không tìm thấy cột PCA trong dữ liệu")
    
    st.markdown("---")
    
    # Energy analysis
    st.markdown("### ⚡ Phân tích Năng lượng")
    
    if 'relative_energy' in results_df.columns:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Energy comparison table
            st.markdown("#### So sánh năng lượng trung bình")
            
            energy_comparison = []
            for method in ['is_hdbscan_noise', 'is_low_probability', 'is_isolation_forest_anomaly', 'is_anomaly_consensus']:
                if method in results_df.columns:
                    anomaly_energy = results_df[results_df[method] == 1]['relative_energy'].mean()
                    normal_energy = results_df[results_df[method] == 0]['relative_energy'].mean()
                    diff = anomaly_energy - normal_energy
                    
                    energy_comparison.append({
                        'Phương pháp': method.replace('is_', '').replace('_', ' ').title(),
                        'Anomaly (eV/atom)': f'{anomaly_energy:.4f}',
                        'Normal (eV/atom)': f'{normal_energy:.4f}',
                        'Chênh lệch': f'{diff:+.4f}'
                    })
            
            energy_df = pd.DataFrame(energy_comparison)
            st.dataframe(energy_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Energy distribution
            method_for_dist = st.selectbox(
                "Chọn phương pháp cho histogram:",
                ['is_anomaly_consensus', 'is_hdbscan_noise', 'is_low_probability', 'is_isolation_forest_anomaly'],
                format_func=lambda x: x.replace('is_', '').replace('_', ' ').title()
            )
            
            fig = go.Figure()
            
            normal_energy = results_df[results_df[method_for_dist] == 0]['relative_energy']
            anomaly_energy = results_df[results_df[method_for_dist] == 1]['relative_energy']
            
            fig.add_trace(go.Histogram(
                x=normal_energy,
                name='Normal',
                opacity=0.5,
                marker=dict(color='blue'),
                nbinsx=50
            ))
            
            fig.add_trace(go.Histogram(
                x=anomaly_energy,
                name='Anomaly',
                opacity=0.5,
                marker=dict(color='red'),
                nbinsx=50
            ))
            
            fig.add_vline(
                x=normal_energy.mean(),
                line_dash="dash",
                line_color="blue",
                annotation_text=f"Normal mean: {normal_energy.mean():.3f}"
            )
            
            fig.add_vline(
                x=anomaly_energy.mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Anomaly mean: {anomaly_energy.mean():.3f}"
            )
            
            fig.update_layout(
                title="Phân bố Relative Energy",
                xaxis_title="Relative Energy (eV/atom)",
                yaxis_title="Count",
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Anomaly details
    st.markdown("### 📋 Chi tiết Consensus Anomalies")
    
    if details_df is not None and len(details_df) > 0:
        st.markdown(f"**Tổng số: {len(details_df):,} anomalies**")
        
        # Filter options
        col1, col2 = st.columns([1, 1])
        
        with col1:
            vote_filter = st.multiselect(
                "Lọc theo vote count:",
                options=[2, 3],
                default=[2, 3]
            )
        
        with col2:
            n_display = st.slider(
                "Số lượng hiển thị:",
                min_value=10,
                max_value=min(100, len(details_df)),
                value=20,
                step=10
            )
        
        # Filter and display
        filtered_details = details_df[details_df['anomaly_vote_count'].isin(vote_filter)]
        
        st.dataframe(
            filtered_details.head(n_display),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = filtered_details.to_csv(index=False)
        st.download_button(
            label="📥 Download Anomaly Details (CSV)",
            data=csv,
            file_name="consensus_anomalies.csv",
            mime="text/csv"
        )
    else:
        st.info("Không có consensus anomalies")
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Khuyến nghị sử dụng")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Phân tích sâu**
        - Sử dụng: `is_anomaly_all`
        - Cả 3 phương pháp đồng ý
        - High precision
        - Các điểm thực sự bất thường
        """)
    
    with col2:
        st.markdown("""
        **⚖️ Lọc dữ liệu**
        - Sử dụng: `is_anomaly_consensus`
        - Ít nhất 2/3 phương pháp
        - Balanced precision/recall
        - Phù hợp cho hầu hết ứng dụng
        """)
    
    with col3:
        st.markdown("""
        **🔍 Khám phá**
        - Sử dụng: `is_anomaly_any`
        - Ít nhất 1 phương pháp
        - High recall
        - Phát hiện nhiều điểm tiềm năng
        """)
'''

# Check if functions already exist at the end
if "# ANOMALY DETECTION DATA LOADING" in content[insert_pos:]:
    print("⚠️  Functions đã tồn tại ở cuối file, sẽ xóa và di chuyển lên")
    
    # Find and remove the old functions
    old_start = content.find("# ANOMALY DETECTION DATA LOADING", insert_pos)
    # Find the end (next major section or end of file)
    old_end = len(content)
    
    # Remove old section
    content = content[:old_start] + content[old_end:]
    print("✅ Đã xóa phần cũ")

# Insert at the correct position
new_content = content[:insert_pos] + anomaly_functions + content[insert_pos:]

# Write back
with open('carbon24_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Đã di chuyển anomaly detection functions lên đúng vị trí")
print("\n" + "="*80)
print("✅ HOÀN TẤT")
print("="*80)
print("\n📝 Bây giờ bạn có thể:")
print("   1. Chạy module anomaly detection: python carbon24_anomaly_detection.py")
print("   2. Chạy dashboard: streamlit run carbon24_dashboard.py")
print("   3. Chọn page 'Phát hiện dị biệt' trong sidebar")
