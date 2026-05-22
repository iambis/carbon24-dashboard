"""
Script để thêm các trang phân cụm GMM, Hierarchical và HDBSCAN vào dashboard
"""

# Đọc file dashboard hiện tại
with open('carbon24_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Tìm vị trí chèn (sau trang K-means, trước trang So sánh thuật toán)
marker = "# ============================================================================\n# PAGE: SO SÁNH THUẬT TOÁN\n# ============================================================================"

# Code cho các trang mới
new_pages = '''
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
        col3.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_index']:.2f}")
        col4.metric("Số Clusters", gmm_report['n_clusters'])
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs([" Cluster Overview", " Visualization", " Uncertainty Analysis", " Cluster Profiles"])
        
        with tab1:
            st.markdown("#### Phân Bố Clusters")
            
            cluster_counts = gmm_df['gmm_cluster'].value_counts().sort_index()
            
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
            **Optimal number of clusters:** {gmm_report['n_clusters']}
            
            - **AIC (Akaike Information Criterion):** {gmm_report['aic']:.2f}
            - **BIC (Bayesian Information Criterion):** {gmm_report['bic']:.2f}
            
            Số clusters được chọn dựa trên BIC thấp nhất.
            """)
        
        with tab2:
            st.markdown("#### PCA Visualization")
            
            if 'pca1' in gmm_df.columns and 'pca2' in gmm_df.columns:
                color_by = st.radio("Color by:", ["Cluster", "Max Probability", "Relative Energy"], 
                                   horizontal=True, key="gmm_color")
                
                if color_by == "Cluster":
                    fig = px.scatter(gmm_df, x='pca1', y='pca2', color='gmm_cluster',
                                   title="GMM PCA 2D: Clusters",
                                   labels={'pca1': 'PC1', 'pca2': 'PC2'},
                                   color_continuous_scale='viridis')
                elif color_by == "Max Probability":
                    fig = px.scatter(gmm_df, x='pca1', y='pca2', color='max_probability',
                                   title="GMM PCA 2D: Assignment Probability",
                                   labels={'pca1': 'PC1', 'pca2': 'PC2'},
                                   color_continuous_scale='RdYlGn')
                else:
                    if 'relative_energy' in gmm_df.columns:
                        fig = px.scatter(gmm_df, x='pca1', y='pca2', color='relative_energy',
                                       title="GMM PCA 2D: Relative Energy",
                                       labels={'pca1': 'PC1', 'pca2': 'PC2'},
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
                fig = px.histogram(gmm_df, x='max_probability', nbins=50,
                                 title="Distribution of Max Probabilities")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot by cluster
                fig = px.box(gmm_df, x='gmm_cluster', y='max_probability',
                           title="Max Probability by Cluster")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Uncertain samples
            threshold = st.slider("Ngưỡng xác suất (samples dưới ngưỡng = uncertain):", 
                                 0.5, 1.0, 0.7, 0.05)
            uncertain_samples = gmm_df[gmm_df['max_probability'] < threshold]
            
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
                    
                    fig = px.box(gmm_df, x='gmm_cluster', y='relative_energy',
                               title="Relative Energy Distribution by GMM Cluster")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    energy_stats = gmm_df.groupby('gmm_cluster')['relative_energy'].agg(['mean', 'std', 'min', 'max'])
                    st.dataframe(energy_stats, use_container_width=True)

# ============================================================================
# PAGE: PHÂN CỤM HIERARCHICAL
# ============================================================================
elif page == " Phân cụm Hierarchical":
    st.markdown('<div class="sub-header">🌳 Phân Cụm Hierarchical (Agglomerative)</div>', unsafe_allow_html=True)
    
    if hierarchical_df is None:
        st.warning("⚠️ Chưa có kết quả phân cụm Hierarchical. Vui lòng chạy notebook Hierarchical trước.")
        st.info("📝 Chạy notebook hoặc script tạo kết quả Hierarchical clustering")
    else:
        # Hierarchical metrics
        st.markdown("####  Hierarchical Clustering Info")
        
        col1, col2, col3 = st.columns(3)
        
        n_clusters = hierarchical_df['hierarchical_cluster'].nunique()
        col1.metric("Số Clusters", n_clusters)
        col2.metric("Tổng Samples", f"{len(hierarchical_df):,}")
        col3.metric("Linkage Method", "Ward")
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs([" Cluster Overview", " Visualization", " Cluster Interpretation"])
        
        with tab1:
            st.markdown("#### Phân Bố Clusters")
            
            cluster_counts = hierarchical_df['hierarchical_cluster'].value_counts().sort_index()
            
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
            
            if 'pca1' in hierarchical_df.columns and 'pca2' in hierarchical_df.columns:
                color_by = st.radio("Color by:", ["Cluster", "Relative Energy"], 
                                   horizontal=True, key="hier_color")
                
                if color_by == "Cluster":
                    fig = px.scatter(hierarchical_df, x='pca1', y='pca2', 
                                   color='hierarchical_cluster',
                                   title="Hierarchical PCA 2D: Clusters",
                                   labels={'pca1': 'PC1', 'pca2': 'PC2'},
                                   color_continuous_scale='viridis')
                else:
                    if 'relative_energy' in hierarchical_df.columns:
                        fig = px.scatter(hierarchical_df, x='pca1', y='pca2', 
                                       color='relative_energy',
                                       title="Hierarchical PCA 2D: Relative Energy",
                                       labels={'pca1': 'PC1', 'pca2': 'PC2'},
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
                
                fig = px.box(hierarchical_df, x='hierarchical_cluster', y='relative_energy',
                           title="Relative Energy Distribution by Hierarchical Cluster")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                energy_stats = hierarchical_df.groupby('hierarchical_cluster')['relative_energy'].agg(['mean', 'std', 'min', 'max', 'count'])
                st.dataframe(energy_stats, use_container_width=True)
            
            # Cluster analysis
            st.markdown("#### Phân Tích Theo Cluster")
            
            selected_cluster = st.selectbox("Chọn cluster:", 
                                           sorted(hierarchical_df['hierarchical_cluster'].unique()),
                                           key="hier_cluster_select")
            
            cluster_data = hierarchical_df[hierarchical_df['hierarchical_cluster'] == selected_cluster]
            
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
    st.markdown('<div class="sub-header">🔍 Phân Cụm HDBSCAN (Density-Based)</div>', unsafe_allow_html=True)
    
    if hdbscan_df is None:
        st.warning("⚠️ Chưa có kết quả phân cụm HDBSCAN. Vui lòng chạy notebook HDBSCAN trước.")
        st.info("📝 Chạy: `HDBSCAN.ipynb`")
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
                    if 'membership_probability' in hdbscan_df.columns:
                        fig = px.scatter(viz_df, x='pca1', y='pca2', 
                                       color='membership_probability',
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
                profile_display = hdbscan_cluster_profile[hdbscan_cluster_profile['Cluster'] != -1]
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
                if 'membership_probability' in cluster_data.columns:
                    st.markdown("##### Membership Probability Distribution")
                    fig = px.histogram(cluster_data, x='membership_probability', nbins=30,
                                     title=f"Membership Probability - Cluster {selected_cluster}")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

'''

# Chèn code mới vào đúng vị trí
if marker in content:
    content = content.replace(marker, new_pages + marker)
    
    # Ghi lại file
    with open('carbon24_dashboard.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Đã thêm thành công 3 trang mới vào dashboard:")
    print("   - 🎲 Phân cụm GMM")
    print("   - 🌳 Phân cụm Hierarchical")
    print("   - 🔍 Phân cụm HDBSCAN")
    print("\n🚀 Chạy dashboard: streamlit run carbon24_dashboard.py")
else:
    print("❌ Không tìm thấy vị trí chèn trong file dashboard")
