"""
Script sửa lỗi tên cột PCA trong dashboard
"""

# Đọc file dashboard
with open('carbon24_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Tìm và sửa các phần GMM PCA visualization
# GMM dùng PCA1, PCA2 (chữ hoa)
gmm_viz_old = """            if 'pca1' in gmm_df.columns and 'pca2' in gmm_df.columns:
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
                st.warning("Chưa có PCA components trong kết quả GMM")"""

gmm_viz_new = """            if 'PCA1' in gmm_df.columns and 'PCA2' in gmm_df.columns:
                color_by = st.radio("Color by:", ["Cluster", "Max Probability", "Relative Energy"], 
                                   horizontal=True, key="gmm_color")
                
                if color_by == "Cluster":
                    fig = px.scatter(gmm_df, x='PCA1', y='PCA2', color='gmm_cluster',
                                   title="GMM PCA 2D: Clusters",
                                   labels={'PCA1': 'PC1', 'PCA2': 'PC2'},
                                   color_continuous_scale='viridis')
                elif color_by == "Max Probability":
                    fig = px.scatter(gmm_df, x='PCA1', y='PCA2', color='max_probability',
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
                st.warning("Chưa có PCA components trong kết quả GMM")"""

content = content.replace(gmm_viz_old, gmm_viz_new)

# Hierarchical dùng PC1, PC2 (chữ hoa)
hier_viz_old = """            if 'pca1' in hierarchical_df.columns and 'pca2' in hierarchical_df.columns:
                color_by = st.radio("Color by:", ["Cluster", "Relative Energy"], 
                                   horizontal=True, key="hier_color")
                
                if color_by == "Cluster":
                    fig = px.scatter(hierarchical_df, x='pca1', y='pca2', 
                                   color='cluster_hierarchical',
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
                st.warning("Chưa có PCA components trong kết quả Hierarchical")"""

hier_viz_new = """            if 'PC1' in hierarchical_df.columns and 'PC2' in hierarchical_df.columns:
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
                st.warning("Chưa có PCA components trong kết quả Hierarchical")"""

content = content.replace(hier_viz_old, hier_viz_new)

# Ghi lại file
with open('carbon24_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Đã sửa các lỗi tên cột PCA:")
print("   1. GMM: pca1, pca2 -> PCA1, PCA2")
print("   2. Hierarchical: pca1, pca2 -> PC1, PC2")
print("   3. HDBSCAN: pca1, pca2 (giữ nguyên - đúng rồi)")
print("\n🚀 Dashboard đã sẵn sàng!")
