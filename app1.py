import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import os
import time
import json
from datetime import datetime
import psutil
import glob
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="MLnextstep Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stAlert {
        background-color: #e8f4fd;
        border-left: 5px solid #2196F3;
    }
    h1, h2, h3 {
        color: #262730;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 1em;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'last_run_time' not in st.session_state:
    st.session_state.last_run_time = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'real_data_loaded' not in st.session_state:
    st.session_state.real_data_loaded = False


def load_real_pipeline_data():
    """Load real data from ML pipeline outputs"""
    try:
        # Check if pipeline outputs exist
        combined_output_path = Path("oplogs/model_outputs/combined_output.csv")
        performance_metrics_path = Path("oplogs/model_outputs/performance/performance_metrics.csv")
        
        if not combined_output_path.exists() or not performance_metrics_path.exists():
            st.warning("Pipeline outputs not found. Please run the pipeline first.")
            return None
        
        # Load the combined output data
        df_output = pd.read_csv(combined_output_path)
        
        # Load performance metrics
        df_performance = pd.read_csv(performance_metrics_path)
        
        # Merge the dataframes on window_start
        df_merged = pd.merge(df_output, df_performance, on='window_start', how='left', suffixes=('', '_perf'))
        
        # Clean up duplicate columns
        duplicate_cols = ['dbscan_cluster_perf', 'dbscan_silhouette_score_perf', 
                         'isolation_forest_score_perf', 'isolation_forest_anomaly_perf',
                         'isolation_forest_top_anomaly_perf', 'pca_component_1_perf', 
                         'pca_component_2_perf', 'processing_timestamp_perf']
        df_merged = df_merged.drop(columns=[col for col in duplicate_cols if col in df_merged.columns])
        
        # Convert timestamp columns
        if 'window_start' in df_merged.columns:
            df_merged['window_start'] = pd.to_datetime(df_merged['window_start'])
        if 'processing_timestamp' in df_merged.columns:
            df_merged['processing_timestamp'] = pd.to_datetime(df_merged['processing_timestamp'])
        
        st.session_state.real_data_loaded = True
        return df_merged
        
    except Exception as e:
        st.error(f"Error loading pipeline data: {e}")
        return None

def run_real_pipeline():
    """Run the actual ML pipeline"""
    st.session_state.pipeline_running = True
    st.session_state.current_step = 0
    
    # Run the main pipeline
    steps = [
        "Step 1: Parsing logs...",
        "Step 2: Preprocessing/cleaning...", 
        "Step 3: Feature engineering...",
        "Step 4: Feature scaling...",
        "Step 5: Training ML models...",
        "Step 6: Validation..."
    ]
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    try:
        # Run the pipeline using subprocess
        for i, step in enumerate(steps):
            st.session_state.current_step = i + 1
            progress_bar.progress((i + 1) / len(steps))
            status_text.text(step)
            time.sleep(2)  # Simulate processing time
        
        # Actually run the pipeline
        result = subprocess.run(["python", "main_pipeline.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            st.success("Pipeline completed successfully!")
            st.session_state.last_run_time = datetime.now()
        else:
            st.error(f"Pipeline failed with error:\n{result.stderr}")
            
    except Exception as e:
        st.error(f"Error running pipeline: {e}")
    finally:
        st.session_state.pipeline_running = False
        progress_bar.empty()
        status_text.empty()

def get_system_metrics():
    """Get system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu': cpu_percent,
        'memory_percent': memory.percent,
        'memory_used_gb': round(memory.used / (1024**3), 2),
        'memory_total_gb': round(memory.total / (1024**3), 2),
        'disk_percent': disk.percent,
        'disk_used_gb': round(disk.used / (1024**3), 2),
        'disk_total_gb': round(disk.total / (1024**3), 2)
    }

def main():
    st.title("📊 MLnextstep Pipeline Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        if st.button("🚀 Run Full Pipeline", disabled=st.session_state.pipeline_running):
            run_real_pipeline()
        
        if st.session_state.pipeline_running:
            st.info(f"Running step {st.session_state.current_step}/6...")
        
        if st.session_state.last_run_time:
            st.success(f"Last run: {st.session_state.last_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # Visualization settings
        st.subheader("Chart Settings")
        chart_height = st.slider("Chart Height", 300, 800, 500)
        show_grid = st.checkbox("Show Grid", True)
        
        # Data settings
        st.subheader("Data Settings")
        sample_size = st.slider("Sample Size", 100, 10000, 1000)
        
        st.markdown("---")
        st.header("🖥️ System Metrics")
        metrics = get_system_metrics()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("CPU", f"{metrics['cpu']:.1f}%")
        col2.metric("Memory", f"{metrics['memory_percent']:.1f}%")
        col3.metric("Disk", f"{metrics['disk_percent']:.1f}%")
    
    # Load data
    data = load_real_pipeline_data()
    
    if data is None:
        st.warning("No data available. Please run the pipeline first.")
        return
    
    # Create a copy for display purposes with timestamp columns converted to strings
    display_data = data.copy()
    timestamp_cols = ['window_start', 'processing_timestamp']
    for col in timestamp_cols:
        if col in display_data.columns:
            display_data[col] = display_data[col].astype(str)
    
    # Key metrics
    st.header("🔑 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">{:,}</div>'.format(len(data)), unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Windows</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if 'isolation_forest_anomaly' in data.columns:
            anomaly_count = data['isolation_forest_anomaly'].sum()
        else:
            anomaly_count = 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">{:,}</div>'.format(int(anomaly_count)), unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Anomalies</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if 'isolation_forest_score' in data.columns:
            avg_score = data['isolation_forest_score'].mean()
        else:
            avg_score = 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">{:.2f}</div>'.format(avg_score), unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg Anomaly Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        if 'dbscan_cluster' in data.columns:
            unique_clusters = data['dbscan_cluster'].nunique()
        else:
            unique_clusters = 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">{}</div>'.format(unique_clusters), unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Clusters</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        if 'event_count_scaled' in data.columns:
            avg_events = data['event_count_scaled'].mean()
        else:
            avg_events = 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">{:.1f}</div>'.format(avg_events), unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg Event Count</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Anomaly Detection", "🧩 Clustering", "📊 Feature Analysis", "📋 Data Overview"])
    
    with tab1:
        st.header("Anomaly Detection Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Anomaly score timeline
            if 'window_start' in data.columns and 'isolation_forest_score' in data.columns:
                fig_anomaly_timeline = px.line(
                    data, 
                    x='window_start', 
                    y='isolation_forest_score',
                    title="Anomaly Scores Over Time",
                    labels={'isolation_forest_score': 'Anomaly Score', 'window_start': 'Time'}
                )
                
                # Add threshold line (assuming threshold around 0 for isolation forest)
                fig_anomaly_timeline.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Anomaly Threshold",
                    annotation_position="top left"
                )
                
                # Highlight anomalies
                if 'isolation_forest_anomaly' in data.columns:
                    anomalies = data[data['isolation_forest_anomaly'] == 1]
                    fig_anomaly_timeline.add_scatter(
                        x=anomalies['window_start'],
                        y=anomalies['isolation_forest_score'],
                        mode='markers',
                        marker=dict(color='red', size=8),
                        name='Detected Anomalies'
                    )
                
                fig_anomaly_timeline.update_layout(
                    height=chart_height,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                if not show_grid:
                    fig_anomaly_timeline.update_xaxes(showgrid=False)
                    fig_anomaly_timeline.update_yaxes(showgrid=False)
                
                st.plotly_chart(fig_anomaly_timeline, width='stretch')
            else:
                st.warning("Anomaly detection data not available")
        
        with col2:
            # Anomaly distribution
            if 'isolation_forest_score' in data.columns:
                fig_anomaly_dist = px.histogram(
                    data,
                    x='isolation_forest_score',
                    nbins=50,
                    title="Anomaly Score Distribution",
                    labels={'isolation_forest_score': 'Anomaly Score', 'count': 'Frequency'}
                )
                
                fig_anomaly_dist.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Threshold"
                )
                
                fig_anomaly_dist.update_layout(height=chart_height)
                st.plotly_chart(fig_anomaly_dist, width='stretch')
                
                # Anomaly statistics
                st.subheader("Anomaly Statistics")
                if 'isolation_forest_anomaly' in data.columns:
                    anomaly_count = data['isolation_forest_anomaly'].sum()
                    st.metric("Total Anomalies", int(anomaly_count))
                    st.metric("Anomaly Rate", f"{(anomaly_count/len(data))*100:.2f}%")
                if 'isolation_forest_score' in data.columns:
                    st.metric("Max Anomaly Score", f"{data['isolation_forest_score'].max():.2f}")
            else:
                st.warning("Anomaly statistics not available")
    
    with tab2:
        st.header("Clustering Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'dbscan_cluster' in data.columns and 'pca_component_1' in data.columns and 'pca_component_2' in data.columns:
                # PCA scatter plot
                fig_clusters = px.scatter(
                    data,
                    x='pca_component_1',
                    y='pca_component_2',
                    color='dbscan_cluster',
                    title="DBSCAN Clusters (PCA Reduced)",
                    labels={'pca_component_1': 'PCA Component 1', 'pca_component_2': 'PCA Component 2'}
                )
                fig_clusters.update_layout(height=chart_height)
                st.plotly_chart(fig_clusters, width='stretch')
            else:
                st.warning("Clustering data not available")
        
        with col2:
            if 'dbscan_cluster' in data.columns:
                # Cluster size distribution
                cluster_sizes = data['dbscan_cluster'].value_counts()
                fig_cluster_sizes = px.bar(
                    x=cluster_sizes.index.astype(str),
                    y=cluster_sizes.values,
                    title="Cluster Sizes",
                    labels={'x': 'Cluster ID', 'y': 'Number of Points'}
                )
                fig_cluster_sizes.update_layout(height=chart_height)
                st.plotly_chart(fig_cluster_sizes, width='stretch')
                
                # Cluster statistics
                st.subheader("Cluster Statistics")
                st.metric("Number of Clusters", cluster_sizes[cluster_sizes.index != -1].count())
                st.metric("Noise Points", cluster_sizes.get(-1, 0))
            else:
                st.warning("Cluster statistics not available")
    
    with tab3:
        st.header("Feature Analysis")
        
        # Get numeric columns for feature analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['dbscan_cluster', 'isolation_forest_score', 'isolation_forest_anomaly', 
                                                                 'pca_component_1', 'pca_component_2', 'dbscan_silhouette_score']]
        
        if feature_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature correlation heatmap
                corr_matrix = data[feature_cols[:10]].corr()  # Limit to first 10 features
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Heatmap",
                    aspect="auto"
                )
                fig_corr.update_layout(height=chart_height)
                st.plotly_chart(fig_corr, width='stretch')
            
            with col2:
                # Feature distribution
                selected_feature = st.selectbox("Select Feature", feature_cols)
                fig_feature_dist = px.histogram(
                    data,
                    x=selected_feature,
                    title=f"Distribution of {selected_feature}",
                    nbins=50
                )
                fig_feature_dist.update_layout(height=chart_height)
                st.plotly_chart(fig_feature_dist, width='stretch')
        else:
            st.warning("No feature data available for analysis")
    
    with tab4:
        st.header("Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Summary")
            st.dataframe(display_data.describe(), width='stretch')
        
        with col2:
            st.subheader("Raw Data Preview")
            st.dataframe(display_data.head(10), width='stretch')
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': display_data.columns,
            'Data Type': display_data.dtypes.astype(str),  # Convert dtypes to strings
            'Non-Null Count': display_data.count().astype(str),  # Convert counts to strings
            'Null Count': display_data.isnull().sum().astype(str)  # Convert null counts to strings
        })
        st.dataframe(col_info, width='stretch')


if __name__ == "__main__":
    main()