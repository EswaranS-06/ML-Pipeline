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
from datetime import datetime, timedelta
import psutil
import glob
from pathlib import Path
import re
import shutil
from collections import deque
import sys  # Added missing import

# Set page config
st.set_page_config(
    page_title="SIEM Dashboard - MLnextstep",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SIEM-like styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #fafafa;
    }
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    .stAlert {
        background-color: #2d2d2d;
        border-left: 5px solid #ff4b4b;
        color: #fafafa;
    }
    .alert-high {
        background-color: #ff4b4b !important;
        color: white !important;
        border-left: 5px solid #ff0000 !important;
    }
    .alert-medium {
        background-color: #ffa500 !important;
        color: white !important;
        border-left: 5px solid #ff8c00 !important;
    }
    .alert-low {
        background-color: #4CAF50 !important;
        color: white !important;
        border-left: 5px solid #45a049 !important;
    }
    .metric-card {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid #444;
    }
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 1em;
        color: #aaa;
    }
    .threat-card {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #ff4b4b;
    }
    .log-entry {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-family: monospace;
        font-size: 12px;
        border-left: 3px solid #444;
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
if 'alerts' not in st.session_state:
    st.session_state.alerts = deque(maxlen=100)
if 'threats' not in st.session_state:
    st.session_state.threats = []
if 'selected_log_file' not in st.session_state:
    st.session_state.selected_log_file = None

def add_alert(level, message, source=None):
    """Add a new alert to the alert system"""
    alert = {
        'timestamp': datetime.now(),
        'level': level,
        'message': message,
        'source': source,
        'acknowledged': False
    }
    st.session_state.alerts.append(alert)
    
    # If high severity, also add to threats
    if level == 'high':
        st.session_state.threats.append({
            'timestamp': datetime.now(),
            'message': message,
            'source': source,
            'status': 'new',
            'escalation_level': 1
        })

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

def get_available_log_files():
    """Get list of available log files"""
    log_files = []
    
    # Check logs directory
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files.extend([f"logs/{f.name}" for f in logs_dir.glob("*.log") if f.is_file()])
    
    # Check logs1 directory
    logs1_dir = Path("logs1")
    if logs1_dir.exists():
        log_files.extend([f"logs1/{f.name}" for f in logs1_dir.glob("*.log") if f.is_file()])
    
    return sorted(log_files)

def analyze_log_patterns(log_file_path):
    """Analyze log patterns for suspicious activities"""
    suspicious_patterns = [
        (r'(?i)(error|fail|failed|exception|critical|fatal)', 'medium'),
        (r'(?i)(unauthorized|access denied|permission denied|forbidden)', 'high'),
        (r'(?i)(attack|malicious|virus|malware|trojan|ransomware)', 'high'),
        (r'(?i)(brute force|password guess|login attempt)', 'medium'),
        (r'(?i)(sql injection|xss|cross site)', 'high'),
        (r'(?i)(ddos|denial of service)', 'high'),
        (r'(?i)(port scan|nmap|scanning)', 'medium'),
        (r'(?i)(rootkit|backdoor|exploit)', 'high'),
        (r'(?i)(firewall drop|blocked|rejected)', 'low'),
        (r'(?i)(timeout|connection reset)', 'low')
    ]
    
    alerts = []
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines[:1000]):  # Check first 1000 lines
            for pattern, severity in suspicious_patterns:
                if re.search(pattern, line):
                    alerts.append({
                        'line_number': i + 1,
                        'severity': severity,
                        'pattern': pattern,
                        'log_line': line.strip(),
                        'timestamp': datetime.now()
                    })
                    
    except Exception as e:
        add_alert('medium', f"Error analyzing log file {log_file_path}: {e}", "Log Analyzer")
    
    return alerts

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
            time.sleep(1)  # Simulate processing time
        
        # Actually run the pipeline
        result = subprocess.run(["python", "main_pipeline.py"], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            st.success("Pipeline completed successfully!")
            st.session_state.last_run_time = datetime.now()
            add_alert('low', 'ML Pipeline completed successfully', 'ML Pipeline')
        else:
            st.error(f"Pipeline failed with error:\n{result.stderr}")
            add_alert('high', f'ML Pipeline failed: {result.stderr}', 'ML Pipeline')
            
    except Exception as e:
        st.error(f"Error running pipeline: {e}")
        add_alert('high', f'Pipeline execution error: {e}', 'ML Pipeline')
    finally:
        st.session_state.pipeline_running = False
        progress_bar.empty()
        status_text.empty()

def monitor_logs_realtime(log_file_path, max_lines=100):
    """Monitor log files in real-time for new entries"""
    try:
        if not os.path.exists(log_file_path):
            return []
        
        # Get the last few lines
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        return lines[-max_lines:] if len(lines) > max_lines else lines
        
    except Exception as e:
        add_alert('medium', f"Error monitoring log file {log_file_path}: {e}", "Log Monitor")
        return []

def get_threat_intelligence():
    """Get threat intelligence data (mock function for demonstration)"""
    # This would typically connect to threat intelligence APIs
    # For now, return mock data
    return {
        'known_malicious_ips': ['192.168.1.100', '10.0.0.5', '172.16.0.23'],
        'suspicious_domains': ['malicious-site.com', 'phishing-attempt.net'],
        'recent_attacks': ['SQL Injection', 'XSS Attempt', 'Brute Force'],
        'last_update': datetime.now()
    }

def check_for_known_threats(log_line):
    """Check if log line contains known threats"""
    threat_intel = get_threat_intelligence()
    
    threats_found = []
    
    # Check for known malicious IPs
    for ip in threat_intel['known_malicious_ips']:
        if ip in log_line:
            threats_found.append(f"Known malicious IP: {ip}")
    
    # Check for suspicious domains
    for domain in threat_intel['suspicious_domains']:
        if domain in log_line:
            threats_found.append(f"Suspicious domain: {domain}")
    
    # Check for attack patterns
    for attack in threat_intel['recent_attacks']:
        if attack.lower() in log_line.lower():
            threats_found.append(f"Attack pattern: {attack}")
    
    return threats_found

def automated_threat_response(threat):
    """Automated response actions for detected threats"""
    responses = []
    
    # Example automated responses based on threat type
    if 'unauthorized' in threat['message'].lower() or 'access denied' in threat['message'].lower():
        responses.append({
            'action': 'Block IP temporarily',
            'severity': 'high',
            'description': 'Temporarily block source IP for 1 hour'
        })
    
    if 'brute force' in threat['message'].lower() or 'password guess' in threat['message'].lower():
        responses.append({
            'action': 'Increase authentication security',
            'severity': 'medium', 
            'description': 'Enable CAPTCHA and increase lockout threshold'
        })
    
    if 'sql injection' in threat['message'].lower() or 'xss' in threat['message'].lower():
        responses.append({
            'action': 'Web Application Firewall rules',
            'severity': 'high',
            'description': 'Add WAF rules to block specific attack patterns'
        })
    
    return responses

def simulate_incident_response():
    """Simulate incident response procedures"""
    st.header("🛡️ Incident Response Simulation")
    
    if not st.session_state.threats:
        st.info("No threats available for incident response simulation")
        return
    
    selected_threat = st.selectbox(
        "Select Threat for Response Simulation",
        options=[f"Threat #{i+1}: {t['message'][:50]}..." for i, t in enumerate(st.session_state.threats)],
        key="response_threat_select"
    )
    
    if selected_threat:
        threat_index = int(selected_threat.split('#')[1].split(':')[0]) - 1
        threat = st.session_state.threats[threat_index]
        
        st.subheader("Selected Threat Details")
        st.json(threat)
        
        # Get automated responses
        responses = automated_threat_response(threat)
        
        if responses:
            st.subheader("🤖 Automated Response Recommendations")
            
            for i, response in enumerate(responses):
                with st.expander(f"Response #{i+1}: {response['action']}"):
                    st.write(f"**Severity:** {response['severity'].upper()}")
                    st.write(f"**Description:** {response['description']}")
                    
                    if st.button(f"Apply Response #{i+1}", key=f"apply_response_{i}"):
                        add_alert('low', f"Applied response: {response['action']}", "Incident Response")
                        st.success(f"Response #{i+1} applied successfully")
        else:
            st.info("No automated responses available for this threat type")
        
        # Manual response options
        st.subheader("👤 Manual Response Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🕵️ Investigate Threat"):
                st.session_state.threats[threat_index]['status'] = 'investigating'
                add_alert('medium', f"Threat investigation initiated", "Incident Response")
                st.success("Threat investigation started")
        
        with col2:
            if st.button("✅ Mark as Resolved"):
                st.session_state.threats[threat_index]['status'] = 'resolved'
                add_alert('low', f"Threat marked as resolved", "Incident Response")
                st.success("Threat marked as resolved")
        
        with col3:
            if st.button("❌ Mark as False Positive"):
                st.session_state.threats[threat_index]['status'] = 'false positive'
                add_alert('low', f"Threat marked as false positive", "Incident Response")
                st.success("Threat marked as false positive")

def generate_security_report():
    """Generate a comprehensive security report"""
    report = {
        'timestamp': datetime.now(),
        'total_alerts': len(st.session_state.alerts),
        'high_severity_alerts': len([a for a in st.session_state.alerts if a['level'] == 'high']),
        'escalated_threats': len(st.session_state.threats),
        'system_health': get_system_metrics(),
        'log_files_analyzed': len(get_available_log_files()),
        'pipeline_status': 'Running' if st.session_state.pipeline_running else 'Idle'
    }
    
    if st.session_state.last_run_time:
        report['last_pipeline_run'] = st.session_state.last_run_time
    
    return report

def main():
    st.title("🛡️ SIEM Dashboard - MLnextstep")
    st.markdown("---")
    
    # Sidebar - Alert Panel
    with st.sidebar:
        st.header("🚨 Alert Panel")
        
        # Display recent alerts
        if st.session_state.alerts:
            for alert in list(st.session_state.alerts)[-5:]:  # Show last 5 alerts
                alert_class = f"alert-{alert['level']}"
                st.markdown(f'''
                <div class="threat-card {alert_class}">
                    <strong>{alert['timestamp'].strftime('%H:%M:%S')}</strong> - {alert['level'].upper()}<br>
                    <small>{alert['message']}</small>
                    {f"<br><small>Source: {alert['source']}</small>" if alert['source'] else ""}
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No recent alerts")
        
        st.markdown("---")
        st.header("🎛️ Control Panel")
        
        if st.button("🚀 Run Full Pipeline", disabled=st.session_state.pipeline_running):
            run_real_pipeline()
        
        if st.session_state.pipeline_running:
            st.info(f"Running step {st.session_state.current_step}/6...")
        
        if st.session_state.last_run_time:
            st.success(f"Last run: {st.session_state.last_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        st.header("📊 Log Management")
        
        # Log file selection
        log_files = get_available_log_files()
        if log_files:
            selected_log = st.selectbox("Select Log File", log_files)
            st.session_state.selected_log_file = selected_log
            
            if st.button("🔍 Analyze Log Patterns"):
                with st.spinner("Analyzing log patterns..."):
                    alerts = analyze_log_patterns(selected_log)
                    for alert in alerts:
                        add_alert(alert['severity'], 
                                 f"Suspicious pattern in {selected_log}: {alert['pattern']}", 
                                 f"Log Analysis - Line {alert['line_number']}")
                    st.success(f"Found {len(alerts)} suspicious patterns")
        else:
            st.warning("No log files found in logs/ or logs1/ directories")
        
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
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🚨 Threats", "📈 Analytics", "📋 Logs", "⚙️ Configuration"])
    
    with tab1:
        st.header("SIEM Dashboard Overview")
        
        # Load data
        data = load_real_pipeline_data()
        
        if data is None:
            st.warning("No ML pipeline data available. Please run the pipeline first.")
            
            # Show log insights if available
            if st.session_state.selected_log_file:
                st.subheader("Log File Insights")
                try:
                    with open(st.session_state.selected_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Lines", len(lines))
                    col2.metric("File Size", f"{os.path.getsize(st.session_state.selected_log_file) / 1024:.1f} KB")
                    col3.metric("Last Modified", datetime.fromtimestamp(os.path.getmtime(st.session_state.selected_log_file)).strftime('%Y-%m-%d %H:%M'))
                    
                    # Show sample log entries
                    st.subheader("Sample Log Entries")
                    for line in lines[:5]:
                        st.markdown(f'<div class="log-entry">{line.strip()}</div>', unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error reading log file: {e}")
            
            return
        
        # Key metrics
        st.header("🔑 Security Metrics")
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
            st.markdown('<div class="metric-label">Anomalies Detected</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="metric-label">Behavior Clusters</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            alert_count = len([a for a in st.session_state.alerts if a['level'] == 'high'])
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">{}</div>'.format(alert_count), unsafe_allow_html=True)
            st.markdown('<div class="metric-label">High Severity Alerts</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Real-time monitoring section
        st.header("📡 Real-time Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly timeline
            if 'window_start' in data.columns and 'isolation_forest_score' in data.columns:
                fig_anomaly_timeline = px.line(
                    data, 
                    x='window_start', 
                    y='isolation_forest_score',
                    title="Anomaly Detection Timeline",
                    labels={'isolation_forest_score': 'Anomaly Score', 'window_start': 'Time'}
                )
                
                fig_anomaly_timeline.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Anomaly Threshold",
                    annotation_position="top left"
                )
                
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
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#fafafa')
                )
                
                if not show_grid:
                    fig_anomaly_timeline.update_xaxes(showgrid=False)
                    fig_anomaly_timeline.update_yaxes(showgrid=False)
                
                st.plotly_chart(fig_anomaly_timeline, use_container_width=True)
        
        with col2:
            # Alert distribution
            alert_levels = ['high', 'medium', 'low']
            alert_counts = [
                len([a for a in st.session_state.alerts if a['level'] == level])
                for level in alert_levels
            ]
            
            fig_alerts = px.pie(
                values=alert_counts,
                names=[f'{level.upper()} Severity' for level in alert_levels],
                title="Alert Severity Distribution",
                color=alert_levels,
                color_discrete_map={'high': '#ff4b4b', 'medium': '#ffa500', 'low': '#4CAF50'}
            )
            fig_alerts.update_layout(height=300)
            st.plotly_chart(fig_alerts, use_container_width=True)
            
            # System health
            metrics = get_system_metrics()
            fig_system = go.Figure()
            fig_system.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics['cpu'],
                title={'text': "CPU Usage"},
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 0, 'column': 0}
            ))
            fig_system.update_layout(height=200)
            st.plotly_chart(fig_system, use_container_width=True)
    
    with tab2:
        st.header("🚨 Threat Management")
        
        if st.session_state.threats:
            st.subheader("Escalated Threats")
            for i, threat in enumerate(st.session_state.threats):
                with st.expander(f"Threat #{i+1} - {threat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
                    col1, col2 = st.columns(2)
                    col1.write(f"**Message:** {threat['message']}")
                    col1.write(f"**Source:** {threat['source']}")
                    col1.write(f"**Status:** {threat['status']}")
                    col1.write(f"**Escalation Level:** {threat['escalation_level']}")
                    
                    col2.selectbox(f"Update Status #{i+1}", ["new", "investigating", "resolved", "false positive"], 
                                  key=f"threat_status_{i}")
                    if st.button(f"Acknowledge Threat #{i+1}", key=f"ack_{i}"):
                        st.session_state.threats[i]['status'] = "investigating"
                        st.success(f"Threat #{i+1} acknowledged")
        else:
            st.info("No escalated threats detected")
        
        st.subheader("All Alerts")
        if st.session_state.alerts:
            for alert in reversed(list(st.session_state.alerts)):
                alert_class = f"alert-{alert['level']}"
                st.markdown(f'''
                <div class="threat-card {alert_class}">
                    <strong>{alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</strong> - {alert['level'].upper()}<br>
                    {alert['message']}<br>
                    <small>Source: {alert['source']}</small>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No alerts recorded")
        
        # Threat timeline visualization
        st.subheader("Threat Timeline")
        
        if st.session_state.threats:
            threat_dates = [t['timestamp'] for t in st.session_state.threats]
            threat_counts = pd.Series(threat_dates).value_counts().sort_index()
            
            fig_threat_timeline = px.line(
                x=threat_counts.index,
                y=threat_counts.values,
                title="Threat Occurrences Over Time",
                labels={'x': 'Date', 'y': 'Number of Threats'}
            )
            fig_threat_timeline.update_layout(height=300)
            st.plotly_chart(fig_threat_timeline, use_container_width=True)
        
        # Threat severity analysis
        st.subheader("Threat Severity Analysis")
        
        if st.session_state.alerts:
            alert_levels = ['high', 'medium', 'low']
            alert_counts = [
                len([a for a in st.session_state.alerts if a['level'] == level])
                for level in alert_levels
            ]
            
            fig_alert_severity = px.bar(
                x=[level.upper() for level in alert_levels],
                y=alert_counts,
                title="Alerts by Severity Level",
                labels={'x': 'Severity Level', 'y': 'Number of Alerts'},
                color=alert_levels,
                color_discrete_map={'high': '#ff4b4b', 'medium': '#ffa500', 'low': '#4CAF50'}
            )
            fig_alert_severity.update_layout(height=300)
            st.plotly_chart(fig_alert_severity, use_container_width=True)
        
        # Threat source analysis
        st.subheader("Threat Source Analysis")
        
        if st.session_state.alerts:
            sources = [a['source'] for a in st.session_state.alerts if a['source']]
            if sources:
                source_counts = pd.Series(sources).value_counts()
                fig_threat_sources = px.pie(
                    values=source_counts.values,
                    names=source_counts.index,
                    title="Threat Sources Distribution"
                )
                fig_threat_sources.update_layout(height=300)
                st.plotly_chart(fig_threat_sources, use_container_width=True)
            else:
                st.info("No source information available for alerts")
        
        # Add incident response simulation to threats tab
        simulate_incident_response()

    with tab3:
        st.header("📈 Advanced Analytics")
        
        if data is not None:
            # Clustering analysis
            st.subheader("Behavior Clustering")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'dbscan_cluster' in data.columns and 'pca_component_1' in data.columns and 'pca_component_2' in data.columns:
                    fig_clusters = px.scatter(
                        data,
                        x='pca_component_1',
                        y='pca_component_2',
                        color='dbscan_cluster',
                        title="Behavior Clusters (PCA Reduced)",
                        labels={'pca_component_1': 'PCA Component 1', 'pca_component_2': 'PCA Component 2'}
                    )
                    fig_clusters.update_layout(height=400)
                    st.plotly_chart(fig_clusters, use_container_width=True)
            
            with col2:
                if 'dbscan_cluster' in data.columns:
                    cluster_sizes = data['dbscan_cluster'].value_counts()
                    fig_cluster_sizes = px.bar(
                        x=cluster_sizes.index.astype(str),
                        y=cluster_sizes.values,
                        title="Cluster Size Distribution",
                        labels={'x': 'Cluster ID', 'y': 'Number of Points'}
                    )
                    fig_cluster_sizes.update_layout(height=400)
                    st.plotly_chart(fig_cluster_sizes, use_container_width=True)
            
            # Feature analysis
            st.subheader("Feature Correlation")
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col not in ['dbscan_cluster', 'isolation_forest_score', 'isolation_forest_anomaly', 
                                                                     'pca_component_1', 'pca_component_2', 'dbscan_silhouette_score']]
            
            if feature_cols:
                corr_matrix = data[feature_cols[:8]].corr()  # Limit to first 8 features
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Heatmap",
                    aspect="auto",
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        st.header("📋 Log Management")
        
        if st.session_state.selected_log_file:
            st.subheader(f"Log File: {st.session_state.selected_log_file}")
            
            try:
                with open(st.session_state.selected_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Lines", len(lines))
                col2.metric("File Size", f"{os.path.getsize(st.session_state.selected_log_file) / 1024:.1f} KB")
                col3.metric("Last Modified", datetime.fromtimestamp(os.path.getmtime(st.session_state.selected_log_file)).strftime('%Y-%m-%d %H:%M'))
                
                # Log search
                st.subheader("Log Search")
                search_term = st.text_input("Search log entries")
                
                if search_term:
                    matching_lines = [line for line in lines if search_term.lower() in line.lower()]
                    st.write(f"Found {len(matching_lines)} matching lines")
                    for line in matching_lines[:10]:  # Show first 10 matches
                        st.markdown(f'<div class="log-entry">{line.strip()}</div>', unsafe_allow_html=True)
                else:
                    # Show sample log entries
                    st.subheader("Sample Log Entries")
                    for line in lines[:10]:
                        st.markdown(f'<div class="log-entry">{line.strip()}</div>', unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error reading log file: {e}")
        else:
            st.info("Select a log file from the sidebar to view its contents")
    
    with tab5:
        st.header("⚙️ System Configuration")
        
        st.subheader("Pipeline Status")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("ML Pipeline Components")
            st.checkbox("Log Parser", True, disabled=True)
            st.checkbox("Preprocessor", True, disabled=True)
            st.checkbox("Feature Engineering", True, disabled=True)
            st.checkbox("ML Models", True, disabled=True)
        
        with col2:
            st.info("System Information")
            st.write(f"Python Version: {sys.version.split()[0]}")
            st.write(f"Working Directory: {os.getcwd()}")
            st.write(f"Total Log Files: {len(get_available_log_files())}")
        
        st.subheader("Performance Settings")
        st.slider("Analysis Window Size", 100, 5000, 1000, help="Number of log lines to analyze per window")
        st.slider("Anomaly Threshold", -2.0, 2.0, 0.0, step=0.1, help="Adjust anomaly detection sensitivity")
        
        if st.button("🔄 Clear All Alerts"):
            st.session_state.alerts.clear()
            st.session_state.threats.clear()
            st.success("All alerts and threats cleared")
    
    # Real-time log monitoring section
    st.header("🔍 Real-time Log Monitoring")
    
    if st.session_state.selected_log_file:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔄 Refresh Log View"):
                st.rerun()
            
            recent_logs = monitor_logs_realtime(st.session_state.selected_log_file, 20)
            
            if recent_logs:
                st.subheader("Recent Log Entries")
                log_container = st.container()
                
                with log_container:
                    for i, line in enumerate(recent_logs[-10:]):  # Show last 10 entries
                        line_number = len(recent_logs) - 10 + i
                        threats = check_for_known_threats(line)
                        
                        if threats:
                            # Highlight suspicious entries
                            st.markdown(f'''
                            <div class="log-entry" style="border-left-color: #ff4b4b; background-color: #2d2d2d;">
                                <strong style="color: #ff4b4b;">Line {line_number}:</strong><br>
                                {line.strip()}<br>
                                <small style="color: #ff4b4b;">🚨 {', '.join(threats)}</small>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="log-entry">
                                <strong>Line {line_number}:</strong><br>
                                {line.strip()}
                            </div>
                            ''', unsafe_allow_html=True)
            else:
                st.info("No recent log entries found")
        
        with col2:
            st.subheader("Threat Intelligence")
            threat_intel = get_threat_intelligence()
            
            st.metric("Known Malicious IPs", len(threat_intel['known_malicious_ips']))
            st.metric("Suspicious Domains", len(threat_intel['suspicious_domains']))
            st.metric("Recent Attack Types", len(threat_intel['recent_attacks']))
            
            st.info(f"Last update: {threat_intel['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            if st.button("📊 Generate Security Report"):
                report = generate_security_report()
                st.session_state.security_report = report
                st.success("Security report generated!")
    else:
        st.info("Select a log file from the sidebar to enable real-time monitoring")
    
    # Security report display
    if 'security_report' in st.session_state:
        st.header("📋 Security Report")
        report = st.session_state.security_report
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Alerts", report['total_alerts'])
        col2.metric("High Severity", report['high_severity_alerts'])
        col3.metric("Escalated Threats", report['escalated_threats'])
        
        st.json(report)
    
    # Export functionality
    st.header("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Alerts to CSV"):
            if st.session_state.alerts:
                alerts_df = pd.DataFrame(list(st.session_state.alerts))
                csv = alerts_df.to_csv(index=False)
                st.download_button(
                    label="Download Alerts CSV",
                    data=csv,
                    file_name=f"alerts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No alerts to export")
    
    with col2:
        if st.button("📊 Export Threats to JSON"):
            if st.session_state.threats:
                threats_json = json.dumps(st.session_state.threats, default=str, indent=2)
                st.download_button(
                    label="Download Threats JSON",
                    data=threats_json,
                    file_name=f"threats_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No threats to export")
    
    with col3:
        if st.button("🔒 Export Security Report"):
            if 'security_report' in st.session_state:
                report_json = json.dumps(st.session_state.security_report, default=str, indent=2)
                st.download_button(
                    label="Download Security Report",
                    data=report_json,
                    file_name=f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No security report available")

if __name__ == "__main__":
    main()
