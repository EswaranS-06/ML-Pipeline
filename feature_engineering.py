import pandas as pd
import numpy as np
import pathlib
import json
import re
import logging
from typing import Optional, List, Dict, Any
from scipy.stats import entropy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_cleaned_data(path_csv: str = "oplogs/cleaned/cleaned.csv", 
                     path_parquet: str = "oplogs/cleaned/cleaned.parquet") -> pd.DataFrame:
    """Load cleaned data from Parquet (preferred) or CSV fallback.
    
    Args:
        path_csv: Path to CSV file
        path_parquet: Path to Parquet file
        
    Returns:
        DataFrame with cleaned log data
    """
    logger.info("Loading cleaned data...")
    
    # Check if Parquet file exists
    parquet_path = pathlib.Path(path_parquet)
    csv_path = pathlib.Path(path_csv)
    
    if parquet_path.exists():
        logger.info(f"Loading from Parquet: {path_parquet}")
        try:
            df = pd.read_parquet(path_parquet)
        except Exception as e:
            logger.error(f"Error reading Parquet file: {e}")
            df = pd.DataFrame()
    elif csv_path.exists():
        logger.info(f"Loading from CSV: {path_csv}")
        try:
            df = pd.read_csv(path_csv)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            df = pd.DataFrame()
    else:
        logger.error("No cleaned data files found")
        return pd.DataFrame()
    
    if df.empty:
        logger.warning("Loaded empty DataFrame")
        return df
    
    # Expected columns
    expected_columns = [
        'timestamp', 'source_file', 'level', 'indicator_tags_list', 
        'ip_src', 'ip_dst', 'service', 'message', 'peer_port', 'line_number',
        'ip_src_valid', 'ip_dst_valid', 'message_raw', 'day_of_week', 
        'hour_of_day', 'is_weekend'
    ]
    
    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in df.columns:
            df[col] = pd.NA
            logger.warning(f"Missing column: {col}")
    
    # Parse timestamp if it's a string
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        logger.info("Parsing timestamp column...")
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df

def generate_feature_vectors(df: pd.DataFrame, window_size: str = "5min", 
                            group_by_actor: bool = False) -> pd.DataFrame:
    """Generate feature vectors from cleaned log data.
    
    Args:
        df: Input DataFrame with cleaned log data
        window_size: Time window size (e.g., '5min', '1H')
        group_by_actor: Whether to group by actor (IP) instead of time
        
    Returns:
        DataFrame with feature vectors
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for feature engineering")
        return pd.DataFrame()
    
    logger.info(f"Generating feature vectors with window_size={window_size}, group_by_actor={group_by_actor}")
    
    # Create working copy
    df_working = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' not in df_working.columns or df_working['timestamp'].isna().all():
        logger.error("Timestamp column missing or all NaN")
        return pd.DataFrame()
    
    # Set timestamp as index for time-based operations
    df_working = df_working.set_index('timestamp')
    
    # Define grouping strategy
    if group_by_actor:
        # Group by actor (IP) and time window
        if 'ip_src' in df_working.columns:
            group_cols = ['ip_src', pd.Grouper(freq=window_size)]
        else:
            logger.warning("ip_src column missing, falling back to time-only grouping")
            group_cols = [pd.Grouper(freq=window_size)]
    else:
        # Group by time window only
        group_cols = [pd.Grouper(freq=window_size)]
    
    # Group data
    grouped = df_working.groupby(group_cols)
    
    feature_vectors = []
    
    for group_name, group_data in grouped:
        if group_data.empty:
            continue
            
        features = {}
        
        # Debug: print group_name type to understand structure
        logger.debug(f"Group name type: {type(group_name)}, value: {group_name}")
        
        # Handle different group_name structures
        if isinstance(group_name, tuple) and len(group_name) == 2:
            # group_by_actor=True case: (actor_ip, timestamp)
            features['actor_ip'] = group_name[0]
            features['window_start'] = group_name[1]
        elif isinstance(group_name, tuple) and len(group_name) == 1:
            # Single element tuple case
            features['window_start'] = group_name[0]
        else:
            # Single timestamp case
            features['window_start'] = group_name
            
        features['event_count'] = len(group_data)
        
        # 1. Log level statistics
        if 'level' in group_data.columns:
            level_counts = group_data['level'].value_counts()
            for level in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'OTHER']:
                features[f'count_{level.lower()}'] = level_counts.get(level, 0)
            
            # Ratios
            total_events = len(group_data)
            if total_events > 0:
                features['error_ratio'] = features.get('count_error', 0) / total_events
                features['warning_ratio'] = features.get('count_warning', 0) / total_events
                features['critical_ratio'] = features.get('count_critical', 0) / total_events
                
                # Fail-to-success ratio (assuming INFO/DEBUG are success)
                success_count = features.get('count_info', 0) + features.get('count_debug', 0)
                fail_count = features.get('count_error', 0) + features.get('count_critical', 0) + features.get('count_warning', 0)
                features['fail_success_ratio'] = fail_count / (success_count + 1)  # +1 to avoid division by zero
        
        # 2. IP address statistics
        ip_features = {}
        for ip_col in ['ip_src', 'ip_dst']:
            if ip_col in group_data.columns:
                valid_ips = group_data[ip_col].dropna()
                if not valid_ips.empty:
                    ip_features[f'unique_{ip_col}'] = valid_ips.nunique()
                    ip_features[f'{ip_col}_entropy'] = entropy(valid_ips.value_counts(normalize=True))
        
        features.update(ip_features)
        
        # 3. Service statistics
        if 'service' in group_data.columns:
            valid_services = group_data['service'].dropna()
            if not valid_services.empty:
                features['unique_services'] = valid_services.nunique()
                features['service_entropy'] = entropy(valid_services.value_counts(normalize=True))
                
                # Top service frequencies
                top_services = valid_services.value_counts().head(3)
                for i, (service, count) in enumerate(top_services.items()):
                    features[f'top_service_{i+1}'] = service
                    features[f'top_service_{i+1}_freq'] = count / len(valid_services)
        
        # 4. Temporal features
        if not group_data.index.empty:
            time_diffs = group_data.index.to_series().diff().dt.total_seconds().dropna()
            if not time_diffs.empty:
                features['mean_inter_event_time'] = time_diffs.mean()
                features['std_inter_event_time'] = time_diffs.std()
                features['max_inter_event_time'] = time_diffs.max()
                features['min_inter_event_time'] = time_diffs.min()
            
            # Hour distribution
            if 'hour_of_day' in group_data.columns:
                hours = group_data['hour_of_day'].dropna()
                if not hours.empty:
                    features['mean_hour'] = hours.mean()
                    features['after_hours_ratio'] = ((hours < 6) | (hours > 18)).mean()
            
            # Weekend ratio
            if 'is_weekend' in group_data.columns:
                weekends = group_data['is_weekend'].dropna()
                if not weekends.empty:
                    features['weekend_ratio'] = weekends.mean()
        
        # 5. Burstiness features
        if features['event_count'] > 1:
            # Calculate events per sub-window (1 minute granularity)
            sub_windows = group_data.groupby(pd.Grouper(freq='1min')).size()
            if not sub_windows.empty:
                features['peak_events_per_min'] = sub_windows.max()
                features['burstiness_index'] = sub_windows.max() / (sub_windows.mean() + 1e-10)
                features['events_per_min_std'] = sub_windows.std()
        
        # 6. Indicator tags features
        if 'indicator_tags_list' in group_data.columns:
            all_tags = []
            for tags_list in group_data['indicator_tags_list'].dropna():
                if isinstance(tags_list, list):
                    all_tags.extend(tags_list)
            
            if all_tags:
                features['unique_tags_count'] = len(set(all_tags))
                features['total_tags_count'] = len(all_tags)
                features['tags_per_event'] = len(all_tags) / features['event_count']
                
                # Tag entropy
                tag_counts = pd.Series(all_tags).value_counts(normalize=True)
                features['tags_entropy'] = entropy(tag_counts)
                
                # Top tags
                top_tags = pd.Series(all_tags).value_counts().head(3)
                for i, (tag, count) in enumerate(top_tags.items()):
                    features[f'top_tag_{i+1}'] = tag
                    features[f'top_tag_{i+1}_freq'] = count / len(all_tags)
        
        # 7. Message-derived features
        if 'message' in group_data.columns:
            messages = group_data['message'].dropna().astype(str)
            
            # Connection attempt patterns
            connection_patterns = [
                r'connect', r'connection', r'request', r'received', 
                r'establish', r'open', r'close', r'terminate'
            ]
            
            for pattern in connection_patterns:
                pattern_count = messages.str.contains(pattern, case=False, regex=True).sum()
                features[f'msg_{pattern}_count'] = pattern_count
            
            # Port statistics
            if 'peer_port' in group_data.columns:
                valid_ports = group_data['peer_port'].dropna()
                if not valid_ports.empty:
                    features['unique_ports'] = valid_ports.nunique()
                    features['port_min'] = valid_ports.min()
                    features['port_max'] = valid_ports.max()
                    features['port_mean'] = valid_ports.mean()
                    features['port_std'] = valid_ports.std()
        
        # 8. Source file diversity
        if 'source_file' in group_data.columns:
            source_files = group_data['source_file'].dropna()
            if not source_files.empty:
                features['unique_source_files'] = source_files.nunique()
                features['source_file_entropy'] = entropy(source_files.value_counts(normalize=True))
        
        feature_vectors.append(features)
    
    if not feature_vectors:
        logger.warning("No feature vectors generated")
        return pd.DataFrame()
    
    # Create DataFrame from feature vectors
    feature_df = pd.DataFrame(feature_vectors)
    
    # Handle datetime columns
    if 'window_start' in feature_df.columns:
        # Extract timestamp from tuples if needed
        if any(isinstance(x, tuple) for x in feature_df['window_start']):
            feature_df['window_start'] = feature_df['window_start'].apply(lambda x: x[1] if isinstance(x, tuple) and len(x) == 2 else x[0] if isinstance(x, tuple) and len(x) == 1 else x)
        feature_df['window_start'] = pd.to_datetime(feature_df['window_start'])
    
    logger.info(f"Generated {len(feature_df)} feature vectors")
    return feature_df

def normalize_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and encode features for ML readiness.
    
    Args:
        feature_df: Raw feature DataFrame
        
    Returns:
        Normalized feature DataFrame
    """
    if feature_df.empty:
        return feature_df
    
    logger.info("Normalizing features...")
    
    df_normalized = feature_df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = []
    
    # Identify categorical columns (non-numeric with reasonable cardinality)
    for col in df_normalized.columns:
        if col not in numeric_cols and df_normalized[col].nunique() <= 50:
            categorical_cols.append(col)
    
    # Standardize numeric features
    for col in numeric_cols:
        if df_normalized[col].notna().sum() > 0:
            # Handle columns with zero variance
            if df_normalized[col].std() > 0:
                df_normalized[col] = (df_normalized[col] - df_normalized[col].mean()) / df_normalized[col].std()
            else:
                # Constant column, set to 0
                df_normalized[col] = 0
    
    # One-hot encode categorical features
    for col in categorical_cols:
        if col in df_normalized.columns:
            # Get top categories (limit to avoid explosion)
            top_categories = df_normalized[col].value_counts().head(10).index.tolist()
            for category in top_categories:
                new_col_name = f"{col}_{category}"
                df_normalized[new_col_name] = (df_normalized[col] == category).astype(int)
            
            # Drop original categorical column
            df_normalized = df_normalized.drop(columns=[col])
    
    # Fill remaining NaN values
    df_normalized = df_normalized.fillna(0)
    
    logger.info(f"Normalized features. Final shape: {df_normalized.shape}")
    return df_normalized

def main():
    """Main function to run the feature engineering pipeline."""
    try:
        # Load cleaned data
        df = load_cleaned_data()
        if df.empty:
            logger.error("No data loaded. Exiting.")
            return
        
        # Generate feature vectors
        feature_df = generate_feature_vectors(df, window_size="5min", group_by_actor=False)
        if feature_df.empty:
            logger.error("No features generated. Exiting.")
            return
        
        # Normalize features
        normalized_df = normalize_features(feature_df)
        
        # Print summary
        print(f"\nFeature DataFrame shape: {normalized_df.shape}")
        print("\nData types:")
        print(normalized_df.dtypes)
        print("\nFirst 5 rows:")
        print(normalized_df.head(5))
        
        # Print numeric statistics
        numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            print("\nNumeric feature statistics:")
            print(normalized_df[numeric_cols].describe())
        
        # Save feature vectors
        output_dir = pathlib.Path("oplogs/features/")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet
        parquet_path = output_dir / "feature_vectors.parquet"
        normalized_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved feature vectors to {parquet_path}")
        
        # Save as CSV
        csv_path = output_dir / "feature_vectors.csv"
        normalized_df.to_csv(csv_path, index=False)
        logger.info(f"Saved feature vectors to {csv_path}")
        
        return normalized_df
        
    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {e}")
        raise

if __name__ == "__main__":
    main()