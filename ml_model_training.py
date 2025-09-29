"""
Machine Learning Model Training Script

This script loads scaled feature vectors, trains clustering and anomaly detection models,
analyzes the results, and saves trained models and outputs.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Machine learning imports
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Visualization libraries not available. Skipping plots.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_scaled_features(path_csv: str, path_parquet: str) -> pd.DataFrame:
    """
    Load scaled feature vectors from parquet or CSV file.
    
    Args:
        path_csv (str): Path to CSV file
        path_parquet (str): Path to parquet file
        
    Returns:
        pd.DataFrame: Loaded scaled features
    """
    logger.info("Loading scaled feature vectors...")
    
    # Try to load from parquet first
    if os.path.exists(path_parquet):
        try:
            df = pd.read_parquet(path_parquet)
            logger.info(f"Loaded {len(df)} rows from parquet file: {path_parquet}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load parquet file: {e}")
    
    # Fallback to CSV
    if os.path.exists(path_csv):
        try:
            df = pd.read_csv(path_csv)
            logger.info(f"Loaded {len(df)} rows from CSV file: {path_csv}")
            
            # Convert timestamp column to datetime if present
            timestamp_cols = ['window_start', 'timestamp', 'time']
            for col in timestamp_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], utc=True)
                        logger.info(f"Converted column '{col}' to datetime64[ns, UTC]")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to convert column '{col}' to datetime: {e}")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
    
    raise FileNotFoundError("Neither parquet nor CSV file found")


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for modeling by separating timestamp and feature columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame with timestamp and features
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Features DataFrame and feature column names
    """
    logger.info("Preparing features for modeling...")
    
    # Identify timestamp column
    timestamp_cols = ['window_start', 'timestamp', 'time']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break
    
    if timestamp_col:
        logger.info(f"Found timestamp column: {timestamp_col}")
        # Extract features (all columns except timestamp)
        feature_cols = [col for col in df.columns if col != timestamp_col]
        X = df[feature_cols].copy()
    else:
        logger.warning("No timestamp column found. Using all columns as features.")
        feature_cols = df.columns.tolist()
        X = df.copy()
    
    # Ensure all features are numeric
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) != len(feature_cols):
        non_numeric = [col for col in feature_cols if col not in numeric_cols]
        logger.warning(f"Dropping non-numeric columns: {non_numeric}")
        X = X[numeric_cols]
        feature_cols = numeric_cols
    
    logger.info(f"Using {len(feature_cols)} numeric features for modeling")
    logger.info(f"Feature columns: {feature_cols}")
    
    return X, feature_cols


def train_cluster_models(df_features: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Train clustering models on the scaled features.
    
    Args:
        df_features (pd.DataFrame): Features DataFrame
        feature_cols (List[str]): List of feature column names
        
    Returns:
        Dict[str, Any]: Dictionary containing trained models and their outputs
    """
    logger.info("Training clustering models...")
    
    X = df_features[feature_cols].values
    results = {}
    
    # DBSCAN clustering
    logger.info("Training DBSCAN model...")
    try:
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X)
        
        # Calculate silhouette score (excluding noise points)
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 1:  # Need at least 2 samples for silhouette
            sil_score = silhouette_score(X[non_noise_mask], dbscan_labels[non_noise_mask])
        else:
            sil_score = -1  # Invalid score
        
        cluster_counts = pd.Series(dbscan_labels).value_counts().sort_index()
        
        dbscan_results = {
            'model': dbscan,
            'labels': dbscan_labels,
            'silhouette_score': sil_score,
            'cluster_counts': cluster_counts,
            'n_clusters': len(cluster_counts) - (1 if -1 in cluster_counts.index else 0),
            'n_noise': cluster_counts.get(-1, 0)
        }
        
        logger.info(f"DBSCAN results: {dbscan_results['n_clusters']} clusters, "
                   f"{dbscan_results['n_noise']} noise points, "
                   f"silhouette score: {dbscan_results['silhouette_score']:.4f}")
        logger.info(f"Cluster sizes: {dict(cluster_counts)}")
        
        results['dbscan'] = dbscan_results
        
    except Exception as e:
        logger.error(f"DBSCAN training failed: {e}")
    
    # Isolation Forest for anomaly detection
    logger.info("Training Isolation Forest model...")
    try:
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_scores = iso_forest.fit_predict(X)
        
        # Convert scores to anomaly labels (1 = normal, -1 = anomaly)
        anomaly_labels = np.where(iso_scores == -1, 1, 0)  # 1 = anomaly, 0 = normal
        anomaly_scores = iso_forest.decision_function(X)
        
        # Mark top 5% as anomalies
        threshold = np.percentile(anomaly_scores, 5)
        top_anomalies = anomaly_scores <= threshold
        
        iso_results = {
            'model': iso_forest,
            'scores': anomaly_scores,
            'labels': anomaly_labels,
            'top_anomalies': top_anomalies,
            'n_anomalies': np.sum(anomaly_labels),
            'n_top_anomalies': np.sum(top_anomalies)
        }
        
        logger.info(f"Isolation Forest results: {iso_results['n_anomalies']} anomalies, "
                   f"{iso_results['n_top_anomalies']} top anomalies")
        logger.info(f"Anomaly score range: [{anomaly_scores.min():.4f}, {anomaly_scores.max():.4f}]")
        
        results['isolation_forest'] = iso_results
        
    except Exception as e:
        logger.error(f"Isolation Forest training failed: {e}")
    
    # Optional: PCA for dimensionality reduction and visualization
    logger.info("Performing PCA for dimensionality reduction...")
    try:
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(X)
        
        pca_results = {
            'model': pca,
            'features': pca_features,
            'explained_variance': pca.explained_variance_ratio_.sum()
        }
        
        logger.info(f"PCA explained variance: {pca_results['explained_variance']:.4f}")
        
        results['pca'] = pca_results
        
    except Exception as e:
        logger.error(f"PCA failed: {e}")
    
    return results


def create_output_dataframe(df_original: pd.DataFrame, model_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create output DataFrame with model predictions.
    
    Args:
        df_original (pd.DataFrame): Original DataFrame with timestamp
        model_results (Dict[str, Any]): Model results dictionary
        
    Returns:
        pd.DataFrame: Output DataFrame with predictions
    """
    logger.info("Creating output DataFrame with model predictions...")
    
    # Start with original DataFrame
    df_output = df_original.copy()
    
    # Add DBSCAN cluster labels if available
    if 'dbscan' in model_results:
        dbscan_results = model_results['dbscan']
        df_output['dbscan_cluster'] = dbscan_results['labels']
        df_output['dbscan_silhouette_score'] = dbscan_results['silhouette_score']
    
    # Add Isolation Forest results if available
    if 'isolation_forest' in model_results:
        iso_results = model_results['isolation_forest']
        df_output['isolation_forest_score'] = iso_results['scores']
        df_output['isolation_forest_anomaly'] = iso_results['labels']
        df_output['isolation_forest_top_anomaly'] = iso_results['top_anomalies']
    
    # Add PCA features if available
    if 'pca' in model_results:
        pca_results = model_results['pca']
        df_output['pca_component_1'] = pca_results['features'][:, 0]
        df_output['pca_component_2'] = pca_results['features'][:, 1]
    
    # Add timestamp for tracking
    df_output['processing_timestamp'] = datetime.now().isoformat()
    
    logger.info(f"Output DataFrame shape: {df_output.shape}")
    logger.info(f"Output columns: {df_output.columns.tolist()}")
    
    return df_output


def save_models(models_dict: Dict[str, Any], output_dir: str) -> None:
    """
    Save trained models to disk.
    
    Args:
        models_dict (Dict[str, Any]): Dictionary containing trained models
        output_dir (str): Output directory path
    """
    logger.info("Saving trained models...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    saved_models = []
    
    for model_name, results in models_dict.items():
        if 'model' in results:
            model = results['model']
            model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
            
            try:
                joblib.dump(model, model_path)
                saved_models.append(model_name)
                logger.info(f"Saved {model_name} model to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save {model_name} model: {e}")
    
    logger.info(f"Saved {len(saved_models)} models: {saved_models}")


def save_model_outputs(df_output: pd.DataFrame, output_dir: str, model_name: str) -> None:
    """
    Save model outputs to disk.
    
    Args:
        df_output (pd.DataFrame): Output DataFrame with predictions
        output_dir (str): Output directory path
        model_name (str): Name of the model
    """
    logger.info(f"Saving {model_name} outputs...")
    
    # Create output directory if it doesn't exist
    outputs_dir = os.path.join(output_dir, "model_outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Save to parquet
    parquet_path = os.path.join(outputs_dir, f"{model_name}_output.parquet")
    df_output.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {model_name} outputs to parquet: {parquet_path}")
    
    # Save to CSV
    csv_path = os.path.join(outputs_dir, f"{model_name}_output.csv")
    df_output.to_csv(csv_path, index=False)
    logger.info(f"Saved {model_name} outputs to CSV: {csv_path}")


def save_performance_metrics(df_output: pd.DataFrame, output_dir: str) -> None:
    """
    Save performance metrics separately in the performance folder.
    
    Args:
        df_output (pd.DataFrame): Output DataFrame with performance metrics
        output_dir (str): Output directory path
    """
    logger.info("Saving performance metrics separately...")
    
    # Define the performance metrics columns to save
    performance_cols = [
        'dbscan_cluster',
        'dbscan_silhouette_score',
        'isolation_forest_score',
        'isolation_forest_anomaly',
        'isolation_forest_top_anomaly',
        'pca_component_1',
        'pca_component_2',
        'processing_timestamp'
    ]
    
    # Check if all required columns exist
    available_cols = [col for col in performance_cols if col in df_output.columns]
    missing_cols = [col for col in performance_cols if col not in df_output.columns]
    
    if missing_cols:
        logger.warning(f"Missing performance metrics columns: {missing_cols}")
    
    if not available_cols:
        logger.error("No performance metrics columns found to save")
        return
    
    # Create performance directory if it doesn't exist
    performance_dir = os.path.join(output_dir, "model_outputs", "performance")
    os.makedirs(performance_dir, exist_ok=True)
    
    # Extract performance metrics
    df_performance = df_output[['window_start'] + available_cols].copy()
    
    # Save to parquet
    parquet_path = os.path.join(performance_dir, "performance_metrics.parquet")
    df_performance.to_parquet(parquet_path, index=False)
    logger.info(f"Saved performance metrics to parquet: {parquet_path}")
    
    # Save to CSV
    csv_path = os.path.join(performance_dir, "performance_metrics.csv")
    df_performance.to_csv(csv_path, index=False)
    logger.info(f"Saved performance metrics to CSV: {csv_path}")
    
    # Also save individual model outputs
    save_individual_model_outputs(df_output, performance_dir)
    
    logger.info(f"Performance metrics saved with {len(available_cols)} columns")


def save_individual_model_outputs(df_output: pd.DataFrame, output_dir: str) -> None:
    """
    Save individual model outputs separately.
    
    Args:
        df_output (pd.DataFrame): Output DataFrame with all metrics
        output_dir (str): Output directory path
    """
    logger.info("Saving individual model outputs...")
    
    # DBSCAN outputs
    if 'dbscan_cluster' in df_output.columns:
        dbscan_cols = ['window_start', 'dbscan_cluster', 'dbscan_silhouette_score']
        df_dbscan = df_output[dbscan_cols].copy()
        df_dbscan.to_parquet(os.path.join(output_dir, "dbscan_output.parquet"), index=False)
        df_dbscan.to_csv(os.path.join(output_dir, "dbscan_output.csv"), index=False)
    
    # Isolation Forest outputs
    if 'isolation_forest_score' in df_output.columns:
        iso_cols = ['window_start', 'isolation_forest_score', 'isolation_forest_anomaly', 'isolation_forest_top_anomaly']
        df_iso = df_output[iso_cols].copy()
        df_iso.to_parquet(os.path.join(output_dir, "isolation_forest_output.parquet"), index=False)
        df_iso.to_csv(os.path.join(output_dir, "isolation_forest_output.csv"), index=False)
    
    # PCA outputs
    if 'pca_component_1' in df_output.columns:
        pca_cols = ['window_start', 'pca_component_1', 'pca_component_2']
        df_pca = df_output[pca_cols].copy()
        df_pca.to_parquet(os.path.join(output_dir, "pca_output.parquet"), index=False)
        df_pca.to_csv(os.path.join(output_dir, "pca_output.csv"), index=False)
    
    # Combine all model-specific outputs into a single models_output file
    try:
        logger.info("Combining individual model outputs into a single models_output file...")

        # Start from window_start to ensure consistent ordering and full join
        df_models = df_output[['window_start']].copy()

        if 'df_dbscan' in locals():
            df_models = df_models.merge(df_dbscan, on='window_start', how='left')
        elif 'dbscan_cluster' in df_output.columns:
            # fallback if df_dbscan wasn't created for some reason
            df_models = df_models.merge(df_output[['window_start', 'dbscan_cluster', 'dbscan_silhouette_score']], on='window_start', how='left')

        if 'df_iso' in locals():
            df_models = df_models.merge(df_iso, on='window_start', how='left')
        elif 'isolation_forest_score' in df_output.columns:
            df_models = df_models.merge(df_output[['window_start', 'isolation_forest_score', 'isolation_forest_anomaly', 'isolation_forest_top_anomaly']], on='window_start', how='left')

        if 'df_pca' in locals():
            df_models = df_models.merge(df_pca, on='window_start', how='left')
        elif 'pca_component_1' in df_output.columns:
            df_models = df_models.merge(df_output[['window_start', 'pca_component_1', 'pca_component_2']], on='window_start', how='left')

        # Save combined models output to performance directory
        models_combined_parquet = os.path.join(output_dir, "models_output.parquet")
        models_combined_csv = os.path.join(output_dir, "models_output.csv")
        df_models.to_parquet(models_combined_parquet, index=False)
        df_models.to_csv(models_combined_csv, index=False)
        logger.info(f"Saved combined models output to parquet: {models_combined_parquet}")
        logger.info(f"Saved combined models output to CSV: {models_combined_csv}")

    except Exception as e:
        logger.error(f"Failed to create combined models_output file: {e}")

    logger.info("Individual model outputs saved")


def visualize_results(df_output: pd.DataFrame, models_dict: Dict[str, Any]) -> None:
    """
    Create visualizations of model results (optional).
    
    Args:
        df_output (pd.DataFrame): Output DataFrame with predictions
        models_dict (Dict[str, Any]): Model results dictionary
    """
    if not VISUALIZATION_AVAILABLE:
        logger.info("Skipping visualizations - matplotlib/seaborn not available")
        return
    
    logger.info("Creating visualizations...")
    
    try:
        # Create plots directory
        plots_dir = "oplogs/model_outputs/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: DBSCAN clusters (if PCA available)
        if 'dbscan' in models_dict and 'pca' in models_dict:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df_output['pca_component_1'], df_output['pca_component_2'], 
                                c=df_output['dbscan_cluster'], cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title('DBSCAN Clusters (PCA Reduced)')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.savefig(os.path.join(plots_dir, 'dbscan_clusters_pca.png'))
            plt.show()
            plt.close()
        
        # Plot 2: Isolation Forest anomaly scores histogram
        if 'isolation_forest' in models_dict:
            plt.figure(figsize=(10, 6))
            plt.hist(df_output['isolation_forest_score'], bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=np.percentile(df_output['isolation_forest_score'], 5), 
                       color='red', linestyle='--', label='Top 5% threshold')
            plt.title('Isolation Forest Anomaly Scores Distribution')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(plots_dir, 'isolation_forest_scores.png'))
            plt.show()
            plt.close()
        
        # Plot 3: Cluster size distribution
        if 'dbscan' in models_dict:
            plt.figure(figsize=(10, 6))
            cluster_counts = models_dict['dbscan']['cluster_counts']
            # Exclude noise cluster (-1) for better visualization
            valid_clusters = cluster_counts[cluster_counts.index != -1]
            if len(valid_clusters) > 0:
                valid_clusters.plot(kind='bar')
                plt.title('DBSCAN Cluster Sizes (Excluding Noise)')
                plt.xlabel('Cluster ID')
                plt.ylabel('Number of Points')
                plt.savefig(os.path.join(plots_dir, 'dbscan_cluster_sizes.png'))
                plt.show()
                plt.close()
        
        logger.info("Visualizations saved to oplogs/model_outputs/plots/")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def main():
    """Main function to execute the ML model training pipeline."""
    try:
        # Define file paths
        base_dir = "oplogs/scaled_features"
        csv_path = os.path.join(base_dir, "feature_vectors_scaled.csv")
        parquet_path = os.path.join(base_dir, "feature_vectors_scaled.parquet")
        output_base_dir = "oplogs"
        
        # Load scaled features
        df = load_scaled_features(csv_path, parquet_path)
        
        # Prepare features for modeling
        X, feature_cols = prepare_features(df)
        
        # Train models
        model_results = train_cluster_models(X, feature_cols)
        
        # Create output DataFrame
        df_output = create_output_dataframe(df, model_results)
        
        # Save models
        save_models(model_results, output_base_dir)
        
        # Save outputs
        save_model_outputs(df_output, output_base_dir, "combined")
        
        # Save performance metrics
        save_performance_metrics(df_output, output_base_dir)
        
        # Create visualizations
        visualize_results(df_output, model_results)
        
        logger.info("ML model training completed successfully!")
        
        # Print final summary
        if 'dbscan' in model_results:
            dbscan = model_results['dbscan']
            logger.info(f"DBSCAN Summary: {dbscan['n_clusters']} clusters, "
                       f"{dbscan['n_noise']} noise points, "
                       f"Silhouette: {dbscan['silhouette_score']:.4f}")
        
        if 'isolation_forest' in model_results:
            iso = model_results['isolation_forest']
            logger.info(f"Isolation Forest Summary: {iso['n_anomalies']} anomalies, "
                       f"{iso['n_top_anomalies']} top anomalies")
        
        if 'pca' in model_results:
            pca = model_results['pca']
            logger.info(f"PCA Summary: {pca['explained_variance']:.4f} explained variance")
        
    except Exception as e:
        logger.error(f"Error in ML model training: {e}")
        raise


if __name__ == "__main__":
    main()