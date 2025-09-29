"""
Feature Scaling and Encoding Script

This script loads feature vectors, validates them, selects relevant features,
and applies scaling/encoding to prepare data for machine learning.
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_feature_vectors(path_csv: str, path_parquet: str) -> pd.DataFrame:
    """
    Load feature vectors from parquet or CSV file.
    
    Args:
        path_csv (str): Path to CSV file
        path_parquet (str): Path to parquet file
        
    Returns:
        pd.DataFrame: Loaded feature vectors
    """
    logger.info("Loading feature vectors...")
    
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


def validate_and_select_features(df: pd.DataFrame, correlation_threshold: float = 0.95) -> pd.DataFrame:
    """
    Validate features and select relevant ones for ML.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        correlation_threshold (float): Threshold for high correlation
        
    Returns:
        pd.DataFrame: Selected features DataFrame
    """
    logger.info("Validating and selecting features...")
    
    # Basic validation
    logger.info(f"Original DataFrame shape: {df.shape}")
    logger.info("Data types:\n" + str(df.dtypes))
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
        # Impute missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        logger.info("Missing values imputed with median")
    else:
        logger.info("No missing values found")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Numeric columns: {len(numeric_cols)}")
    logger.info(f"Categorical columns: {len(categorical_cols)}")
    
    # Summary statistics for numeric features
    if numeric_cols:
        numeric_stats = df[numeric_cols].describe()
        logger.info("Numeric features summary statistics:")
        for col in numeric_cols:
            logger.info(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
                      f"min={df[col].min():.4f}, max={df[col].max():.4f}")
    
    # Identify constant or near-zero variance columns
    constant_cols = []
    for col in numeric_cols:
        if df[col].std() < 1e-10:  # Near zero variance
            constant_cols.append(col)
            logger.warning(f"Constant/near-zero variance column: {col}")
    
    # Drop constant columns
    if constant_cols:
        logger.info(f"Dropping constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)
        numeric_cols = [col for col in numeric_cols if col not in constant_cols]
    
    # Compute correlation matrix for numeric features
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                                          corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            logger.warning("Highly correlated feature pairs (>0.95):")
            for pair in high_corr_pairs:
                logger.warning(f"  {pair[0]} - {pair[1]}: {pair[2]:.4f}")
            
            # Drop one from each highly correlated pair
            cols_to_drop = set()
            for pair in high_corr_pairs:
                col1, col2, _ = pair
                if col1 not in cols_to_drop and col2 not in cols_to_drop:
                    # Keep the one with higher variance
                    if df[col1].var() > df[col2].var():
                        cols_to_drop.add(col2)
                    else:
                        cols_to_drop.add(col1)
            
            if cols_to_drop:
                logger.info(f"Dropping highly correlated columns: {list(cols_to_drop)}")
                df = df.drop(columns=list(cols_to_drop))
                numeric_cols = [col for col in numeric_cols if col not in cols_to_drop]
        else:
            logger.info("No highly correlated feature pairs found")
    
    logger.info(f"Final selected features shape: {df.shape}")
    return df


def scale_and_encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numeric features and encode categorical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Scaled and encoded DataFrame
    """
    logger.info("Scaling and encoding features...")
    
    # Create a copy to avoid modifying original data
    df_scaled = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Scale numeric features using StandardScaler (z-score normalization)
    if numeric_cols:
        scaler = StandardScaler()
        scaled_numeric = scaler.fit_transform(df[numeric_cols])
        
        # Create new column names with _scaled suffix
        scaled_col_names = [f"{col}_scaled" for col in numeric_cols]
        
        # Add scaled features to DataFrame
        df_scaled[scaled_col_names] = scaled_numeric
        
        # Drop original numeric columns
        df_scaled = df_scaled.drop(columns=numeric_cols)
        
        logger.info(f"Scaled {len(numeric_cols)} numeric features")
    
    # Encode categorical features using one-hot encoding
    if categorical_cols:
        for col in categorical_cols:
            # Check if column has reasonable number of categories
            unique_count = df[col].nunique()
            if unique_count <= 20:  # Reasonable for one-hot encoding
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_scaled = pd.concat([df_scaled, dummies], axis=1)
                logger.info(f"One-hot encoded '{col}' with {unique_count} categories")
            else:
                # Label encode for high cardinality
                le = LabelEncoder()
                df_scaled[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                logger.info(f"Label encoded '{col}' with {unique_count} categories")
        
        # Drop original categorical columns
        df_scaled = df_scaled.drop(columns=categorical_cols)
    
    logger.info(f"Final scaled DataFrame shape: {df_scaled.shape}")
    logger.info("First 5 rows of scaled DataFrame:")
    logger.info(df_scaled.head())
    
    return df_scaled


def save_processed_features(df: pd.DataFrame, output_dir: str) -> None:
    """
    Save processed features to parquet and CSV files.
    
    Args:
        df (pd.DataFrame): Processed DataFrame
        output_dir (str): Output directory path
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to parquet
    parquet_path = os.path.join(output_dir, "feature_vectors_scaled.parquet")
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved scaled features to parquet: {parquet_path}")
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "feature_vectors_scaled.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved scaled features to CSV: {csv_path}")


def main():
    """Main function to execute the feature processing pipeline."""
    try:
        # Define file paths
        base_dir = "oplogs/features"
        csv_path = os.path.join(base_dir, "feature_vectors.csv")
        parquet_path = os.path.join(base_dir, "feature_vectors.parquet")
        output_dir = "oplogs/scaled_features"
        
        # Load feature vectors
        df = load_feature_vectors(csv_path, parquet_path)
        
        # Validate and select features
        df_selected = validate_and_select_features(df)
        
        # Scale and encode features
        df_scaled = scale_and_encode_features(df_selected)
        
        # Save processed features
        save_processed_features(df_scaled, output_dir)
        
        logger.info("Feature processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature processing: {e}")
        raise


if __name__ == "__main__":
    main()