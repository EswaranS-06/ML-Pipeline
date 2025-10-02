"""
Main pipeline script for MLnextstep.

This script orchestrates the full workflow:
 1. Parse logs -> oplogs/csv, json
 2. Preprocessing/cleaning -> oplogs/cleaned/
 3. Feature engineering -> oplogs/features/
 4. Feature scaling -> oplogs/scaled_features/
 5. ML training -> oplogs/models/ and oplogs/model_outputs/
 6. Validation -> results/results.csv

The pipeline tries to keep data in-memory where convenient, but also relies on
the existing modules' file outputs to remain compatible with the rest of the
project.
"""
import logging
import time
import sys
from pathlib import Path
import traceback
import pandas as pd

# Project modules
import log_parser_drain3_v3 as parser_mod
from preprocessing import log_cleaner_v1 as cleaner_mod
from preprocessing import text_preprocessor_v3 as tp_mod
import feature_engineering as fe_mod
import feature_scaling as fs_mod
import ml_model_training as ml_mod
from validation.field_validator_v3 import FieldValidator


# Configure logging
LOG_FILE = Path("pipeline.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOG_FILE, mode='a')])
logger = logging.getLogger("main_pipeline")


def timeit(func):
    """Decorator to measure execution time of steps."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


@timeit
def step1_parse_logs():
    logger.info("Step 1: Parsing logs…")
    try:
        # The parser module's main() writes parsed CSV/JSON into oplogs/
        parser_mod.main()
        logger.info("Step 1: Parsing logs… DONE")
        return True
    except SystemExit as se:
        logger.error(f"Log parser exited: {se}")
        return False
    except Exception:
        logger.error("Step 1 failed:\n" + traceback.format_exc())
        return False


@timeit
def step2_preprocessing():
    logger.info("Step 2: Preprocessing/cleaning…")
    try:
        # Ensure TextPreprocessor is importable (no main to run)
        _ = tp_mod.TextPreprocessor()

        # Run log_cleaner which loads oplogs/csv/*.csv and writes oplogs/cleaned/
        df_clean = cleaner_mod.main()

        if df_clean is None or (isinstance(df_clean, pd.DataFrame) and df_clean.empty):
            logger.warning("Preprocessing produced no cleaned data")
        else:
            logger.info(f"Cleaned DataFrame shape: {df_clean.shape}")
            logger.info(f"Sample rows:\n{df_clean.head(3)}")

        logger.info("Step 2: Preprocessing/cleaning… DONE")
        return df_clean
    except Exception:
        logger.error("Step 2 failed:\n" + traceback.format_exc())
        raise


@timeit
def step3_feature_engineering(df_clean: pd.DataFrame):
    logger.info("Step 3: Feature engineering…")
    try:
        if df_clean is None or df_clean.empty:
            logger.error("No cleaned data available for feature engineering")
            return pd.DataFrame()

        # Prefer using functions directly so we keep data in memory  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        feature_df = fe_mod.generate_feature_vectors(df_clean, window_size="5min", group_by_actor=True) #<--here window set to 5min and group by actor True
        if feature_df.empty:                                         #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            logger.error("No feature vectors generated")
            return pd.DataFrame()

        normalized_df = fe_mod.normalize_features(feature_df)

        # Save via same output paths the module uses
        out_dir = Path("oplogs/features")
        out_dir.mkdir(parents=True, exist_ok=True)
        normalized_df.to_parquet(out_dir / "feature_vectors.parquet", index=False)
        normalized_df.to_csv(out_dir / "feature_vectors.csv", index=False)

        logger.info(f"Feature DataFrame shape: {normalized_df.shape}")
        logger.info(f"Sample features:\n{normalized_df.head(3)}")
        logger.info("Step 3: Feature engineering… DONE")
        return normalized_df
    except Exception:
        logger.error("Step 3 failed:\n" + traceback.format_exc())
        raise


@timeit
def step4_feature_scaling(feature_df: pd.DataFrame):
    logger.info("Step 4: Feature scaling/normalization…")
    try:
        if feature_df is None or feature_df.empty:
            logger.error("No feature vectors available for scaling")
            return pd.DataFrame()

        # Use functions from feature_scaling to validate/select and scale
        df_selected = fs_mod.validate_and_select_features(feature_df)
        df_scaled = fs_mod.scale_and_encode_features(df_selected)

        # Save scaled features
        fs_mod.save_processed_features(df_scaled, "oplogs/scaled_features")

        logger.info(f"Scaled features shape: {df_scaled.shape}")
        logger.info(f"Sample scaled features:\n{df_scaled.head(3)}")
        logger.info("Step 4: Feature scaling/normalization… DONE")
        return df_scaled
    except Exception:
        logger.error("Step 4 failed:\n" + traceback.format_exc())
        raise


@timeit
def step5_ml_training(df_scaled: pd.DataFrame):
    logger.info("Step 5: Train ML models and analyze outputs…")
    try:
        if df_scaled is None or df_scaled.empty:
            logger.error("No scaled features available for ML training")
            return None

        # Prepare features for modeling (this will drop non-numeric columns)
        X, feature_cols = ml_mod.prepare_features(df_scaled)

        # Train models
        model_results = ml_mod.train_cluster_models(X, feature_cols)

        # Create output DataFrame and save models/outputs
        df_output = ml_mod.create_output_dataframe(df_scaled, model_results)

        # Save artifacts
        ml_mod.save_models(model_results, "oplogs")
        ml_mod.save_model_outputs(df_output, "oplogs", "combined")
        ml_mod.save_performance_metrics(df_output, "oplogs")
        ml_mod.visualize_results(df_output, model_results)

        logger.info("Step 5: Train ML models and analyze outputs… DONE")
        return df_output, model_results
    except Exception:
        logger.error("Step 5 failed:\n" + traceback.format_exc())
        raise


@timeit
def step6_validation(df_clean: pd.DataFrame, df_output: pd.DataFrame):
    logger.info("Step 6: Validation…")
    results = {}
    try:
        validator = FieldValidator()

        # Validate cleaned logs (structural validation)
        try:
            df_validated = validator.validate_dataframe(df_clean.copy())
            results['cleaned_valid_rows'] = len(df_validated) if df_validated is not None else 0
            logger.info(f"Validated cleaned data rows: {results['cleaned_valid_rows']}")
        except Exception as e:
            logger.warning(f"Cleaned data validation failed: {e}")

        # Basic checks on model outputs
        if df_output is not None and not df_output.empty:
            results['model_output_rows'] = len(df_output)
            results['n_clusters'] = df_output['dbscan_cluster'].nunique() if 'dbscan_cluster' in df_output.columns else None
            results['n_anomalies'] = int(df_output['isolation_forest_anomaly'].sum()) if 'isolation_forest_anomaly' in df_output.columns else None
            logger.info(f"Model outputs rows: {results['model_output_rows']}, clusters: {results['n_clusters']}, anomalies: {results['n_anomalies']}")
        else:
            logger.warning("No model output available to validate")

        # Save final summarized results to results/results.csv
        Path("results").mkdir(parents=True, exist_ok=True)
        final_results_path = Path("results/results.csv")

        # Build a small summary DataFrame
        summary_df = pd.DataFrame([{
            'timestamp': pd.Timestamp.now().isoformat(),
            'cleaned_rows': results.get('cleaned_valid_rows', 0),
            'model_output_rows': results.get('model_output_rows', 0),
            'n_clusters': results.get('n_clusters', 0),
            'n_anomalies': results.get('n_anomalies', 0)
        }])

        if final_results_path.exists():
            summary_df.to_csv(final_results_path, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(final_results_path, index=False)

        logger.info(f"Saved final results summary to {final_results_path}")
        logger.info("Step 6: Validation… DONE")
        return results
    except Exception:
        logger.error("Step 6 failed:\n" + traceback.format_exc())
        raise


def main():
    logger.info("Starting MLnextstep main pipeline")

    # Step 1
    ok = step1_parse_logs()
    if not ok:
        logger.error("Aborting pipeline due to parsing failure")
        return 1

    # Step 2
    df_clean = step2_preprocessing()

    # Step 3
    feature_df = step3_feature_engineering(df_clean)

    # Step 4
    df_scaled = step4_feature_scaling(feature_df)

    # Step 5
    ml_outputs = step5_ml_training(df_scaled)
    df_output, model_results = (ml_outputs if ml_outputs is not None else (None, None))

    # Step 6
    step6_validation(df_clean, df_output)

    logger.info("Pipeline finished")
    return 0


if __name__ == '__main__':
    sys.exit(main())
