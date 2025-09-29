import pandas as pd
import pathlib
import glob
import ipaddress
import re
import json
import logging
from dateutil import parser
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_parsed_csvs(path_pattern: str = "oplogs/csv/*.csv", chunksize: Optional[int] = None) -> pd.DataFrame:
    """Read all CSVs matching path_pattern into a single DataFrame.
       - Use chunksize when provided (iterate and concat).
       - Use on_bad_lines='skip' to avoid crash on malformed rows.
       - Ensure correct dtypes where possible (read as strings then convert)."""
    
    logger.info(f"Loading CSV files matching pattern: {path_pattern}")
    
    # Get all matching files
    csv_files = glob.glob(path_pattern)
    if not csv_files:
        logger.warning(f"No CSV files found matching pattern: {path_pattern}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    # Expected columns to handle missing files gracefully
    expected_columns = [
        'timestamp', 'source_file', 'level', 'indicator_tags', 
        'ip_src', 'ip_dst', 'service', 'message', 'line_number'
    ]
    
    all_dfs = []
    
    for file_path in csv_files:
        logger.info(f"Processing file: {file_path}")
        
        try:
            if chunksize:
                # Process in chunks for large files
                chunk_list = []
                for chunk in pd.read_csv(file_path, chunksize=chunksize, on_bad_lines='skip', dtype=str):
                    # Ensure all expected columns exist
                    for col in expected_columns:
                        if col not in chunk.columns:
                            chunk[col] = pd.NA
                    chunk_list.append(chunk)
                df_file = pd.concat(chunk_list, ignore_index=True)
            else:
                # Read entire file at once
                df_file = pd.read_csv(file_path, on_bad_lines='skip', dtype=str)
                # Ensure all expected columns exist
                for col in expected_columns:
                    if col not in df_file.columns:
                        df_file[col] = pd.NA
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
        
        all_dfs.append(df_file)
    
    if not all_dfs:
        logger.warning("No data loaded from any CSV files")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    final_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Loaded {len(final_df)} rows from {len(csv_files)} files")
    
    return final_df

def clean_normalize_logs(df: pd.DataFrame, tz: str = "UTC", convert_tz: bool = False) -> pd.DataFrame:
    """Clean and normalize columns:
       - normalize column names to snake_case lowercase
       - parse timestamp to timezone-aware datetime (coerce invalid -> NaT). If convert_tz True, convert to tz argument.
       - trim whitespace and replace empty strings with NaN
       - normalize log levels to set {CRITICAL, ERROR, WARNING, INFO, DEBUG, OTHER} (map synonyms and choose highest severity when multiple)
       - split indicator_tags into list (support separators ; , | / whitespace), store as JSON-string or list
       - validate ip_src and ip_dst: mark invalids as NaN and add boolean columns ip_src_valid, ip_dst_valid
       - convert line_number to Int64 (nullable integer) and coerce errors
       - deduplicate rows (use timestamp + source_file + line_number if available; else drop exact duplicates)
       - sort by timestamp ascending
       - add derived helper columns: day_of_week, hour_of_day (0-23), is_weekend(bool)
       - store original message under column `message_raw` and optionally extract simple structured fields (port if message contains '/:PORT' or 'port XXX')
       - return cleaned DataFrame"""
    
    if df.empty:
        logger.warning("Empty DataFrame provided for cleaning")
        return df
    
    logger.info("Starting data cleaning and normalization")
    df_clean = df.copy()
    
    # 1. Normalize column names to snake_case lowercase
    df_clean.columns = [col.strip().lower().replace(' ', '_') for col in df_clean.columns]
    logger.info("Column names normalized to snake_case")
    
    # 2. Trim whitespace and replace empty strings with NaN for all string columns
    string_cols = df_clean.select_dtypes(include=['object']).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].replace(['', 'nan', 'None'], pd.NA)
    
    
    # 3. Parse timestamp to timezone-aware datetime  <----v1 timestamp parsing (commented out)
    # logger.info("Parsing timestamps...")
    # if 'timestamp' in df_clean.columns:
    #     # First attempt with pandas to_datetime
    #     df_clean['timestamp'] = pd.to_datetime(
    #         df_clean['timestamp'], 
    #         infer_datetime_format=True, 
    #         utc=True, 
    #         errors='coerce'
    #     )
        
    #     # Fallback for rows that couldn't be parsed by pandas
    #     nat_mask = df_clean['timestamp'].isna()
    #     if nat_mask.any():
    #         logger.warning(f"{nat_mask.sum()} timestamps couldn't be parsed by pandas, trying dateutil fallback")
    #         for idx in df_clean[nat_mask].index:
    #             try:
    #                 original_ts = df_clean.at[idx, 'timestamp']
    #                 if pd.notna(original_ts) and isinstance(original_ts, str):
    #                     parsed_ts = parser.parse(original_ts, fuzzy=True)
    #                     if parsed_ts.tzinfo is None:
    #                         parsed_ts = parsed_ts.replace(tzinfo=parser.UTC)
    #                     df_clean.at[idx, 'timestamp'] = parsed_ts
    #             except (ValueError, TypeError):
    #                 continue
        
    #     # Convert timezone if requested
    #     if convert_tz and df_clean['timestamp'].notna().any():
    #         df_clean['timestamp'] = df_clean['timestamp'].dt.tz_convert(tz)
    
    
    # 3. Parse timestamp to timezone-aware datetime and normalize to ISO 8601
    # <----v2 improved timestamp parsing with dateutil
    if 'timestamp' in df_clean.columns:
        logger.info("Parsing timestamps and normalizing to ISO 8601 UTC...")
        
        def parse_to_iso(ts):
            if pd.isna(ts):
                return pd.NaT
            try:
                # Use dateutil parser to handle multiple formats
                dt = parser.parse(str(ts), fuzzy=True)
                # Force UTC if no timezone info
                if dt.tzinfo is None:
                    from datetime import timezone
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    # Convert to UTC
                    dt = dt.astimezone(timezone.utc)
                return dt
            except Exception:
                return pd.NaT
        
        # Apply parser to all timestamps
        df_clean['timestamp'] = df_clean['timestamp'].apply(parse_to_iso)
        
        # Convert to ISO 8601 string format if desired
        df_clean['timestamp_iso'] = df_clean['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Convert back to datetime type for downstream processing if needed
        df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], utc=True)
    
    # 4. Normalize log levels
    if 'level' in df_clean.columns:
        logger.info("Normalizing log levels...")
        
        # Mapping dictionary for common level variations
        level_mapping = {
            'warn': 'WARNING',
            'err': 'ERROR', 
            'fatal': 'CRITICAL',
            'dbg': 'DEBUG',
            'information': 'INFO',
            'informational': 'INFO'
        }
        
        # Priority order for multiple levels
        priority_order = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
        
        def normalize_level(level_str):
            if pd.isna(level_str):
                return 'OTHER'
            
            level_str = str(level_str).upper().strip()
            
            # Split on non-alphanumeric characters
            tokens = re.split(r'[^a-zA-Z0-9]+', level_str)
            tokens = [token.strip() for token in tokens if token.strip()]
            
            if not tokens:
                return 'OTHER'
            
            # Map tokens to canonical levels
            canonical_tokens = []
            for token in tokens:
                if token in priority_order:
                    canonical_tokens.append(token)
                elif token in level_mapping:
                    canonical_tokens.append(level_mapping[token])
                else:
                    # Try to match by prefix or similar patterns
                    for canon_level in priority_order:
                        if canon_level.startswith(token) or token.startswith(canon_level[:3]):
                            canonical_tokens.append(canon_level)
                            break
                    else:
                        canonical_tokens.append('OTHER')
            
            # Return highest priority level found
            for level in priority_order:
                if level in canonical_tokens:
                    return level
            
            return 'OTHER'
        
        df_clean['level'] = df_clean['level'].apply(normalize_level)
    
    # 5. Process indicator_tags
    if 'indicator_tags' in df_clean.columns:
        logger.info("Processing indicator tags...")
        
        def split_tags(tags_str):
            if pd.isna(tags_str):
                return []
            
            # Split on common separators: ; , | / and whitespace
            tags = re.split(r'[;,|/\s]+', str(tags_str))
            # Clean and filter tags
            tags = [tag.strip().lower() for tag in tags if tag.strip()]
            return tags
        
        df_clean['indicator_tags_list'] = df_clean['indicator_tags'].apply(split_tags)
        df_clean['indicator_tags_json'] = df_clean['indicator_tags_list'].apply(json.dumps)
    
    # 6. Validate IP addresses
    for ip_col in ['ip_src', 'ip_dst']:
        if ip_col in df_clean.columns:
            logger.info(f"Validating {ip_col}...")
            
            def validate_ip(ip_str):
                if pd.isna(ip_str):
                    return False
                try:
                    ipaddress.ip_address(str(ip_str))
                    return True
                except ValueError:
                    return False
            
            valid_col = f"{ip_col}_valid"
            df_clean[valid_col] = df_clean[ip_col].apply(validate_ip)
            
            # Mark invalid IPs as NaN
            invalid_mask = ~df_clean[valid_col]
            df_clean.loc[invalid_mask, ip_col] = pd.NA
    
    # 7. Convert line_number to nullable integer
    if 'line_number' in df_clean.columns:
        logger.info("Converting line numbers to integers...")
        df_clean['line_number'] = pd.to_numeric(df_clean['line_number'], errors='coerce').astype('Int64')
    
    # 8. Store original message and extract ports
    if 'message' in df_clean.columns:
        logger.info("Processing messages and extracting ports...")
        df_clean['message_raw'] = df_clean['message']
        
        # Extract port numbers from message
        def extract_port(message):
            if pd.isna(message):
                return pd.NA
            
            message_str = str(message)
            
            # Pattern 1: /:PORT (e.g., /:3888)
            port_match = re.search(r'/:(\d+)', message_str)
            if port_match:
                return int(port_match.group(1))
            
            # Pattern 2: port PORT (e.g., port 12345)
            port_match = re.search(r'port\s+(\d+)', message_str, re.IGNORECASE)
            if port_match:
                return int(port_match.group(1))
            
            # Pattern 3: :PORT (e.g., :45307)
            port_match = re.search(r':(\d+)(?:\D|$)', message_str)
            if port_match:
                return int(port_match.group(1))
            
            return pd.NA
        
        df_clean['peer_port'] = df_clean['message'].apply(extract_port).astype('Int64')
    
    # 9. Deduplicate rows
    logger.info("Deduplicating rows...")
    original_count = len(df_clean)
    
    if 'line_number' in df_clean.columns and df_clean['line_number'].notna().any():
        # Use timestamp + source_file + line_number as unique key
        subset_cols = ['timestamp', 'source_file', 'line_number']
        # Ensure all subset columns exist and have data
        available_cols = [col for col in subset_cols if col in df_clean.columns and df_clean[col].notna().any()]
        if available_cols:
            df_clean = df_clean.drop_duplicates(subset=available_cols)
        else:
            df_clean = df_clean.drop_duplicates()
    else:
        # Drop exact duplicates across all columns
        df_clean = df_clean.drop_duplicates()
    
    logger.info(f"Removed {original_count - len(df_clean)} duplicate rows")
    
    # 10. Sort by timestamp ascending
    if 'timestamp' in df_clean.columns:
        df_clean = df_clean.sort_values('timestamp', ascending=True).reset_index(drop=True)
    
    # 11. Add derived time columns
    if 'timestamp' in df_clean.columns and df_clean['timestamp'].notna().any():
        logger.info("Adding derived time columns...")
        df_clean['day_of_week'] = df_clean['timestamp'].dt.day_name()
        df_clean['hour_of_day'] = df_clean['timestamp'].dt.hour
        df_clean['is_weekend'] = df_clean['timestamp'].dt.dayofweek.isin([5, 6])
    
    logger.info(f"Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean

def main():
    """Main function to run the data processing pipeline"""
    try:
        # Load data
        df = load_parsed_csvs()
        if df.empty:
            logger.error("No data loaded. Exiting.")
            return
        
        # Clean and normalize
        df_clean = clean_normalize_logs(df)
        
        # Print summary information
        print(f"\nFinal DataFrame shape: {df_clean.shape}")
        print("\nData types:")
        print(df_clean.dtypes)
        print("\nFirst 5 rows:")
        print(df_clean.head(5))
        
        # Sanity checks
        if 'timestamp' in df_clean.columns:
            assert pd.api.types.is_datetime64_any_dtype(df_clean['timestamp']), "Timestamp should be datetime type"
            if df_clean['timestamp'].notna().any():
                assert df_clean['timestamp'].dt.tz is not None, "Timestamps should be timezone-aware"
        
        # Check IP validation rates
        ip_cols = [col for col in df_clean.columns if col.endswith('_valid')]
        for ip_col in ip_cols:
            valid_rate = df_clean[ip_col].mean()
            print(f"{ip_col} validation rate: {valid_rate:.2%}")
            if valid_rate < 0.5:
                logger.warning(f"Low validation rate for {ip_col}: {valid_rate:.2%}")
        
        # Print sample of extracted data
        if 'indicator_tags_list' in df_clean.columns:
            sample_tags = df_clean['indicator_tags_list'].dropna().head(3)
            print("\nSample indicator tags:")
            for tags in sample_tags:
                print(f"  {tags}")
        
        if 'peer_port' in df_clean.columns and df_clean['peer_port'].notna().any():
            sample_ports = df_clean['peer_port'].dropna().head(5)
            print("\nSample extracted ports:")
            print(sample_ports.tolist())
        
        # Save cleaned data
        output_dir = pathlib.Path("oplogs/cleaned/")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as Parquet (preferred)
        parquet_path = output_dir / "cleaned.parquet"
        df_clean.to_parquet(parquet_path, index=False)
        logger.info(f"Saved cleaned data to {parquet_path}")
        
        # Save as CSV
        csv_path = output_dir / "cleaned.csv"
        df_clean.to_csv(csv_path, index=False)
        logger.info(f"Saved cleaned data to {csv_path}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main()