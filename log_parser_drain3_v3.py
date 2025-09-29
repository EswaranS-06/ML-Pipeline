import sys
import json
import logging
import re
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Import our custom modules
from preprocessing.text_preprocessor_v3 import TextPreprocessor
from validation.field_validator_v3 import FieldValidator
from utils.error_handler_v3 import ErrorHandler
from utils.config_manager_v3 import ConfigManager

class LogParserV3:
    def __init__(self, config_path: str = "drain3.ini"):
        """Initialize the enhanced log parser with improved components."""
        self.config = ConfigManager(config_path)
        self.preprocessor = TextPreprocessor()
        self.validator = FieldValidator()
        self.error_handler = ErrorHandler()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("LogParserV3")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    def parse_log_file(self, file_path: str) -> Tuple[pd.DataFrame, List[dict]]:
        """Parse a log file with enhanced error handling and validation."""
        try:
            self.logger.info(f"Starting to parse file: {file_path}")
            
            # Read and preprocess the file
            raw_logs = self.preprocessor.read_file(file_path)
            preprocessed_logs = [
                self.preprocessor.preprocess_line(line) 
                for line in raw_logs
            ]

            # Parse and validate each log entry
            parsed_logs = []
            errors = []
            
            for idx, log in enumerate(preprocessed_logs, 1):
                try:
                    parsed_entry = self.parse_log_entry(log, file_path, idx)
                    if self.validator.validate_entry(parsed_entry):
                        parsed_logs.append(parsed_entry)
                    else:
                        errors.append({
                            'line': idx,
                            'error': 'Validation failed',
                            'content': log
                        })
                except Exception as e:
                    self.error_handler.handle_parsing_error(e, idx, log)
                    errors.append({
                        'line': idx,
                        'error': str(e),
                        'content': log
                    })

            # Convert to DataFrame
            df = pd.DataFrame(parsed_logs)
            
            # Post-processing validation
            df = self.validator.validate_dataframe(df)
            
            self.logger.info(f"Successfully parsed {len(parsed_logs)} logs with {len(errors)} errors")
            return df, errors

        except Exception as e:
            self.error_handler.handle_critical_error(e)
            raise

    def parse_log_entry(self, log: str, source_file: str, line_number: int) -> Dict:
        """Parse a single log entry with enhanced field extraction."""
        try:
            # Special handling for Windows CBS logs
            cbs_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), (\w+) (.*)', log)
            if 'CBS' in log and cbs_match:
                timestamp = cbs_match.group(1)
                level = cbs_match.group(2).upper()
                message = cbs_match.group(3)
                
                # Extract additional fields from message
                ips = self.preprocessor.extract_ips(message)
                indicators = self.preprocessor.extract_indicators(message)
                
                entry = {
                    'timestamp': timestamp,
                    'source_file': Path(source_file).name,
                    'level': level,
                    'indicator_tags': ';'.join(indicators) if indicators else '',
                    'ip_src': ips.get('source', ''),
                    'ip_dst': ips.get('destination', ''),
                    'service': 'CBS',  # Fixed service name for CBS logs
                    'message': message,
                    'line_number': line_number
                }
            else:
                # Standard log parsing
                timestamp = self.preprocessor.extract_timestamp(log)
                level = self.preprocessor.extract_log_level(log)
                ips = self.preprocessor.extract_ips(log)
                service = self.preprocessor.extract_service(log)
                indicators = self.preprocessor.extract_indicators(log)
                
                # Clean and structure the message
                message = self.preprocessor.clean_message(log)
                
                # Create the parsed entry
                entry = {
                    'timestamp': timestamp,
                    'source_file': Path(source_file).name,
                    'level': level,
                    'indicator_tags': ';'.join(indicators) if indicators else '',
                    'ip_src': ips.get('source', ''),
                    'ip_dst': ips.get('destination', ''),
                    'service': service,
                    'message': message,
                    'line_number': line_number
                }
            
            # Validate the entry
            if not self.validator.validate_entry(entry):
                raise ValueError("Invalid log entry format")
                
            return entry
            
        except Exception as e:
            self.error_handler.handle_entry_error(e, line_number, log)
            raise

    def save_results(self, df: pd.DataFrame, output_dir: str, filename: str, errors: List[dict]):
        """Save parsing results and error log."""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save parsed logs in both CSV and JSON formats
            csv_path = output_path / f"csv/{filename}.csv"
            json_path = output_path / f"json/{filename}.json"
            error_path = output_path / f"errors/{filename}_errors.json"
            
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with error handling
            try:
                df.to_csv(csv_path, index=False)
                df.to_json(json_path, orient='records', lines=True)
                
                if errors:
                    with open(error_path, 'w') as f:
                        json.dump(errors, f, indent=2)
                
                self.logger.info(f"Results saved successfully: {csv_path}")
            except Exception as e:
                self.error_handler.handle_save_error(e)
                raise
                
        except Exception as e:
            self.error_handler.handle_critical_error(e)
            raise

def main():
    """Main function to run the parser."""
    parser = LogParserV3()
    
    try:
        # Process each log file in the input directory
        input_dir = Path("logs")
        output_dir = Path("oplogs")
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        for log_file in input_dir.glob("*.log"):
            try:
                # Parse the log file
                df, errors = parser.parse_log_file(str(log_file))
                
                # Save results
                filename = log_file.stem
                parser.save_results(df, str(output_dir), filename, errors)
                
            except Exception as e:
                parser.error_handler.handle_file_error(e, str(log_file))
                continue
                
    except Exception as e:
        parser.error_handler.handle_critical_error(e)
        sys.exit(1)

if __name__ == "__main__":
    main()