import re
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

try:
    import chardet
except ImportError:
    # Fallback for environments where chardet might not be available
    chardet = None
    print("Warning: chardet module not available. Character encoding detection will be limited.")

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with enhanced patterns."""
        self.timestamp_patterns = [
            # Windows CBS style (prioritized)
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\s*\w+)?',
            # Standard ISO format
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',
            # Apache style
            r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',
            # Windows style
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}',
            # Unix style
            r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
            # Android style
            r'\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3}',
            # Custom format 1
            r'\d{8}-\d{2}:\d{2}:\d{2}:\d{3}',
            # Custom format 2
            r'\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}',
            # Windows Update style
            r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}:\d{3}'
        ]
        
        self.level_patterns = {
            'ERROR': r'\b(?:ERROR|ERR|SEVERE|FATAL)\b',
            'WARNING': r'\b(?:WARN(?:ING)?|ATTENTION)\b',
            'INFO': r'\b(?:INFO|INFORMATION|NOTICE)\b',
            'DEBUG': r'\b(?:DEBUG|TRACE|FINE)\b',
            'CRITICAL': r'\b(?:CRITICAL|FATAL|EMERG)\b'
        }
        
        self.ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        self.service_patterns = [
            r'service=(\w+)',
            r'(\w+)Service',
            r'(\w+)\.service',
            r'(\w+)_service'
        ]
        
        self.indicator_patterns = {
            'error': r'\b(?:error|failed|failure|invalid|exception)\b',
            'warning': r'\b(?:warning|warn|attention)\b',
            'success': r'\b(?:success|successful|completed|ok)\b',
            'info': r'\b(?:info|information|notice)\b',
            'security': r'\b(?:security|auth|authentication|permission)\b'
        }

    def read_file(self, file_path: str) -> List[str]:
        """Read a file with automatic encoding detection."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read a sample of the file to detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB
            if chardet:
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            else:
                # Fallback: try common encodings
                encoding = 'utf-8'
                # Try to detect by checking if it's valid UTF-8
                try:
                    raw_data.decode('utf-8')
                except UnicodeDecodeError:
                    # Try other common encodings
                    encoding = 'latin-1'  # Fallback to latin-1
                    
        # Read the full file with detected encoding
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            # Fallback to utf-8 if detection fails
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return [line.strip() for line in f if line.strip()]

    def preprocess_line(self, line: str) -> str:
        """Preprocess a single log line."""
        # Remove null bytes and control characters
        line = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', line)
        
        # Normalize whitespace
        line = re.sub(r'\s+', ' ', line)
        
        # Remove common noise patterns
        line = re.sub(r'\[pid:\d+\]', '', line)
        line = re.sub(r'\[tid:\d+\]', '', line)
        
        return line.strip()

    def extract_timestamp(self, log: str) -> Optional[str]:
        """Extract and normalize timestamp from log line."""
        for pattern in self.timestamp_patterns:
            match = re.search(pattern, log)
            if match:
                timestamp = match.group(0)
                try:
                    # Convert to ISO format if possible
                    dt = self._parse_timestamp(timestamp)
                    return dt.isoformat() if dt else timestamp
                except ValueError:
                    continue
        return None

    def _parse_timestamp(self, timestamp: str) -> Optional[datetime]:
        """Parse timestamp string to datetime object."""
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%d %H:%M:%S,%f',
            '%d/%b/%Y:%H:%M:%S',
            '%b %d %H:%M:%S',
            '%m-%d %H:%M:%S.%f',
            '%Y%m%d-%H:%M:%S:%f',
            '%Y.%m.%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S',  # Windows CBS format
            '%Y.%m.%d %H:%M:%S:%f'  # Windows Update format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        return None

    def extract_log_level(self, log: str) -> str:
        """Extract log level from log line."""
        log_upper = log.upper()
        for level, pattern in self.level_patterns.items():
            if re.search(pattern, log_upper):
                return level
        return 'INFO'  # Default level

    def extract_ips(self, log: str) -> Dict[str, str]:
        """Extract source and destination IPs from log line."""
        ips = re.findall(self.ip_pattern, log)
        result = {'source': '', 'destination': ''}
        
        if len(ips) >= 2:
            result['source'] = ips[0]
            result['destination'] = ips[1]
        elif len(ips) == 1:
            result['source'] = ips[0]
            
        return result

    def extract_service(self, log: str) -> str:
        """Extract service name from log line."""
        for pattern in self.service_patterns:
            match = re.search(pattern, log, re.IGNORECASE)
            if match:
                return match.group(1)
        return ''

    def extract_indicators(self, log: str) -> List[str]:
        """Extract indicator tags from log line."""
        indicators = []
        log_lower = log.lower()
        
        for indicator, pattern in self.indicator_patterns.items():
            if re.search(pattern, log_lower):
                indicators.append(indicator)
                
        return indicators

    def clean_message(self, log: str) -> str:
        """Clean and normalize the log message."""
        # Handle Windows CBS logs specially
        if 'CBS' in log:
            # Extract the actual message after the timestamp and level
            match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}, \w+ (.*)', log)
            if match:
                return match.group(1).strip()

        # Remove known timestamp patterns
        for pattern in self.timestamp_patterns:
            log = re.sub(pattern, '', log)
            
        # Remove log level indicators
        for pattern in self.level_patterns.values():
            log = re.sub(pattern, '', log, flags=re.IGNORECASE)
            
        # Remove IP addresses
        log = re.sub(self.ip_pattern, '', log)
        
        # Normalize whitespace
        log = re.sub(r'\s+', ' ', log)
        
        return log.strip()