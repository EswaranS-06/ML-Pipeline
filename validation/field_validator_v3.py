import re
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

class FieldValidator:
    def __init__(self):
        """Initialize the field validator with validation rules."""
        self.required_fields = {
            'timestamp': str,
            'source_file': str,
            'level': str,
            'message': str,
            'line_number': int
        }
        
        self.optional_fields = {
            'indicator_tags': str,
            'ip_src': str,
            'ip_dst': str,
            'service': str
        }
        
        self.ip_pattern = re.compile(r'^(?:\d{1,3}\.){3}\d{1,3}$')
        self.log_levels = {'ERROR', 'WARNING', 'INFO', 'DEBUG', 'CRITICAL'}

    def validate_entry(self, entry: Dict) -> bool:
        """Validate a single log entry."""
        try:
            # Check required fields
            for field, field_type in self.required_fields.items():
                if field not in entry:
                    return False
                if not isinstance(entry[field], field_type):
                    return False

            # Validate timestamp format
            if entry['timestamp'] and not self._validate_timestamp(entry['timestamp']):
                return False

            # Validate log level
            if entry['level'] and not self._validate_log_level(entry['level']):
                return False

            # Validate IP addresses
            if entry.get('ip_src') and not self._validate_ip(entry['ip_src']):
                entry['ip_src'] = ''
            if entry.get('ip_dst') and not self._validate_ip(entry['ip_dst']):
                entry['ip_dst'] = ''

            # Validate indicator tags
            if entry.get('indicator_tags'):
                entry['indicator_tags'] = self._validate_indicator_tags(entry['indicator_tags'])

            return True

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the entire DataFrame."""
        try:
            # Check required columns
            for col in self.required_fields:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Validate log levels
            df['level'] = df['level'].apply(lambda x: x if x in self.log_levels else 'INFO')

            # Clean IP addresses
            if 'ip_src' in df.columns:
                df['ip_src'] = df['ip_src'].apply(lambda x: x if self._validate_ip(x) else '')
            if 'ip_dst' in df.columns:
                df['ip_dst'] = df['ip_dst'].apply(lambda x: x if self._validate_ip(x) else '')

            # Ensure line numbers are integers
            df['line_number'] = df['line_number'].astype(int)

            # Sort by timestamp and line number
            df = df.sort_values(['timestamp', 'line_number'])

            return df

        except Exception as e:
            print(f"DataFrame validation error: {str(e)}")
            return df

    def _validate_timestamp(self, timestamp: str) -> bool:
        """Validate timestamp format."""
        try:
            if isinstance(timestamp, str):
                # Check ISO format
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except (ValueError, TypeError):
            return False

    def _validate_log_level(self, level: str) -> bool:
        """Validate log level."""
        return level in self.log_levels

    def _validate_ip(self, ip: str) -> bool:
        """Validate IP address format."""
        if not ip:
            return True
        if not self.ip_pattern.match(ip):
            return False
        # Validate each octet
        try:
            octets = ip.split('.')
            return all(0 <= int(octet) <= 255 for octet in octets)
        except ValueError:
            return False

    def _validate_indicator_tags(self, tags: str) -> str:
        """Validate and clean indicator tags."""
        if not tags:
            return ''
        # Split by semicolon, clean each tag, and rejoin
        valid_tags = []
        for tag in tags.split(';'):
            tag = tag.strip().lower()
            if tag and re.match(r'^[a-z0-9_-]+$', tag):
                valid_tags.append(tag)
        return ';'.join(valid_tags)