import configparser
from pathlib import Path
from typing import Any, Dict, Optional

class ConfigManager:
    def __init__(self, config_path: str = "drain3.ini"):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.config = self._load_config()
        self.defaults = self._set_defaults()

    def _load_config(self) -> configparser.ConfigParser:
        """Load configuration from file or create default."""
        config = configparser.ConfigParser()
        
        if Path(self.config_path).exists():
            config.read(self.config_path)
        else:
            self._create_default_config(config)
            
        return config

    def _create_default_config(self, config: configparser.ConfigParser):
        """Create default configuration."""
        config['DRAIN3'] = {
            'sim_th': '0.4',
            'depth': '4',
            'max_children': '100',
            'max_clusters': '1000',
            'extra_delimiters': '[]'
        }
        
        config['PREPROCESSING'] = {
            'remove_timestamps': 'true',
            'normalize_paths': 'true',
            'normalize_numbers': 'true',
            'normalize_whitespace': 'true'
        }
        
        config['VALIDATION'] = {
            'strict_timestamp': 'true',
            'strict_level': 'true',
            'validate_ips': 'true',
            'max_message_length': '10000'
        }
        
        config['ERROR_HANDLING'] = {
            'max_errors_per_file': '1000',
            'stop_on_critical': 'true',
            'log_detailed_errors': 'true',
            'error_log_path': 'logs/errors/parser_errors.log'
        }
        
        # Save default configuration
        with open(self.config_path, 'w') as f:
            config.write(f)

    def _set_defaults(self) -> Dict[str, Any]:
        """Set default values for configuration."""
        return {
            'sim_th': 0.4,
            'depth': 4,
            'max_children': 100,
            'max_clusters': 1000,
            'extra_delimiters': '[]',
            'remove_timestamps': True,
            'normalize_paths': True,
            'normalize_numbers': True,
            'normalize_whitespace': True,
            'strict_timestamp': True,
            'strict_level': True,
            'validate_ips': True,
            'max_message_length': 10000,
            'max_errors_per_file': 1000,
            'stop_on_critical': True,
            'log_detailed_errors': True,
            'error_log_path': 'logs/errors/parser_errors.log'
        }

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get configuration value with type conversion."""
        try:
            value = self.config.get(section, key)
            
            # Convert to appropriate type based on default value
            default = self.defaults.get(key, fallback)
            if isinstance(default, bool):
                return self.config.getboolean(section, key)
            elif isinstance(default, int):
                return self.config.getint(section, key)
            elif isinstance(default, float):
                return self.config.getfloat(section, key)
            return value
            
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback or self.defaults.get(key)

    def set(self, section: str, key: str, value: Any):
        """Set configuration value."""
        if not self.config.has_section(section):
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
        self.save()

    def save(self):
        """Save current configuration to file."""
        with open(self.config_path, 'w') as f:
            self.config.write(f)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration values as dictionary."""
        result = {}
        for section in self.config.sections():
            result[section] = {}
            for key in self.config[section]:
                result[section][key] = self.get(section, key)
        return result