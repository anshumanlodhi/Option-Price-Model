"""
Configuration management for Indian Options Pricing Model
Handles loading and validation of configuration parameters from config.yaml
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Please install it with: pip install pyyaml")
    yaml = None
from datetime import datetime, timedelta


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors"""
    pass


class Config:
    """
    Main configuration class for the options pricing project
    Loads parameters from config.yaml and provides structured access
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with automatic YAML loading
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        # Set up project paths - try multiple possible locations
        possible_roots = [
            Path(__file__).parent.parent.parent,  # src/config/settings.py -> project_root
            Path.cwd(),  # Current working directory
            Path.cwd().parent,  # Parent of current directory
        ]
        
        # Find the project root by looking for config.yaml
        self.PROJECT_ROOT = None
        for root in possible_roots:
            if (root / "config.yaml").exists():
                self.PROJECT_ROOT = root
                break
        
        if self.PROJECT_ROOT is None:
            # Fallback to the first option
            self.PROJECT_ROOT = possible_roots[0]
        
        self.CONFIG_FILE = config_path or self.PROJECT_ROOT / "config.yaml"
        
        # Initialize configuration storage
        self._config_data: Dict[str, Any] = {}
        self._loaded = False
        
        # Load configuration automatically
        self.load_configuration()
        
        # Create data directories
        self._ensure_directories()
        
        # Set up logging
        self._setup_logging()
    
    def load_configuration(self) -> None:
        """Load configuration from YAML file with error handling"""
        try:
            if yaml is None:
                raise ConfigurationError("PyYAML is not installed. Please install it with: pip install pyyaml")
            
            if not self.CONFIG_FILE.exists():
                raise ConfigurationError(f"Configuration file not found: {self.CONFIG_FILE}")
            
            with open(self.CONFIG_FILE, 'r', encoding='utf-8') as file:
                self._config_data = yaml.safe_load(file)
            
            self._loaded = True
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def reload_configuration(self) -> None:
        """Reload configuration from file - useful during development"""
        self._config_data = {}
        self._loaded = False
        self.load_configuration()
        logging.info("Configuration reloaded")
    
    def _get_config_value(self, path: str, default: Any = None, required: bool = False) -> Any:
        """
        Helper method to get nested configuration values
        
        Args:
            path: Dot-separated path to configuration value (e.g., 'data.start_date')
            default: Default value if path not found
            required: Whether to raise exception if value not found
        
        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        value = self._config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            if required:
                raise ConfigurationError(f"Required configuration parameter '{path}' not found")
            if default is None:
                warnings.warn(f"Configuration parameter '{path}' not found, using None")
            return default
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.storage_paths.base,
            self.storage_paths.raw,
            self.storage_paths.processed,
            self.storage_paths.external,
            Path("logs")  # For logging
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Configure logging based on configuration parameters"""
        log_level = getattr(logging, self.logging.level.upper())
        log_format = self.logging.format
        log_file = Path(self.logging.file)
        
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )


    
    # =============================================================================
    # DATA CONFIGURATION PROPERTIES
    # =============================================================================
    
    @property
    def data(self) -> 'DataConfig':
        """Access to data collection configuration"""
        return DataConfig(self._config_data.get('data', {}), self.PROJECT_ROOT)
    
    @property
    def options(self) -> 'OptionsConfig':
        """Access to options specifications configuration"""
        return OptionsConfig(self._config_data.get('options', {}))
    
    @property
    def sources(self) -> 'SourcesConfig':
        """Access to data sources configuration"""
        return SourcesConfig(self._config_data.get('sources', {}))
    
    @property
    def storage_paths(self) -> 'StoragePathsConfig':
        """Access to storage paths configuration"""
        storage_config = self._config_data.get('storage', {})
        return StoragePathsConfig(storage_config, self.PROJECT_ROOT)
    
    @property
    def storage(self) -> 'StorageConfig':
        """Access to storage configuration"""
        return StorageConfig(self._config_data.get('storage', {}))
    
    @property
    def model(self) -> 'ModelConfig':
        """Access to model parameters configuration"""
        return ModelConfig(self._config_data.get('model', {}))
    
    @property
    def calculations(self) -> 'CalculationsConfig':
        """Access to calculation parameters configuration"""
        return CalculationsConfig(self._config_data.get('calculations', {}))
    
    @property
    def validation(self) -> 'ValidationConfig':
        """Access to validation rules configuration"""
        return ValidationConfig(self._config_data.get('validation', {}))
    
    @property
    def development(self) -> 'DevelopmentConfig':
        """Access to development settings configuration"""
        return DevelopmentConfig(self._config_data.get('development', {}))
    
    @property
    def logging(self) -> 'LoggingConfig':
        """Access to logging configuration"""
        return LoggingConfig(self._config_data.get('logging', {}))
    
    @property
    def alerts(self) -> 'AlertsConfig':
        """Access to alerts configuration"""
        return AlertsConfig(self._config_data.get('alerts', {}))


# =============================================================================
# CONFIGURATION SECTION CLASSES
# =============================================================================

class DataConfig:
    """Configuration for data collection parameters"""
    
    def __init__(self, config: Dict[str, Any], project_root: Path):
        self._config = config
        self._project_root = project_root
    
    @property
    def start_date(self) -> str:
        return self._config.get('start_date', '2022-01-01')
    
    @property
    def end_date(self) -> str:
        return self._config.get('end_date', '2024-12-31')
    
    @property
    def symbols(self) -> Dict[str, str]:
        return self._config.get('symbols', {
            'primary': 'NIFTY',
            'nse_symbol': 'NIFTY',
            'yahoo_symbol': '^NSEI'
        })
    
    @property
    def frequency(self) -> str:
        return self._config.get('frequency', 'daily')
    
    @property
    def timezone(self) -> str:
        return self._config.get('timezone', 'Asia/Kolkata')


class OptionsConfig:
    """Configuration for options specifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def expiry_type(self) -> str:
        return self._config.get('expiry_type', 'monthly')
    
    @property
    def expiry_day(self) -> str:
        return self._config.get('expiry_day', 'last_thursday')
    
    @property
    def strike_range(self) -> int:
        return self._config.get('strike_range', 10)
    
    @property
    def strike_spacing(self) -> int:
        return self._config.get('strike_spacing', 50)
    
    @property
    def option_types(self) -> List[str]:
        return self._config.get('option_types', ['CE', 'PE'])
    
    @property
    def min_dte(self) -> int:
        return self._config.get('min_dte', 1)
    
    @property
    def max_dte(self) -> int:
        return self._config.get('max_dte', 60)


class SourcesConfig:
    """Configuration for data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def primary(self) -> str:
        return self._config.get('primary', 'nsepy')
    
    @property
    def backup(self) -> List[str]:
        return self._config.get('backup', ['yfinance', 'pandas_datareader'])
    
    @property
    def risk_free_rate(self) -> Dict[str, str]:
        return self._config.get('risk_free_rate', {
            'source': 'rbi_historical',
            'api_url': 'https://www.rbi.org.in',
            'parameter': 'repo_rate'
        })
    
    @property
    def volatility_index(self) -> Dict[str, str]:
        return self._config.get('volatility_index', {
            'symbol': 'INDIAVIX',
            'source': 'nsepy'
        })


class StoragePathsConfig:
    """Configuration for storage paths with automatic path resolution"""
    
    def __init__(self, config: Dict[str, Any], project_root: Path):
        self._config = config
        self._project_root = project_root
        self._paths = config.get('paths', {})
    
    @property
    def base(self) -> Path:
        return self._project_root / self._paths.get('base', 'data')
    
    @property
    def raw(self) -> Path:
        return self._project_root / self._paths.get('raw', 'data/raw')
    
    @property
    def processed(self) -> Path:
        return self._project_root / self._paths.get('processed', 'data/processed')
    
    @property
    def external(self) -> Path:
        return self._project_root / self._paths.get('external', 'data/external')


class StorageConfig:
    """Configuration for storage settings"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def format(self) -> str:
        return self._config.get('format', 'csv')
    
    @property
    def encoding(self) -> str:
        return self._config.get('encoding', 'utf-8')
    
    @property
    def frequency(self) -> str:
        return self._config.get('frequency', 'monthly')
    
    @property
    def naming(self) -> Dict[str, str]:
        return self._config.get('naming', {
            'options_data': 'nifty_options_{year}_{month:02d}.csv',
            'spot_data': 'nifty_spot_{year}_{month:02d}.csv',
            'volatility_data': 'india_vix_{year}_{month:02d}.csv',
            'risk_free_rates': 'rbi_repo_rates_{year}.csv'
        })
    
    @property
    def compression(self) -> Optional[str]:
        return self._config.get('compression')


class ModelConfig:
    """Configuration for model parameters"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def volatility(self) -> Dict[str, Union[int, str]]:
        return self._config.get('volatility', {
            'window': 30,
            'method': 'log_returns',
            'annualization_factor': 252
        })
    
    @property
    def risk_free_rate(self) -> Dict[str, Union[float, str]]:
        return self._config.get('risk_free_rate', {
            'default': 0.065,
            'interpolation': 'linear'
        })
    
    @property
    def dividend_yield(self) -> float:
        return self._config.get('dividend_yield', 0.0)
    
    @property
    def precision(self) -> Dict[str, Union[int, str]]:
        return self._config.get('precision', {
            'decimal_places': 6,
            'float_format': 'f64'
        })


class CalculationsConfig:
    """Configuration for calculation parameters"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def greeks(self) -> Dict[str, Union[str, float]]:
        return self._config.get('greeks', {
            'method': 'analytical',
            'numerical_method': 'finite_difference',
            'bump_size': 0.01
        })
    
    @property
    def black_scholes(self) -> Dict[str, Union[str, bool]]:
        return self._config.get('black_scholes', {
            'option_style': 'european',
            'dividend_adjustment': True,
            'early_exercise': False
        })
    
    @property
    def monte_carlo(self) -> Dict[str, int]:
        return self._config.get('monte_carlo', {
            'simulations': 10000,
            'time_steps': 252,
            'random_seed': 42
        })


class ValidationConfig:
    """Configuration for validation rules"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def quality_checks(self) -> Dict[str, Union[float, int]]:
        return self._config.get('quality_checks', {
            'max_bid_ask_spread': 0.5,
            'min_volume': 1,
            'max_missing_data': 0.05
        })
    
    @property
    def options_validation(self) -> Dict[str, Union[int, float]]:
        return self._config.get('options_validation', {
            'max_strike_range': 20,
            'min_time_value': 0.01,
            'max_moneyness': 2.0
        })
    
    @property
    def volatility_validation(self) -> Dict[str, float]:
        return self._config.get('volatility_validation', {
            'min_volatility': 0.05,
            'max_volatility': 1.0,
            'outlier_threshold': 3.0
        })


class DevelopmentConfig:
    """Configuration for development settings"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def debug(self) -> bool:
        return self._config.get('debug', True)
    
    @property
    def verbose_logging(self) -> bool:
        return self._config.get('verbose_logging', True)
    
    @property
    def parallel_processing(self) -> bool:
        return self._config.get('parallel_processing', False)
    
    @property
    def batch_size(self) -> int:
        return self._config.get('batch_size', 1000)
    
    @property
    def cache_data(self) -> bool:
        return self._config.get('cache_data', True)
    
    @property
    def cache_duration(self) -> int:
        return self._config.get('cache_duration', 3600)
    
    @property
    def api_rate_limit(self) -> float:
        return self._config.get('api_rate_limit', 1.0)


class LoggingConfig:
    """Configuration for logging settings"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def level(self) -> str:
        return self._config.get('level', 'INFO')
    
    @property
    def format(self) -> str:
        return self._config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    @property
    def file(self) -> str:
        return self._config.get('file', 'logs/options_pricing.log')
    
    @property
    def max_file_size(self) -> str:
        return self._config.get('max_file_size', '10MB')
    
    @property
    def backup_count(self) -> int:
        return self._config.get('backup_count', 5)


class AlertsConfig:
    """Configuration for alerts and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
    
    @property
    def data_quality(self) -> Dict[str, float]:
        return self._config.get('data_quality', {
            'missing_data_threshold': 0.1,
            'price_anomaly_threshold': 0.2,
            'volume_anomaly_threshold': 0.5
        })
    
    @property
    def model_validation(self) -> Dict[str, float]:
        return self._config.get('model_validation', {
            'pricing_error_threshold': 0.05,
            'greeks_error_threshold': 0.1
        })


# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

# Create global configuration instance for easy access throughout the project
try:
    config = Config()
except ConfigurationError as e:
    warnings.warn(f"Failed to load configuration: {e}")
    config = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config() -> Config:
    """
    Get the global configuration instance
    
    Returns:
        Config: Global configuration instance
    
    Raises:
        ConfigurationError: If configuration is not loaded
    """
    global config
    
    if config is None:
        try:
            config = Config()
            return config
        except Exception as e:
            # Create a minimal fallback configuration
            config = Config.__new__(Config)
            config._config_data = {
                'data': {
                    'start_date': '2022-01-01',
                    'end_date': '2024-12-31',
                    'symbols': {'primary': 'NIFTY', 'nse_symbol': 'NIFTY', 'yahoo_symbol': '^NSEI'},
                    'frequency': 'daily',
                    'timezone': 'Asia/Kolkata'
                },
                'options': {
                    'expiry_type': 'monthly',
                    'expiry_day': 'last_thursday',
                    'strike_range': 10,
                    'strike_spacing': 50,
                    'option_types': ['CE', 'PE'],
                    'min_dte': 1,
                    'max_dte': 60
                },
                'sources': {
                    'primary': 'nsepy',
                    'backup': ['yfinance', 'pandas_datareader'],
                    'risk_free_rate': {'source': 'rbi_historical', 'api_url': 'https://www.rbi.org.in', 'parameter': 'repo_rate'},
                    'volatility_index': {'symbol': 'INDIAVIX', 'source': 'nsepy'}
                },
                'storage': {
                    'format': 'csv',
                    'encoding': 'utf-8',
                    'frequency': 'monthly',
                    'paths': {'base': 'data', 'raw': 'data/raw', 'processed': 'data/processed', 'external': 'data/external'},
                    'naming': {'options_data': 'nifty_options_{year}_{month:02d}.csv', 'spot_data': 'nifty_spot_{year}_{month:02d}.csv'},
                    'compression': None
                },
                'model': {
                    'volatility': {'window': 30, 'method': 'log_returns', 'annualization_factor': 252},
                    'risk_free_rate': {'default': 0.065, 'interpolation': 'linear'},
                    'dividend_yield': 0.0,
                    'precision': {'decimal_places': 6, 'float_format': 'f64'}
                },
                'calculations': {
                    'greeks': {'method': 'analytical', 'numerical_method': 'finite_difference', 'bump_size': 0.01},
                    'black_scholes': {'option_style': 'european', 'dividend_adjustment': True, 'early_exercise': False},
                    'monte_carlo': {'simulations': 10000, 'time_steps': 252, 'random_seed': 42}
                },
                'validation': {
                    'quality_checks': {'max_bid_ask_spread': 0.5, 'min_volume': 1, 'max_missing_data': 0.05},
                    'options_validation': {'max_strike_range': 20, 'min_time_value': 0.01, 'max_moneyness': 2.0},
                    'volatility_validation': {'min_volatility': 0.05, 'max_volatility': 1.0, 'outlier_threshold': 3.0}
                },
                'development': {
                    'debug': True,
                    'verbose_logging': True,
                    'parallel_processing': False,
                    'batch_size': 1000,
                    'cache_data': True,
                    'cache_duration': 3600,
                    'api_rate_limit': 1.0
                },
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file': 'logs/options_pricing.log',
                    'max_file_size': '10MB',
                    'backup_count': 5
                },
                'alerts': {
                    'data_quality': {'missing_data_threshold': 0.1, 'price_anomaly_threshold': 0.2, 'volume_anomaly_threshold': 0.5},
                    'model_validation': {'pricing_error_threshold': 0.05, 'greeks_error_threshold': 0.1}
                }
            }
            config._loaded = True
            config.PROJECT_ROOT = Path.cwd()
            config.CONFIG_FILE = Path.cwd() / "config.yaml"
            return config
    
    return config


def reload_config() -> Config:
    """
    Reload the global configuration instance
    
    Returns:
        Config: Newly loaded configuration instance
    """
    global config
    config = Config()
    return config

    import os
    from pathlib import Path



if __name__ == "__main__":
    # Test configuration loading
    try:
        test_config = Config()
        print("Configuration loaded successfully!")
        print(f"Project root: {test_config.PROJECT_ROOT}")
        print(f"Data start date: {test_config.data.start_date}")
        print(f"Options strike range: {test_config.options.strike_range}")
        print(f"Primary data source: {test_config.sources.primary}")
        print(f"Raw data path: {test_config.storage_paths.raw}")
    except Exception as e:
        print(f"Configuration test failed: {e}")
