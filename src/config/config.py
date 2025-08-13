"""
Configuration Management for ETH/FDUSD Trading Bot
Centralized configuration with environment variable support
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class APIConfig:
    """API configuration"""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    testnet: bool = True
    
    def __post_init__(self):
        # Load from environment if not provided
        if not self.binance_api_key:
            self.binance_api_key = os.getenv('BINANCE_API_KEY', '')
        if not self.binance_api_secret:
            self.binance_api_secret = os.getenv('BINANCE_API_SECRET', '')


@dataclass
class TradingConfig:
    """Trading configuration"""
    symbol: str = "ETHFDUSD"
    base_asset: str = "ETH"
    quote_asset: str = "FDUSD"
    
    # Risk Management
    max_position_size: float = 0.1  # 10% of account
    risk_per_trade: float = 0.02    # 2% risk per trade
    max_daily_loss: float = 0.05    # 5% max daily loss
    max_positions: int = 1          # Maximum concurrent positions
    
    # Signal Parameters
    min_signal_confidence: float = 70.0
    signal_timeout: int = 300       # 5 minutes between signals
    
    # Order Management
    order_timeout: int = 30         # seconds
    max_slippage: float = 0.001     # 0.1%
    use_limit_orders: bool = False  # Use market orders by default
    
    # Position Management
    partial_take_profit: bool = True
    tp_levels: list = None          # Take profit levels (percentages)
    trailing_stop: bool = False
    trailing_stop_distance: float = 0.02  # 2%
    
    def __post_init__(self):
        if self.tp_levels is None:
            self.tp_levels = [0.25, 0.5, 0.25]  # 25%, 50%, 25% at different levels


@dataclass
class AnalysisConfig:
    """Analysis engine configuration"""
    analysis_interval: int = 60     # seconds
    min_data_points: int = 100
    lookback_period: int = 200
    
    # Mathematical Model Parameters
    cci_threshold: float = 2.5      # Capitulation Confluence Index
    ddm_threshold: float = 3.0      # Distribution Detection Matrix
    
    # Technical Indicator Parameters
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_oversold: float = 20
    stoch_overbought: float = 80
    
    bb_period: int = 20
    bb_std: float = 2.0
    
    atr_period: int = 14
    
    # Volume Analysis
    volume_sma_period: int = 20
    volume_spike_threshold: float = 1.5
    
    # Chaos Theory Parameters
    fractal_dimension_window: int = 50
    lyapunov_window: int = 50
    hurst_window: int = 100


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration"""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: str = "trading_bot.log"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    
    # Performance Monitoring
    performance_update_interval: int = 300  # 5 minutes
    status_log_interval: int = 300          # 5 minutes
    
    # Alerts
    enable_alerts: bool = True
    alert_on_trade: bool = True
    alert_on_error: bool = True
    alert_on_daily_loss: bool = True
    
    # External Notifications (optional)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""
    email_smtp_server: str = ""
    email_username: str = ""
    email_password: str = ""
    email_recipients: list = None
    
    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_balance: float = 10000.0
    commission_rate: float = 0.001  # 0.1%
    slippage: float = 0.0005       # 0.05%
    
    # Data source
    data_source: str = "binance"    # binance, file, database
    data_file_path: str = ""
    timeframe: str = "15m"
    
    # Analysis
    generate_report: bool = True
    save_trades: bool = True
    plot_results: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration for data storage"""
    enabled: bool = False
    db_type: str = "sqlite"         # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "trading_bot"
    username: str = ""
    password: str = ""
    
    # SQLite specific
    sqlite_file: str = "trading_bot.db"
    
    # Connection pool
    pool_size: int = 5
    max_overflow: int = 10


class ConfigManager:
    """
    Centralized configuration manager
    Handles loading, saving, and validation of all configurations
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.logger = self._setup_logging()
        
        # Configuration objects
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.analysis = AnalysisConfig()
        self.monitoring = MonitoringConfig()
        self.backtest = BacktestConfig()
        self.database = DatabaseConfig()
        
        # Load configuration if file exists
        if self.config_file.exists():
            self.load_config()
        else:
            self.logger.info("No config file found, using defaults")
            self.save_config()  # Save default configuration
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for config manager"""
        logger = logging.getLogger('ConfigManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration objects
            if 'api' in config_data:
                self.api = APIConfig(**config_data['api'])
            
            if 'trading' in config_data:
                self.trading = TradingConfig(**config_data['trading'])
            
            if 'analysis' in config_data:
                self.analysis = AnalysisConfig(**config_data['analysis'])
            
            if 'monitoring' in config_data:
                self.monitoring = MonitoringConfig(**config_data['monitoring'])
            
            if 'backtest' in config_data:
                self.backtest = BacktestConfig(**config_data['backtest'])
            
            if 'database' in config_data:
                self.database = DatabaseConfig(**config_data['database'])
            
            self.logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            config_data = {
                'api': asdict(self.api),
                'trading': asdict(self.trading),
                'analysis': asdict(self.analysis),
                'monitoring': asdict(self.monitoring),
                'backtest': asdict(self.backtest),
                'database': asdict(self.database)
            }
            
            # Remove sensitive data from saved config
            config_data['api']['binance_api_key'] = ""
            config_data['api']['binance_api_secret'] = ""
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def validate_config(self) -> Dict[str, list]:
        """Validate configuration and return any errors"""
        errors = {
            'api': [],
            'trading': [],
            'analysis': [],
            'monitoring': [],
            'backtest': [],
            'database': []
        }
        
        # Validate API configuration
        if not self.api.binance_api_key:
            errors['api'].append("Binance API key is required")
        if not self.api.binance_api_secret:
            errors['api'].append("Binance API secret is required")
        
        # Validate trading configuration
        if self.trading.max_position_size <= 0 or self.trading.max_position_size > 1:
            errors['trading'].append("Max position size must be between 0 and 1")
        
        if self.trading.risk_per_trade <= 0 or self.trading.risk_per_trade > 0.1:
            errors['trading'].append("Risk per trade must be between 0 and 0.1 (10%)")
        
        if self.trading.max_daily_loss <= 0 or self.trading.max_daily_loss > 0.2:
            errors['trading'].append("Max daily loss must be between 0 and 0.2 (20%)")
        
        if self.trading.min_signal_confidence < 50 or self.trading.min_signal_confidence > 100:
            errors['trading'].append("Min signal confidence must be between 50 and 100")
        
        # Validate analysis configuration
        if self.analysis.min_data_points < 50:
            errors['analysis'].append("Min data points should be at least 50")
        
        if self.analysis.lookback_period < self.analysis.min_data_points:
            errors['analysis'].append("Lookback period should be >= min data points")
        
        # Validate monitoring configuration
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.monitoring.log_level not in valid_log_levels:
            errors['monitoring'].append(f"Log level must be one of: {valid_log_levels}")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display"""
        return {
            'api': {
                'testnet': self.api.testnet,
                'has_credentials': bool(self.api.binance_api_key and self.api.binance_api_secret)
            },
            'trading': {
                'symbol': self.trading.symbol,
                'max_position_size': f"{self.trading.max_position_size * 100}%",
                'risk_per_trade': f"{self.trading.risk_per_trade * 100}%",
                'max_daily_loss': f"{self.trading.max_daily_loss * 100}%",
                'min_signal_confidence': f"{self.trading.min_signal_confidence}%"
            },
            'analysis': {
                'analysis_interval': f"{self.analysis.analysis_interval}s",
                'min_data_points': self.analysis.min_data_points,
                'cci_threshold': self.analysis.cci_threshold,
                'ddm_threshold': self.analysis.ddm_threshold
            },
            'monitoring': {
                'log_level': self.monitoring.log_level,
                'alerts_enabled': self.monitoring.enable_alerts,
                'log_to_file': self.monitoring.log_to_file
            }
        }
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update specific configuration section"""
        try:
            if section == 'api':
                for key, value in updates.items():
                    if hasattr(self.api, key):
                        setattr(self.api, key, value)
            elif section == 'trading':
                for key, value in updates.items():
                    if hasattr(self.trading, key):
                        setattr(self.trading, key, value)
            elif section == 'analysis':
                for key, value in updates.items():
                    if hasattr(self.analysis, key):
                        setattr(self.analysis, key, value)
            elif section == 'monitoring':
                for key, value in updates.items():
                    if hasattr(self.monitoring, key):
                        setattr(self.monitoring, key, value)
            elif section == 'backtest':
                for key, value in updates.items():
                    if hasattr(self.backtest, key):
                        setattr(self.backtest, key, value)
            elif section == 'database':
                for key, value in updates.items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            else:
                self.logger.error(f"Unknown configuration section: {section}")
                return False
            
            self.logger.info(f"Updated {section} configuration")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    def reset_to_defaults(self, section: str = None) -> bool:
        """Reset configuration to defaults"""
        try:
            if section is None or section == 'all':
                self.api = APIConfig()
                self.trading = TradingConfig()
                self.analysis = AnalysisConfig()
                self.monitoring = MonitoringConfig()
                self.backtest = BacktestConfig()
                self.database = DatabaseConfig()
                self.logger.info("All configuration reset to defaults")
            elif section == 'api':
                self.api = APIConfig()
            elif section == 'trading':
                self.trading = TradingConfig()
            elif section == 'analysis':
                self.analysis = AnalysisConfig()
            elif section == 'monitoring':
                self.monitoring = MonitoringConfig()
            elif section == 'backtest':
                self.backtest = BacktestConfig()
            elif section == 'database':
                self.database = DatabaseConfig()
            else:
                self.logger.error(f"Unknown configuration section: {section}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error resetting configuration: {e}")
            return False


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    return config_manager


if __name__ == "__main__":
    # Example usage
    config = get_config()
    
    print("Configuration Summary:")
    print(json.dumps(config.get_config_summary(), indent=2))
    
    # Validate configuration
    errors = config.validate_config()
    if any(errors.values()):
        print("\nConfiguration Errors:")
        for section, section_errors in errors.items():
            if section_errors:
                print(f"  {section}: {section_errors}")
    else:
        print("\n✓ Configuration is valid")

