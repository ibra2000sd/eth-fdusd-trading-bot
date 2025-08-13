"""
Advanced Logging Utilities for ETH/FDUSD Trading Bot
Professional logging with rotation, filtering, and monitoring
"""

import logging
import logging.handlers
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from dataclasses import dataclass, asdict


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    extra_data: Dict[str, Any] = None


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class TradingLogFilter(logging.Filter):
    """Custom filter for trading-specific logs"""
    
    def __init__(self, include_patterns: List[str] = None, exclude_patterns: List[str] = None):
        super().__init__()
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
    
    def filter(self, record):
        message = record.getMessage().lower()
        
        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if pattern.lower() in message:
                return False
        
        # If include patterns specified, message must match at least one
        if self.include_patterns:
            for pattern in self.include_patterns:
                if pattern.lower() in message:
                    return True
            return False
        
        return True


class PerformanceLogger:
    """Logger for performance metrics and timing"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.timers = {}
        self.counters = {}
        self.lock = threading.Lock()
    
    def start_timer(self, name: str):
        """Start a performance timer"""
        with self.lock:
            self.timers[name] = time.time()
    
    def end_timer(self, name: str, log_result: bool = True) -> float:
        """End a performance timer and return elapsed time"""
        with self.lock:
            if name not in self.timers:
                self.logger.warning(f"Timer '{name}' not found")
                return 0.0
            
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            
            if log_result:
                self.logger.info(f"Timer '{name}': {elapsed:.4f}s")
            
            return elapsed
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a performance counter"""
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + value
    
    def get_counter(self, name: str) -> int:
        """Get counter value"""
        with self.lock:
            return self.counters.get(name, 0)
    
    def reset_counter(self, name: str):
        """Reset a counter"""
        with self.lock:
            self.counters[name] = 0
    
    def log_counters(self):
        """Log all current counter values"""
        with self.lock:
            for name, value in self.counters.items():
                self.logger.info(f"Counter '{name}': {value}")


class TradingLogger:
    """
    Advanced logging system for trading bot
    Provides structured logging, performance monitoring, and alerting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.loggers = {}
        self.performance_logger = PerformanceLogger()
        self.log_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Setup main loggers
        self._setup_loggers()
        
        # Start log monitoring if enabled
        if config.get('enable_monitoring', True):
            self._start_log_monitoring()
    
    def _setup_loggers(self):
        """Setup all logging components"""
        # Main application logger
        self._setup_main_logger()
        
        # Trading-specific loggers
        self._setup_trading_logger()
        self._setup_analysis_logger()
        self._setup_api_logger()
        self._setup_error_logger()
    
    def _setup_main_logger(self):
        """Setup main application logger"""
        logger = logging.getLogger('TradingBot')
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.config.get('log_to_file', True):
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.config.get('log_file_path', 'trading_bot.log'),
                maxBytes=self.config.get('max_log_size', 10 * 1024 * 1024),
                backupCount=self.config.get('log_backup_count', 5)
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        self.loggers['main'] = logger
    
    def _setup_trading_logger(self):
        """Setup trading-specific logger"""
        logger = logging.getLogger('Trading')
        logger.setLevel(logging.INFO)
        
        # Trading log file
        if self.config.get('log_to_file', True):
            handler = logging.handlers.RotatingFileHandler(
                filename='trading.log',
                maxBytes=5 * 1024 * 1024,
                backupCount=10
            )
            handler.setFormatter(JSONFormatter())
            
            # Filter for trading-related messages
            trading_filter = TradingLogFilter(
                include_patterns=['order', 'position', 'signal', 'trade', 'pnl']
            )
            handler.addFilter(trading_filter)
            logger.addHandler(handler)
        
        self.loggers['trading'] = logger
    
    def _setup_analysis_logger(self):
        """Setup analysis engine logger"""
        logger = logging.getLogger('Analysis')
        logger.setLevel(logging.INFO)
        
        if self.config.get('log_to_file', True):
            handler = logging.handlers.RotatingFileHandler(
                filename='analysis.log',
                maxBytes=5 * 1024 * 1024,
                backupCount=5
            )
            handler.setFormatter(JSONFormatter())
            logger.addHandler(handler)
        
        self.loggers['analysis'] = logger
    
    def _setup_api_logger(self):
        """Setup API communication logger"""
        logger = logging.getLogger('API')
        logger.setLevel(logging.INFO)
        
        if self.config.get('log_to_file', True):
            handler = logging.handlers.RotatingFileHandler(
                filename='api.log',
                maxBytes=5 * 1024 * 1024,
                backupCount=5
            )
            handler.setFormatter(JSONFormatter())
            
            # Filter for API-related messages
            api_filter = TradingLogFilter(
                include_patterns=['request', 'response', 'websocket', 'connection']
            )
            handler.addFilter(api_filter)
            logger.addHandler(handler)
        
        self.loggers['api'] = logger
    
    def _setup_error_logger(self):
        """Setup error-specific logger"""
        logger = logging.getLogger('Errors')
        logger.setLevel(logging.WARNING)
        
        if self.config.get('log_to_file', True):
            handler = logging.handlers.RotatingFileHandler(
                filename='errors.log',
                maxBytes=5 * 1024 * 1024,
                backupCount=10
            )
            handler.setFormatter(JSONFormatter())
            logger.addHandler(handler)
        
        self.loggers['error'] = logger
    
    def _start_log_monitoring(self):
        """Start log monitoring thread"""
        def monitor_logs():
            while True:
                try:
                    self._check_log_health()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    print(f"Error in log monitoring: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_logs)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def _check_log_health(self):
        """Check log system health"""
        try:
            # Check log file sizes
            for logger_name, logger in self.loggers.items():
                for handler in logger.handlers:
                    if isinstance(handler, logging.handlers.RotatingFileHandler):
                        if os.path.exists(handler.baseFilename):
                            size = os.path.getsize(handler.baseFilename)
                            if size > handler.maxBytes * 0.9:  # 90% of max size
                                logger.info(f"Log file {handler.baseFilename} approaching rotation")
            
            # Check for error patterns
            self._analyze_recent_logs()
            
        except Exception as e:
            print(f"Error checking log health: {e}")
    
    def _analyze_recent_logs(self):
        """Analyze recent logs for patterns"""
        try:
            with self.buffer_lock:
                if len(self.log_buffer) > 100:
                    # Analyze last 100 log entries
                    recent_logs = self.log_buffer[-100:]
                    
                    # Count error levels
                    error_count = sum(1 for log in recent_logs if log.get('level') == 'ERROR')
                    warning_count = sum(1 for log in recent_logs if log.get('level') == 'WARNING')
                    
                    if error_count > 10:  # More than 10 errors in recent logs
                        self.loggers['main'].warning(f"High error rate detected: {error_count} errors in recent logs")
                    
                    if warning_count > 20:  # More than 20 warnings
                        self.loggers['main'].info(f"High warning rate: {warning_count} warnings in recent logs")
        
        except Exception as e:
            print(f"Error analyzing logs: {e}")
    
    # Public logging methods
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trading activity"""
        self.loggers['trading'].info("Trade executed", extra={'extra_data': trade_data})
        self._add_to_buffer('trading', 'INFO', 'Trade executed', trade_data)
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal"""
        self.loggers['analysis'].info("Signal generated", extra={'extra_data': signal_data})
        self._add_to_buffer('analysis', 'INFO', 'Signal generated', signal_data)
    
    def log_position_update(self, position_data: Dict[str, Any]):
        """Log position update"""
        self.loggers['trading'].info("Position updated", extra={'extra_data': position_data})
        self._add_to_buffer('trading', 'INFO', 'Position updated', position_data)
    
    def log_api_request(self, request_data: Dict[str, Any]):
        """Log API request"""
        self.loggers['api'].info("API request", extra={'extra_data': request_data})
        self._add_to_buffer('api', 'INFO', 'API request', request_data)
    
    def log_error(self, error_message: str, error_data: Dict[str, Any] = None):
        """Log error with context"""
        self.loggers['error'].error(error_message, extra={'extra_data': error_data or {}})
        self._add_to_buffer('error', 'ERROR', error_message, error_data)
    
    def log_performance(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metric"""
        perf_data = {
            'metric': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        }
        self.performance_logger.logger.info(f"Performance metric: {metric_name} = {value} {unit}")
        self._add_to_buffer('performance', 'INFO', f"Performance: {metric_name}", perf_data)
    
    def _add_to_buffer(self, logger_name: str, level: str, message: str, extra_data: Dict[str, Any] = None):
        """Add log entry to buffer for analysis"""
        try:
            with self.buffer_lock:
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'logger': logger_name,
                    'level': level,
                    'message': message,
                    'extra_data': extra_data or {}
                }
                
                self.log_buffer.append(log_entry)
                
                # Keep buffer size manageable
                if len(self.log_buffer) > 1000:
                    self.log_buffer = self.log_buffer[-500:]  # Keep last 500 entries
        
        except Exception as e:
            print(f"Error adding to log buffer: {e}")
    
    # Utility methods
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get specific logger by name"""
        return self.loggers.get(name, self.loggers['main'])
    
    def get_recent_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent log entries"""
        with self.buffer_lock:
            return self.log_buffer[-count:] if len(self.log_buffer) >= count else self.log_buffer.copy()
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        with self.buffer_lock:
            if not self.log_buffer:
                return {'total_logs': 0}
            
            # Count by level
            level_counts = {}
            logger_counts = {}
            
            for log in self.log_buffer:
                level = log.get('level', 'UNKNOWN')
                logger_name = log.get('logger', 'UNKNOWN')
                
                level_counts[level] = level_counts.get(level, 0) + 1
                logger_counts[logger_name] = logger_counts.get(logger_name, 0) + 1
            
            return {
                'total_logs': len(self.log_buffer),
                'level_counts': level_counts,
                'logger_counts': logger_counts,
                'buffer_size': len(self.log_buffer)
            }
    
    def export_logs(self, filename: str, start_time: datetime = None, end_time: datetime = None):
        """Export logs to file"""
        try:
            with self.buffer_lock:
                logs_to_export = self.log_buffer.copy()
            
            # Filter by time if specified
            if start_time or end_time:
                filtered_logs = []
                for log in logs_to_export:
                    log_time = datetime.fromisoformat(log['timestamp'])
                    if start_time and log_time < start_time:
                        continue
                    if end_time and log_time > end_time:
                        continue
                    filtered_logs.append(log)
                logs_to_export = filtered_logs
            
            # Export to JSON file
            with open(filename, 'w') as f:
                json.dump(logs_to_export, f, indent=2, default=str)
            
            self.loggers['main'].info(f"Exported {len(logs_to_export)} log entries to {filename}")
            
        except Exception as e:
            self.loggers['error'].error(f"Error exporting logs: {e}")
    
    def clear_logs(self):
        """Clear log buffer"""
        with self.buffer_lock:
            self.log_buffer.clear()
        self.loggers['main'].info("Log buffer cleared")


# Context manager for performance timing
class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, logger: TradingLogger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.logger.log_performance(self.operation_name, elapsed, "seconds")


if __name__ == "__main__":
    # Example usage
    config = {
        'log_level': 'INFO',
        'log_to_file': True,
        'log_file_path': 'test_trading_bot.log',
        'max_log_size': 1024 * 1024,
        'log_backup_count': 3
    }
    
    logger = TradingLogger(config)
    
    # Test logging
    logger.log_trade({
        'symbol': 'ETHFDUSD',
        'side': 'BUY',
        'quantity': 0.1,
        'price': 4200.0
    })
    
    logger.log_signal({
        'signal_type': 'BUY',
        'confidence': 85.5,
        'cci_value': 2.8
    })
    
    # Test performance timing
    with PerformanceTimer(logger, "test_operation"):
        time.sleep(0.1)  # Simulate work
    
    print("Logging test completed")
    print("Log stats:", json.dumps(logger.get_log_stats(), indent=2))

