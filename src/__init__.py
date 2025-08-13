"""
ETH/FDUSD Advanced Trading Bot

A sophisticated algorithmic trading system with proprietary mathematical models
for identifying absolute market bottoms and tops.

Version: 2.1.0
Author: Trading Bot Development Team
License: MIT
"""

__version__ = "2.1.0"
__author__ = "Trading Bot Development Team"
__license__ = "MIT"
__description__ = "Advanced algorithmic trading bot for ETH/FDUSD pair"

# Import main components for easy access
from .bot.trading_engine import TradingEngine
from .analyzers.mathematical_models import MathematicalModels
from .risk_management.risk_manager import RiskManager

__all__ = [
    "TradingEngine",
    "MathematicalModels", 
    "RiskManager"
]

