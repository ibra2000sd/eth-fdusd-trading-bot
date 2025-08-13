#!/usr/bin/env python3
"""
ETH/FDUSD Advanced Trading Bot - Main Application Entry Point

This is the main entry point for the ETH/FDUSD Advanced Trading Bot.
It initializes all components and starts the trading system.

Author: Trading Bot Development Team
Version: 2.1.0
License: MIT
"""

import asyncio
import signal
import sys
import os
from pathlib import Path
from typing import Optional
import argparse
import logging
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from bot.trading_engine import TradingEngine
from bot.data_manager import DataManager
from bot.signal_processor import SignalProcessor
from analyzers.market_analyzer import MarketAnalyzer
from analyzers.mathematical_models import MathematicalModels
from risk_management.risk_manager import RiskManager
from risk_management.portfolio_manager import PortfolioManager
from utils.logger import setup_logging
from utils.config_manager import ConfigManager
from utils.database import DatabaseManager
from utils.monitoring import MonitoringSystem
from config.settings import Settings


class TradingBotApplication:
    """Main application class for the ETH/FDUSD Trading Bot."""
    
    def __init__(self, config_path: Optional[str] = None, mode: str = "production"):
        """
        Initialize the trading bot application.
        
        Args:
            config_path: Path to configuration file
            mode: Operating mode (development, testing, production)
        """
        self.mode = mode
        self.config_manager = ConfigManager(config_path)
        self.settings = Settings(mode=mode)
        self.logger = None
        self.components = {}
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def initialize(self):
        """Initialize all trading bot components."""
        try:
            # Setup logging
            self.logger = setup_logging(
                level=self.settings.LOG_LEVEL,
                log_file=self.settings.LOG_FILE
            )
            self.logger.info(f"Starting ETH/FDUSD Trading Bot v{self.settings.VERSION}")
            self.logger.info(f"Mode: {self.mode}")
            
            # Initialize database
            self.logger.info("Initializing database connection...")
            db_manager = DatabaseManager(self.settings.DATABASE_URL)
            await db_manager.initialize()
            self.components['database'] = db_manager
            
            # Initialize data manager
            self.logger.info("Initializing data manager...")
            data_manager = DataManager(
                api_key=self.settings.BINANCE_API_KEY,
                secret_key=self.settings.BINANCE_SECRET_KEY,
                testnet=self.settings.BINANCE_TESTNET,
                database=db_manager
            )
            await data_manager.initialize()
            self.components['data_manager'] = data_manager
            
            # Initialize mathematical models
            self.logger.info("Initializing mathematical models...")
            math_models = MathematicalModels(
                cci_threshold=self.settings.CCI_THRESHOLD,
                ddm_threshold=self.settings.DDM_THRESHOLD,
                lookback_period=self.settings.LOOKBACK_PERIOD
            )
            self.components['math_models'] = math_models
            
            # Initialize market analyzer
            self.logger.info("Initializing market analyzer...")
            market_analyzer = MarketAnalyzer(
                data_manager=data_manager,
                math_models=math_models
            )
            self.components['market_analyzer'] = market_analyzer
            
            # Initialize signal processor
            self.logger.info("Initializing signal processor...")
            signal_processor = SignalProcessor(
                market_analyzer=market_analyzer,
                confidence_threshold=self.settings.SIGNAL_CONFIDENCE_MIN
            )
            self.components['signal_processor'] = signal_processor
            
            # Initialize risk manager
            self.logger.info("Initializing risk manager...")
            risk_manager = RiskManager(
                max_position_size=self.settings.MAX_POSITION_SIZE,
                daily_loss_limit=self.settings.DAILY_LOSS_LIMIT,
                stop_loss_percentage=self.settings.STOP_LOSS_PERCENTAGE
            )
            self.components['risk_manager'] = risk_manager
            
            # Initialize portfolio manager
            self.logger.info("Initializing portfolio manager...")
            portfolio_manager = PortfolioManager(
                data_manager=data_manager,
                risk_manager=risk_manager,
                database=db_manager
            )
            await portfolio_manager.initialize()
            self.components['portfolio_manager'] = portfolio_manager
            
            # Initialize trading engine
            self.logger.info("Initializing trading engine...")
            trading_engine = TradingEngine(
                data_manager=data_manager,
                signal_processor=signal_processor,
                portfolio_manager=portfolio_manager,
                risk_manager=risk_manager,
                trading_pair=self.settings.TRADING_PAIR
            )
            await trading_engine.initialize()
            self.components['trading_engine'] = trading_engine
            
            # Initialize monitoring system
            self.logger.info("Initializing monitoring system...")
            monitoring_system = MonitoringSystem(
                components=self.components,
                settings=self.settings
            )
            await monitoring_system.initialize()
            self.components['monitoring'] = monitoring_system
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize trading bot: {e}")
            raise
    
    async def start(self):
        """Start the trading bot."""
        try:
            self.logger.info("Starting trading bot...")
            self.running = True
            
            # Start all components
            for name, component in self.components.items():
                if hasattr(component, 'start'):
                    self.logger.info(f"Starting {name}...")
                    await component.start()
            
            self.logger.info("Trading bot started successfully")
            
            # Main trading loop
            await self._main_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {e}")
            raise
    
    async def _main_loop(self):
        """Main trading loop."""
        self.logger.info("Entering main trading loop...")
        
        try:
            while self.running:
                # Process market data and generate signals
                await self.components['trading_engine'].process_market_cycle()
                
                # Update monitoring metrics
                await self.components['monitoring'].update_metrics()
                
                # Sleep for the configured interval
                await asyncio.sleep(self.settings.PROCESSING_INTERVAL)
                
        except asyncio.CancelledError:
            self.logger.info("Main loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            raise
    
    async def stop(self):
        """Stop the trading bot gracefully."""
        self.logger.info("Stopping trading bot...")
        self.running = False
        
        # Stop all components in reverse order
        for name, component in reversed(list(self.components.items())):
            if hasattr(component, 'stop'):
                try:
                    self.logger.info(f"Stopping {name}...")
                    await component.stop()
                except Exception as e:
                    self.logger.error(f"Error stopping {name}: {e}")
        
        self.logger.info("Trading bot stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ETH/FDUSD Advanced Trading Bot")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["development", "testing", "production"],
        default="production",
        help="Operating mode"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtesting mode"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Create and initialize the application
    app = TradingBotApplication(
        config_path=args.config,
        mode=args.mode
    )
    
    try:
        await app.initialize()
        
        if args.validate:
            print("Configuration validation successful")
            return
        
        if args.backtest:
            from tests.backtests.backtest_runner import BacktestRunner
            backtest_runner = BacktestRunner(app.components)
            await backtest_runner.run()
            return
        
        # Start the trading bot
        await app.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await app.stop()


if __name__ == "__main__":
    # Set up event loop policy for Windows compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the main function
    asyncio.run(main())

