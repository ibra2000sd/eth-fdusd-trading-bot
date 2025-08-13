"""
Main Trading Bot Engine for ETH/FDUSD
Sophisticated trading bot with advanced mathematical models
"""

import asyncio
import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import os
from decimal import Decimal
import signal
import sys

from core.market_analysis_engine import AdvancedMarketAnalysisEngine, TradingSignal
from core.binance_integration import BinanceIntegration, OrderResult, AccountBalance
from algorithms.mathematical_models import AdvancedMathematicalModels


@dataclass
class TradingConfig:
    """Trading bot configuration"""
    # API Configuration
    api_key: str
    api_secret: str
    testnet: bool = True
    
    # Trading Parameters
    symbol: str = "ETHFDUSD"
    base_asset: str = "ETH"
    quote_asset: str = "FDUSD"
    
    # Risk Management
    max_position_size: float = 0.1  # 10% of account
    risk_per_trade: float = 0.02    # 2% risk per trade
    max_daily_loss: float = 0.05    # 5% max daily loss
    
    # Signal Parameters
    min_signal_confidence: float = 70.0
    signal_timeout: int = 300  # 5 minutes
    
    # Analysis Parameters
    analysis_interval: int = 60  # seconds
    min_data_points: int = 100
    
    # Order Management
    order_timeout: int = 30  # seconds
    max_slippage: float = 0.001  # 0.1%
    
    # Monitoring
    enable_alerts: bool = True
    log_level: str = "INFO"


@dataclass
class Position:
    """Active trading position"""
    position_id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: List[float]
    entry_time: datetime
    signal_confidence: float
    unrealized_pnl: float = 0.0
    status: str = "OPEN"  # OPEN, PARTIAL, CLOSED


@dataclass
class TradingStats:
    """Trading performance statistics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    start_time: datetime = None
    last_update: datetime = None


class AdvancedTradingBot:
    """
    Sophisticated trading bot with institutional-grade features
    Combines advanced mathematical models with professional risk management
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.binance = BinanceIntegration(
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.testnet
        )
        
        analysis_config = {
            'analysis_interval': config.analysis_interval,
            'min_data_points': config.min_data_points
        }
        self.analysis_engine = AdvancedMarketAnalysisEngine(analysis_config)
        
        # Trading state
        self.is_running = False
        self.positions = {}  # position_id -> Position
        self.pending_orders = {}  # order_id -> order_info
        self.trading_stats = TradingStats(start_time=datetime.now())
        
        # Account management
        self.account_balance = 0.0
        self.daily_pnl = 0.0
        self.daily_start_balance = 0.0
        self.last_balance_update = None
        
        # Signal management
        self.last_signal = None
        self.signal_history = []
        
        # Threading
        self.main_loop_thread = None
        self.websocket_thread = None
        self.monitoring_thread = None
        
        # Emergency stop
        self.emergency_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Advanced Trading Bot initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('TradingBot')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            file_handler = logging.FileHandler('trading_bot.log')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.emergency_stop = True
        self.stop()
    
    async def start(self):
        """Start the trading bot"""
        try:
            self.logger.info("Starting Advanced Trading Bot")
            
            # Test connectivity
            if not self.binance.test_connectivity():
                raise Exception("Failed to connect to Binance API")
            
            # Initialize account
            await self._initialize_account()
            
            # Start WebSocket for real-time data
            self._start_websocket()
            
            # Start monitoring thread
            self._start_monitoring()
            
            # Set running state
            self.is_running = True
            
            # Start main trading loop
            await self._main_trading_loop()
            
        except Exception as e:
            self.logger.error(f"Error starting trading bot: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading bot gracefully"""
        try:
            self.logger.info("Stopping trading bot")
            self.is_running = False
            
            # Close all positions if emergency stop
            if self.emergency_stop:
                await self._emergency_close_positions()
            
            # Stop WebSocket
            self.binance.stop_websocket()
            
            # Cancel pending orders
            await self._cancel_all_orders()
            
            # Save final state
            await self._save_trading_state()
            
            self.logger.info("Trading bot stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping trading bot: {e}")
    
    async def _initialize_account(self):
        """Initialize account information"""
        try:
            account_info = self.binance.get_account_info()
            
            # Get FDUSD balance (quote asset)
            quote_balance = self.binance.get_balance(self.config.quote_asset)
            if quote_balance:
                self.account_balance = quote_balance.free
                self.daily_start_balance = self.account_balance
                self.logger.info(f"Account balance: {self.account_balance} {self.config.quote_asset}")
            else:
                raise Exception(f"No {self.config.quote_asset} balance found")
            
            # Check trading permissions
            if not account_info['can_trade']:
                raise Exception("Account does not have trading permissions")
            
            self.last_balance_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error initializing account: {e}")
            raise
    
    def _start_websocket(self):
        """Start WebSocket for real-time market data"""
        try:
            # Define streams for ETH/FDUSD
            streams = [
                f"{self.config.symbol.lower()}@ticker",
                f"{self.config.symbol.lower()}@kline_15m",
                f"{self.config.symbol.lower()}@depth20"
            ]
            
            self.binance.start_websocket(streams, self._websocket_callback)
            self.logger.info("WebSocket started for real-time data")
            
        except Exception as e:
            self.logger.error(f"Error starting WebSocket: {e}")
            raise
    
    def _websocket_callback(self, data: Dict):
        """Process WebSocket data"""
        try:
            if 'stream' in data:
                stream = data['stream']
                stream_data = data['data']
                
                if '@ticker' in stream:
                    self._process_ticker_data(stream_data)
                elif '@kline' in stream:
                    self._process_kline_data(stream_data)
                elif '@depth' in stream:
                    self._process_depth_data(stream_data)
                    
        except Exception as e:
            self.logger.error(f"Error processing WebSocket data: {e}")
    
    def _process_ticker_data(self, data: Dict):
        """Process ticker data"""
        try:
            # Update current price for position monitoring
            current_price = float(data['c'])
            self._update_position_pnl(current_price)
            
        except Exception as e:
            self.logger.error(f"Error processing ticker data: {e}")
    
    def _process_kline_data(self, data: Dict):
        """Process kline data for analysis"""
        try:
            kline = data['k']
            
            # Only process closed klines
            if kline['x']:  # kline is closed
                market_data = {
                    'timestamp': kline['t'],
                    'open': kline['o'],
                    'high': kline['h'],
                    'low': kline['l'],
                    'close': kline['c'],
                    'volume': kline['v']
                }
                
                # Process through analysis engine
                asyncio.create_task(self._process_market_analysis(market_data))
                
        except Exception as e:
            self.logger.error(f"Error processing kline data: {e}")
    
    def _process_depth_data(self, data: Dict):
        """Process order book depth data"""
        try:
            # Store for market microstructure analysis
            # Implementation would depend on specific analysis needs
            pass
            
        except Exception as e:
            self.logger.error(f"Error processing depth data: {e}")
    
    async def _process_market_analysis(self, market_data: Dict):
        """Process market data through analysis engine"""
        try:
            signal = await self.analysis_engine.process_market_data(market_data)
            
            if signal and signal.confidence >= self.config.min_signal_confidence:
                await self._process_trading_signal(signal)
                
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
    
    async def _process_trading_signal(self, signal: TradingSignal):
        """Process trading signal and execute if conditions are met"""
        try:
            self.logger.info(f"Processing signal: {signal.signal_type} with confidence {signal.confidence}")
            
            # Check if we can trade
            if not self._can_trade(signal):
                return
            
            # Check position limits
            if len(self.positions) >= 1:  # Only one position at a time for this strategy
                self.logger.info("Maximum positions reached, skipping signal")
                return
            
            # Execute trade
            if signal.signal_type == 'BUY':
                await self._execute_buy_signal(signal)
            elif signal.signal_type == 'SELL':
                await self._execute_sell_signal(signal)
            
            # Store signal
            self.last_signal = signal
            self.signal_history.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error processing trading signal: {e}")
    
    def _can_trade(self, signal: TradingSignal) -> bool:
        """Check if trading conditions are met"""
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.config.max_daily_loss * self.daily_start_balance:
                self.logger.warning("Daily loss limit reached, stopping trading")
                return False
            
            # Check account balance
            if self.account_balance < 100:  # Minimum balance
                self.logger.warning("Insufficient account balance")
                return False
            
            # Check signal timeout
            if self.last_signal:
                time_diff = (signal.timestamp - self.last_signal.timestamp).total_seconds()
                if time_diff < self.config.signal_timeout:
                    self.logger.info("Signal timeout not reached")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking trading conditions: {e}")
            return False
    
    async def _execute_buy_signal(self, signal: TradingSignal):
        """Execute buy signal"""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            
            if position_size <= 0:
                self.logger.warning("Position size too small, skipping trade")
                return
            
            # Place market buy order
            order_result = self.binance.place_market_order('BUY', position_size)
            
            if order_result.status == 'FILLED':
                # Create position
                position = Position(
                    position_id=f"pos_{int(time.time())}",
                    symbol=self.config.symbol,
                    side='LONG',
                    entry_price=order_result.price,
                    quantity=order_result.quantity,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    entry_time=order_result.timestamp,
                    signal_confidence=signal.confidence
                )
                
                self.positions[position.position_id] = position
                
                # Place stop loss order
                await self._place_stop_loss(position)
                
                self.logger.info(f"Buy order executed: {order_result.quantity} ETH at {order_result.price}")
                
        except Exception as e:
            self.logger.error(f"Error executing buy signal: {e}")
    
    async def _execute_sell_signal(self, signal: TradingSignal):
        """Execute sell signal (close long positions or open short if supported)"""
        try:
            # For spot trading, we can only close long positions
            long_positions = [p for p in self.positions.values() if p.side == 'LONG' and p.status == 'OPEN']
            
            for position in long_positions:
                await self._close_position(position, "SELL_SIGNAL")
                
        except Exception as e:
            self.logger.error(f"Error executing sell signal: {e}")
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on risk management"""
        try:
            # Risk amount
            risk_amount = self.account_balance * self.config.risk_per_trade
            
            # Price difference for risk calculation
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            price_diff = abs(entry_price - stop_loss)
            
            if price_diff == 0:
                return 0.0
            
            # Position size based on risk
            position_value = risk_amount / (price_diff / entry_price)
            position_size = position_value / entry_price
            
            # Apply maximum position size limit
            max_position_value = self.account_balance * self.config.max_position_size
            max_position_size = max_position_value / entry_price
            
            position_size = min(position_size, max_position_size)
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _place_stop_loss(self, position: Position):
        """Place stop loss order for position"""
        try:
            # For spot trading, we'll monitor and execute manually
            # In production, could use OCO orders or other advanced order types
            self.logger.info(f"Stop loss set at {position.stop_loss} for position {position.position_id}")
            
        except Exception as e:
            self.logger.error(f"Error placing stop loss: {e}")
    
    async def _close_position(self, position: Position, reason: str):
        """Close a trading position"""
        try:
            # Place market sell order
            order_result = self.binance.place_market_order('SELL', position.quantity)
            
            if order_result.status == 'FILLED':
                # Calculate PnL
                pnl = (order_result.price - position.entry_price) * position.quantity
                
                # Update position
                position.status = 'CLOSED'
                position.unrealized_pnl = pnl
                
                # Update trading stats
                self._update_trading_stats(position, pnl)
                
                self.logger.info(f"Position closed: {reason}, PnL: {pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _update_position_pnl(self, current_price: float):
        """Update unrealized PnL for open positions"""
        try:
            for position in self.positions.values():
                if position.status == 'OPEN':
                    if position.side == 'LONG':
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    # Check stop loss
                    if position.side == 'LONG' and current_price <= position.stop_loss:
                        asyncio.create_task(self._close_position(position, "STOP_LOSS"))
                    
                    # Check take profit
                    if position.side == 'LONG' and current_price >= position.take_profit[0]:
                        asyncio.create_task(self._close_position(position, "TAKE_PROFIT"))
                        
        except Exception as e:
            self.logger.error(f"Error updating position PnL: {e}")
    
    def _update_trading_stats(self, position: Position, pnl: float):
        """Update trading performance statistics"""
        try:
            self.trading_stats.total_trades += 1
            self.trading_stats.total_pnl += pnl
            
            if pnl > 0:
                self.trading_stats.winning_trades += 1
                self.trading_stats.avg_win = (
                    (self.trading_stats.avg_win * (self.trading_stats.winning_trades - 1) + pnl) /
                    self.trading_stats.winning_trades
                )
            else:
                self.trading_stats.losing_trades += 1
                self.trading_stats.avg_loss = (
                    (self.trading_stats.avg_loss * (self.trading_stats.losing_trades - 1) + abs(pnl)) /
                    self.trading_stats.losing_trades
                )
            
            # Calculate win rate
            self.trading_stats.win_rate = (
                self.trading_stats.winning_trades / self.trading_stats.total_trades * 100
            )
            
            # Calculate profit factor
            total_wins = self.trading_stats.avg_win * self.trading_stats.winning_trades
            total_losses = self.trading_stats.avg_loss * self.trading_stats.losing_trades
            
            if total_losses > 0:
                self.trading_stats.profit_factor = total_wins / total_losses
            
            self.trading_stats.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating trading stats: {e}")
    
    async def _main_trading_loop(self):
        """Main trading loop"""
        try:
            self.logger.info("Starting main trading loop")
            
            while self.is_running and not self.emergency_stop:
                try:
                    # Update account balance
                    await self._update_account_balance()
                    
                    # Monitor positions
                    await self._monitor_positions()
                    
                    # Check daily reset
                    await self._check_daily_reset()
                    
                    # Sleep
                    await asyncio.sleep(10)  # 10 second loop
                    
                except Exception as e:
                    self.logger.error(f"Error in main trading loop: {e}")
                    await asyncio.sleep(30)  # Longer sleep on error
            
        except Exception as e:
            self.logger.error(f"Fatal error in main trading loop: {e}")
            await self.stop()
    
    async def _update_account_balance(self):
        """Update account balance periodically"""
        try:
            if (not self.last_balance_update or 
                (datetime.now() - self.last_balance_update).total_seconds() > 300):  # 5 minutes
                
                quote_balance = self.binance.get_balance(self.config.quote_asset)
                if quote_balance:
                    self.account_balance = quote_balance.free
                    self.last_balance_update = datetime.now()
                    
        except Exception as e:
            self.logger.error(f"Error updating account balance: {e}")
    
    async def _monitor_positions(self):
        """Monitor open positions"""
        try:
            for position in list(self.positions.values()):
                if position.status == 'OPEN':
                    # Check position age
                    position_age = (datetime.now() - position.entry_time).total_seconds()
                    
                    # Close position if too old (24 hours)
                    if position_age > 86400:
                        await self._close_position(position, "TIME_LIMIT")
                        
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    async def _check_daily_reset(self):
        """Check if daily reset is needed"""
        try:
            current_time = datetime.now()
            
            # Reset daily PnL at midnight
            if (self.trading_stats.start_time.date() != current_time.date()):
                self.daily_pnl = 0.0
                self.daily_start_balance = self.account_balance
                self.trading_stats.start_time = current_time
                self.logger.info("Daily reset completed")
                
        except Exception as e:
            self.logger.error(f"Error in daily reset: {e}")
    
    def _start_monitoring(self):
        """Start monitoring thread"""
        def monitoring_loop():
            while self.is_running:
                try:
                    # Log status every 5 minutes
                    self._log_status()
                    time.sleep(300)
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _log_status(self):
        """Log current bot status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'account_balance': self.account_balance,
                'daily_pnl': self.daily_pnl,
                'open_positions': len([p for p in self.positions.values() if p.status == 'OPEN']),
                'total_trades': self.trading_stats.total_trades,
                'win_rate': self.trading_stats.win_rate,
                'total_pnl': self.trading_stats.total_pnl
            }
            
            self.logger.info(f"Bot Status: {json.dumps(status, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error logging status: {e}")
    
    async def _cancel_all_orders(self):
        """Cancel all pending orders"""
        try:
            open_orders = self.binance.get_open_orders()
            for order in open_orders:
                self.binance.cancel_order(order['orderId'])
                self.logger.info(f"Cancelled order: {order['orderId']}")
                
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")
    
    async def _emergency_close_positions(self):
        """Emergency close all positions"""
        try:
            for position in self.positions.values():
                if position.status == 'OPEN':
                    await self._close_position(position, "EMERGENCY_STOP")
                    
        except Exception as e:
            self.logger.error(f"Error in emergency close: {e}")
    
    async def _save_trading_state(self):
        """Save trading state to file"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'positions': [asdict(p) for p in self.positions.values()],
                'trading_stats': asdict(self.trading_stats),
                'account_balance': self.account_balance,
                'daily_pnl': self.daily_pnl
            }
            
            with open('trading_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            self.logger.info("Trading state saved")
            
        except Exception as e:
            self.logger.error(f"Error saving trading state: {e}")
    
    # Public API methods
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'account_balance': self.account_balance,
            'daily_pnl': self.daily_pnl,
            'positions': [asdict(p) for p in self.positions.values()],
            'trading_stats': asdict(self.trading_stats),
            'last_signal': asdict(self.last_signal) if self.last_signal else None
        }
    
    def get_performance_report(self) -> Dict:
        """Get detailed performance report"""
        return {
            'trading_stats': asdict(self.trading_stats),
            'account_summary': {
                'current_balance': self.account_balance,
                'starting_balance': self.daily_start_balance,
                'daily_pnl': self.daily_pnl,
                'total_return': ((self.account_balance - self.daily_start_balance) / self.daily_start_balance * 100) if self.daily_start_balance > 0 else 0
            },
            'position_summary': {
                'total_positions': len(self.positions),
                'open_positions': len([p for p in self.positions.values() if p.status == 'OPEN']),
                'closed_positions': len([p for p in self.positions.values() if p.status == 'CLOSED'])
            }
        }


async def main():
    """Main function for running the trading bot"""
    try:
        # Load configuration
        config = TradingConfig(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            testnet=True  # Set to False for live trading
        )
        
        if not config.api_key or not config.api_secret:
            raise ValueError("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        
        # Create and start trading bot
        bot = AdvancedTradingBot(config)
        await bot.start()
        
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Error running bot: {e}")


if __name__ == "__main__":
    asyncio.run(main())

