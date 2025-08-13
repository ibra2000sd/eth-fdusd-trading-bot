"""
Trading Engine - Core Trading Logic

This module contains the main trading engine that orchestrates
all trading operations, signal processing, and order execution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .data_manager import DataManager
from .signal_processor import SignalProcessor
from ..risk_management.portfolio_manager import PortfolioManager
from ..risk_management.risk_manager import RiskManager


class OrderType(Enum):
    """Order types supported by the trading engine."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderSide(Enum):
    """Order sides."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class TradingSignal:
    """Trading signal data structure."""
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price: float
    cci_value: float
    ddm_value: float
    ml_prediction: float
    risk_score: float
    position_size: float


@dataclass
class Order:
    """Order data structure."""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    timestamp: datetime
    status: str = "NEW"


class TradingEngine:
    """
    Main trading engine that orchestrates all trading operations.
    
    This class is responsible for:
    - Processing market data
    - Generating trading signals
    - Managing positions
    - Executing orders
    - Risk management
    """
    
    def __init__(
        self,
        data_manager: DataManager,
        signal_processor: SignalProcessor,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
        trading_pair: str = "ETHFDUSD"
    ):
        """
        Initialize the trading engine.
        
        Args:
            data_manager: Data management component
            signal_processor: Signal processing component
            portfolio_manager: Portfolio management component
            risk_manager: Risk management component
            trading_pair: Trading pair symbol
        """
        self.data_manager = data_manager
        self.signal_processor = signal_processor
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        self.trading_pair = trading_pair
        
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.current_position = None
        self.pending_orders: List[Order] = []
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        
    async def initialize(self):
        """Initialize the trading engine."""
        self.logger.info("Initializing trading engine...")
        
        # Initialize components
        await self.data_manager.initialize()
        await self.portfolio_manager.initialize()
        
        # Load current position
        self.current_position = await self.portfolio_manager.get_current_position(
            self.trading_pair
        )
        
        # Load pending orders
        self.pending_orders = await self.portfolio_manager.get_pending_orders(
            self.trading_pair
        )
        
        self.logger.info("Trading engine initialized successfully")
    
    async def start(self):
        """Start the trading engine."""
        self.logger.info("Starting trading engine...")
        self.is_running = True
        
        # Start data feeds
        await self.data_manager.start_real_time_feed(self.trading_pair)
        
        self.logger.info("Trading engine started")
    
    async def stop(self):
        """Stop the trading engine."""
        self.logger.info("Stopping trading engine...")
        self.is_running = False
        
        # Stop data feeds
        await self.data_manager.stop_real_time_feed()
        
        # Cancel pending orders
        for order in self.pending_orders:
            await self._cancel_order(order)
        
        self.logger.info("Trading engine stopped")
    
    async def process_market_cycle(self):
        """Process one complete market cycle."""
        try:
            # Get latest market data
            market_data = await self.data_manager.get_latest_data(self.trading_pair)
            if not market_data:
                return
            
            # Generate trading signal
            signal = await self.signal_processor.generate_signal(market_data)
            if not signal:
                return
            
            # Log signal information
            self.logger.info(
                f"Signal generated: {signal.signal_type} "
                f"(confidence: {signal.confidence:.3f}, "
                f"CCI: {signal.cci_value:.2f}, "
                f"DDM: {signal.ddm_value:.2f})"
            )
            
            # Process the signal
            await self._process_signal(signal)
            
            # Update portfolio metrics
            await self.portfolio_manager.update_metrics()
            
            # Check and manage existing positions
            await self._manage_positions()
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in market cycle processing: {e}")
    
    async def _process_signal(self, signal: TradingSignal):
        """Process a trading signal and potentially execute trades."""
        try:
            # Check risk constraints
            risk_check = await self.risk_manager.check_trade_risk(
                signal, self.current_position
            )
            
            if not risk_check.approved:
                self.logger.warning(f"Trade rejected by risk manager: {risk_check.reason}")
                return
            
            # Determine action based on signal
            if signal.signal_type == "BUY" and not self.current_position:
                await self._execute_buy_order(signal)
            elif signal.signal_type == "SELL" and self.current_position:
                await self._execute_sell_order(signal)
            elif signal.signal_type == "HOLD":
                self.logger.debug("Signal indicates HOLD - no action taken")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    async def _execute_buy_order(self, signal: TradingSignal):
        """Execute a buy order based on the signal."""
        try:
            # Calculate position size
            position_size = await self.portfolio_manager.calculate_position_size(
                signal.price, signal.risk_score
            )
            
            if position_size <= 0:
                self.logger.warning("Position size calculation returned zero or negative")
                return
            
            # Create buy order
            order = Order(
                id=f"BUY_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=self.trading_pair,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                quantity=position_size,
                price=signal.price,
                timestamp=signal.timestamp
            )
            
            # Execute order
            result = await self.data_manager.place_order(order)
            
            if result.get('status') == 'FILLED':
                # Update position
                self.current_position = {
                    'symbol': self.trading_pair,
                    'side': 'LONG',
                    'quantity': position_size,
                    'entry_price': result.get('fill_price', signal.price),
                    'entry_time': signal.timestamp,
                    'stop_loss': signal.price * (1 - 0.02),  # 2% stop loss
                    'take_profit': signal.price * (1 + 0.06)  # 6% take profit
                }
                
                # Place stop loss and take profit orders
                await self._place_exit_orders()
                
                self.logger.info(
                    f"BUY order executed: {position_size:.6f} {self.trading_pair} "
                    f"at {result.get('fill_price', signal.price):.2f}"
                )
                
                # Record trade
                self._record_trade(order, result, signal)
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
    
    async def _execute_sell_order(self, signal: TradingSignal):
        """Execute a sell order based on the signal."""
        try:
            if not self.current_position:
                return
            
            # Create sell order
            order = Order(
                id=f"SELL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=self.trading_pair,
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                quantity=self.current_position['quantity'],
                price=signal.price,
                timestamp=signal.timestamp
            )
            
            # Execute order
            result = await self.data_manager.place_order(order)
            
            if result.get('status') == 'FILLED':
                # Calculate P&L
                entry_price = self.current_position['entry_price']
                exit_price = result.get('fill_price', signal.price)
                quantity = self.current_position['quantity']
                pnl = (exit_price - entry_price) * quantity
                
                self.logger.info(
                    f"SELL order executed: {quantity:.6f} {self.trading_pair} "
                    f"at {exit_price:.2f} (P&L: {pnl:.2f})"
                )
                
                # Update performance metrics
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                self.total_pnl += pnl
                
                # Clear position
                self.current_position = None
                
                # Cancel any pending exit orders
                await self._cancel_exit_orders()
                
                # Record trade
                self._record_trade(order, result, signal)
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
    
    async def _place_exit_orders(self):
        """Place stop loss and take profit orders."""
        try:
            if not self.current_position:
                return
            
            # Stop loss order
            stop_loss_order = Order(
                id=f"SL_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=self.trading_pair,
                side=OrderSide.SELL,
                type=OrderType.STOP_LOSS,
                quantity=self.current_position['quantity'],
                stop_price=self.current_position['stop_loss'],
                timestamp=datetime.now()
            )
            
            # Take profit order
            take_profit_order = Order(
                id=f"TP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=self.trading_pair,
                side=OrderSide.SELL,
                type=OrderType.TAKE_PROFIT,
                quantity=self.current_position['quantity'],
                price=self.current_position['take_profit'],
                timestamp=datetime.now()
            )
            
            # Place orders
            await self.data_manager.place_order(stop_loss_order)
            await self.data_manager.place_order(take_profit_order)
            
            self.pending_orders.extend([stop_loss_order, take_profit_order])
            
        except Exception as e:
            self.logger.error(f"Error placing exit orders: {e}")
    
    async def _cancel_exit_orders(self):
        """Cancel pending exit orders."""
        try:
            exit_orders = [
                order for order in self.pending_orders 
                if order.type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]
            ]
            
            for order in exit_orders:
                await self._cancel_order(order)
                self.pending_orders.remove(order)
            
        except Exception as e:
            self.logger.error(f"Error canceling exit orders: {e}")
    
    async def _cancel_order(self, order: Order):
        """Cancel a specific order."""
        try:
            await self.data_manager.cancel_order(order.id)
            self.logger.info(f"Order {order.id} canceled")
        except Exception as e:
            self.logger.error(f"Error canceling order {order.id}: {e}")
    
    async def _manage_positions(self):
        """Manage existing positions and orders."""
        try:
            if not self.current_position:
                return
            
            # Check if position should be closed due to time-based rules
            entry_time = self.current_position['entry_time']
            if datetime.now() - entry_time > timedelta(hours=24):
                self.logger.info("Closing position due to time limit")
                # Create market sell signal
                current_price = await self.data_manager.get_current_price(self.trading_pair)
                signal = TradingSignal(
                    timestamp=datetime.now(),
                    signal_type="SELL",
                    confidence=1.0,
                    price=current_price,
                    cci_value=0,
                    ddm_value=0,
                    ml_prediction=0,
                    risk_score=0,
                    position_size=0
                )
                await self._execute_sell_order(signal)
            
            # Update stop loss if in profit (trailing stop)
            current_price = await self.data_manager.get_current_price(self.trading_pair)
            entry_price = self.current_position['entry_price']
            
            if current_price > entry_price * 1.03:  # 3% profit
                new_stop_loss = current_price * 0.99  # 1% trailing stop
                if new_stop_loss > self.current_position['stop_loss']:
                    self.current_position['stop_loss'] = new_stop_loss
                    self.logger.info(f"Updated trailing stop loss to {new_stop_loss:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")
    
    def _record_trade(self, order: Order, result: Dict, signal: TradingSignal):
        """Record trade details for analysis."""
        trade_record = {
            'timestamp': order.timestamp,
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': result.get('fill_price', order.price),
            'signal_confidence': signal.confidence,
            'cci_value': signal.cci_value,
            'ddm_value': signal.ddm_value,
            'ml_prediction': signal.ml_prediction
        }
        
        self.trade_history.append(trade_record)
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics."""
        try:
            # Calculate win rate
            win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            
            # Calculate current drawdown
            if self.current_position:
                current_price = self.data_manager.get_current_price_sync(self.trading_pair)
                entry_price = self.current_position['entry_price']
                unrealized_pnl = (current_price - entry_price) * self.current_position['quantity']
                
                if unrealized_pnl < 0:
                    drawdown = abs(unrealized_pnl) / (entry_price * self.current_position['quantity'])
                    self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Log performance metrics periodically
            if self.total_trades > 0 and self.total_trades % 10 == 0:
                self.logger.info(
                    f"Performance Update - Trades: {self.total_trades}, "
                    f"Win Rate: {win_rate:.1f}%, "
                    f"Total P&L: {self.total_pnl:.2f}, "
                    f"Max Drawdown: {self.max_drawdown:.3f}"
                )
        
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'current_position': self.current_position,
            'pending_orders': len(self.pending_orders)
        }

