"""
Comprehensive Backtesting Framework

This module provides a complete backtesting system for the ETH/FDUSD trading bot,
including performance metrics, risk analysis, and detailed reporting.
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from bot.data_manager import DataManager
from analyzers.mathematical_models import MathematicalModels
from risk_management.risk_manager import RiskManager


@dataclass
class BacktestTrade:
    """Data structure for backtest trade records."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'BUY' or 'SELL'
    pnl: float
    pnl_percentage: float
    duration_hours: float
    entry_signal_confidence: float
    exit_reason: str


@dataclass
class BacktestMetrics:
    """Data structure for backtest performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    total_return_percentage: float
    sharpe_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_percentage: float
    average_trade_return: float
    average_winning_trade: float
    average_losing_trade: float
    largest_winning_trade: float
    largest_losing_trade: float
    profit_factor: float
    recovery_factor: float
    expectancy: float
    average_trade_duration: float
    trades_per_month: float
    volatility: float
    beta: float
    alpha: float
    information_ratio: float
    sortino_ratio: float


class BacktestRunner:
    """
    Comprehensive backtesting system for the ETH/FDUSD trading bot.
    
    This class provides:
    - Historical data simulation
    - Signal generation and trade execution
    - Performance metrics calculation
    - Risk analysis
    - Detailed reporting and visualization
    """
    
    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        trading_pair: str = "ETHFDUSD",
        commission_rate: float = 0.001
    ):
        """
        Initialize the backtest runner.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital amount
            trading_pair: Trading pair to backtest
            commission_rate: Commission rate per trade
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.trading_pair = trading_pair
        self.commission_rate = commission_rate
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.math_models = MathematicalModels()
        self.risk_manager = RiskManager()
        
        # Backtest state
        self.current_capital = initial_capital
        self.current_position = None
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # Performance tracking
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Market data
        self.market_data: Optional[pd.DataFrame] = None
        self.benchmark_data: Optional[pd.DataFrame] = None
    
    async def run_backtest(
        self,
        data_source: str = "binance",
        timeframe: str = "15m",
        benchmark_symbol: str = "ETHUSDT"
    ) -> BacktestMetrics:
        """
        Run the complete backtest.
        
        Args:
            data_source: Data source for historical data
            timeframe: Timeframe for analysis
            benchmark_symbol: Benchmark symbol for comparison
            
        Returns:
            Comprehensive backtest metrics
        """
        try:
            self.logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
            
            # Load historical data
            await self._load_historical_data(data_source, timeframe)
            await self._load_benchmark_data(benchmark_symbol, timeframe)
            
            # Run simulation
            await self._run_simulation()
            
            # Calculate metrics
            metrics = self._calculate_metrics()
            
            # Generate reports
            await self._generate_reports(metrics)
            
            self.logger.info("Backtest completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    async def _load_historical_data(self, data_source: str, timeframe: str):
        """Load historical market data."""
        try:
            self.logger.info("Loading historical market data...")
            
            if data_source == "binance":
                # Use DataManager to fetch historical data
                data_manager = DataManager(
                    api_key="dummy",  # Not needed for historical data
                    secret_key="dummy",
                    testnet=True
                )
                
                self.market_data = await data_manager.get_historical_data(
                    symbol=self.trading_pair,
                    interval=timeframe,
                    start_time=self.start_date,
                    end_time=self.end_date,
                    limit=5000
                )
            else:
                # Load from CSV or other sources
                self.market_data = self._load_from_csv(data_source)
            
            if self.market_data is None or len(self.market_data) == 0:
                raise ValueError("No historical data loaded")
            
            self.logger.info(f"Loaded {len(self.market_data)} data points")
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            raise
    
    async def _load_benchmark_data(self, benchmark_symbol: str, timeframe: str):
        """Load benchmark data for comparison."""
        try:
            self.logger.info("Loading benchmark data...")
            
            # For simplicity, use ETH price as benchmark
            # In production, this would load actual benchmark data
            self.benchmark_data = self.market_data.copy()
            
        except Exception as e:
            self.logger.warning(f"Failed to load benchmark data: {e}")
            self.benchmark_data = None
    
    def _load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load market data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Failed to load CSV data: {e}")
            raise
    
    async def _run_simulation(self):
        """Run the trading simulation."""
        try:
            self.logger.info("Running trading simulation...")
            
            # Ensure we have enough data for analysis
            lookback_period = 100
            
            for i in range(lookback_period, len(self.market_data)):
                # Get current data window
                current_data = self.market_data.iloc[i-lookback_period:i+1]
                current_time = current_data.index[-1]
                current_price = current_data['close'].iloc[-1]
                
                # Generate trading signal
                signal = self.math_models.generate_signal(current_data)
                
                # Process signal
                await self._process_signal(signal, current_time, current_price)
                
                # Update equity curve
                self._update_equity_curve(current_time, current_price)
                
                # Check for position management
                if self.current_position:
                    await self._manage_position(current_time, current_price)
            
            # Close any remaining position
            if self.current_position:
                await self._close_position(
                    self.market_data.index[-1],
                    self.market_data['close'].iloc[-1],
                    "End of backtest"
                )
            
            self.logger.info(f"Simulation completed. Total trades: {len(self.trades)}")
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            raise
    
    async def _process_signal(self, signal: Dict, timestamp: datetime, price: float):
        """Process a trading signal."""
        try:
            signal_type = signal.get('signal_type', 'HOLD')
            confidence = signal.get('confidence', 0)
            
            # Check if signal meets minimum confidence threshold
            if confidence < 0.7:
                return
            
            # Process BUY signal
            if signal_type == 'BUY' and not self.current_position:
                await self._open_position(timestamp, price, signal)
            
            # Process SELL signal
            elif signal_type == 'SELL' and self.current_position:
                await self._close_position(timestamp, price, "Signal")
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    async def _open_position(self, timestamp: datetime, price: float, signal: Dict):
        """Open a new trading position."""
        try:
            # Calculate position size based on risk management
            risk_amount = self.current_capital * 0.02  # 2% risk per trade
            stop_loss_distance = price * 0.02  # 2% stop loss
            position_size = risk_amount / stop_loss_distance
            
            # Limit position size to available capital
            max_position_value = self.current_capital * 0.1  # 10% max position
            max_quantity = max_position_value / price
            position_size = min(position_size, max_quantity)
            
            if position_size <= 0:
                return
            
            # Calculate commission
            commission = position_size * price * self.commission_rate
            
            # Check if we have enough capital
            total_cost = position_size * price + commission
            if total_cost > self.current_capital:
                return
            
            # Open position
            self.current_position = {
                'entry_time': timestamp,
                'entry_price': price,
                'quantity': position_size,
                'side': 'LONG',
                'stop_loss': price * 0.98,  # 2% stop loss
                'take_profit': price * 1.06,  # 6% take profit
                'signal_confidence': signal.get('confidence', 0)
            }
            
            # Update capital
            self.current_capital -= total_cost
            
            self.logger.debug(f"Opened position: {position_size:.6f} at {price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
    
    async def _close_position(self, timestamp: datetime, price: float, reason: str):
        """Close the current trading position."""
        try:
            if not self.current_position:
                return
            
            # Calculate P&L
            entry_price = self.current_position['entry_price']
            quantity = self.current_position['quantity']
            
            # Calculate gross P&L
            gross_pnl = (price - entry_price) * quantity
            
            # Calculate commission
            commission = quantity * price * self.commission_rate
            
            # Calculate net P&L
            net_pnl = gross_pnl - commission
            pnl_percentage = (net_pnl / (entry_price * quantity)) * 100
            
            # Calculate trade duration
            duration = timestamp - self.current_position['entry_time']
            duration_hours = duration.total_seconds() / 3600
            
            # Create trade record
            trade = BacktestTrade(
                entry_time=self.current_position['entry_time'],
                exit_time=timestamp,
                entry_price=entry_price,
                exit_price=price,
                quantity=quantity,
                side=self.current_position['side'],
                pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                duration_hours=duration_hours,
                entry_signal_confidence=self.current_position['signal_confidence'],
                exit_reason=reason
            )
            
            self.trades.append(trade)
            
            # Update capital
            self.current_capital += quantity * price - commission
            
            # Clear position
            self.current_position = None
            
            self.logger.debug(f"Closed position: P&L {net_pnl:.2f} ({pnl_percentage:.2f}%)")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    async def _manage_position(self, timestamp: datetime, price: float):
        """Manage existing position (stop loss, take profit, trailing stop)."""
        try:
            if not self.current_position:
                return
            
            # Check stop loss
            if price <= self.current_position['stop_loss']:
                await self._close_position(timestamp, price, "Stop Loss")
                return
            
            # Check take profit
            if price >= self.current_position['take_profit']:
                await self._close_position(timestamp, price, "Take Profit")
                return
            
            # Implement trailing stop
            entry_price = self.current_position['entry_price']
            if price > entry_price * 1.03:  # 3% profit
                new_stop_loss = price * 0.99  # 1% trailing stop
                if new_stop_loss > self.current_position['stop_loss']:
                    self.current_position['stop_loss'] = new_stop_loss
            
            # Time-based exit (24 hours maximum)
            duration = timestamp - self.current_position['entry_time']
            if duration.total_seconds() > 24 * 3600:
                await self._close_position(timestamp, price, "Time Limit")
            
        except Exception as e:
            self.logger.error(f"Error managing position: {e}")
    
    def _update_equity_curve(self, timestamp: datetime, price: float):
        """Update the equity curve."""
        try:
            # Calculate current portfolio value
            portfolio_value = self.current_capital
            
            if self.current_position:
                position_value = self.current_position['quantity'] * price
                portfolio_value += position_value
            
            # Update peak and drawdown
            if portfolio_value > self.peak_capital:
                self.peak_capital = portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
            
            # Add to equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'position_value': portfolio_value - self.current_capital,
                'drawdown': self.current_drawdown,
                'price': price
            })
            
            # Calculate daily returns
            if len(self.equity_curve) > 1:
                prev_value = self.equity_curve[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
            
        except Exception as e:
            self.logger.error(f"Error updating equity curve: {e}")
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        try:
            if not self.trades:
                # Return empty metrics if no trades
                return BacktestMetrics(
                    total_trades=0, winning_trades=0, losing_trades=0,
                    win_rate=0, total_return=0, total_return_percentage=0,
                    sharpe_ratio=0, calmar_ratio=0, max_drawdown=0,
                    max_drawdown_percentage=0, average_trade_return=0,
                    average_winning_trade=0, average_losing_trade=0,
                    largest_winning_trade=0, largest_losing_trade=0,
                    profit_factor=0, recovery_factor=0, expectancy=0,
                    average_trade_duration=0, trades_per_month=0,
                    volatility=0, beta=0, alpha=0, information_ratio=0,
                    sortino_ratio=0
                )
            
            # Basic trade statistics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t.pnl > 0])
            losing_trades = len([t for t in self.trades if t.pnl < 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Return calculations
            final_value = self.equity_curve[-1]['portfolio_value'] if self.equity_curve else self.initial_capital
            total_return = final_value - self.initial_capital
            total_return_percentage = (total_return / self.initial_capital) * 100
            
            # Trade statistics
            trade_returns = [t.pnl for t in self.trades]
            winning_returns = [t.pnl for t in self.trades if t.pnl > 0]
            losing_returns = [t.pnl for t in self.trades if t.pnl < 0]
            
            average_trade_return = np.mean(trade_returns) if trade_returns else 0
            average_winning_trade = np.mean(winning_returns) if winning_returns else 0
            average_losing_trade = np.mean(losing_returns) if losing_returns else 0
            largest_winning_trade = max(winning_returns) if winning_returns else 0
            largest_losing_trade = min(losing_returns) if losing_returns else 0
            
            # Risk metrics
            max_drawdown_percentage = self.max_drawdown * 100
            
            # Profit factor
            gross_profit = sum(winning_returns) if winning_returns else 0
            gross_loss = abs(sum(losing_returns)) if losing_returns else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Recovery factor
            recovery_factor = total_return_percentage / max_drawdown_percentage if max_drawdown_percentage > 0 else 0
            
            # Expectancy
            win_probability = win_rate / 100
            loss_probability = 1 - win_probability
            avg_win = average_winning_trade if average_winning_trade > 0 else 0
            avg_loss = abs(average_losing_trade) if average_losing_trade < 0 else 0
            expectancy = (win_probability * avg_win) - (loss_probability * avg_loss)
            
            # Time-based metrics
            trade_durations = [t.duration_hours for t in self.trades]
            average_trade_duration = np.mean(trade_durations) if trade_durations else 0
            
            # Calculate trades per month
            backtest_days = (self.end_date - self.start_date).days
            trades_per_month = (total_trades / backtest_days) * 30 if backtest_days > 0 else 0
            
            # Risk-adjusted returns
            if len(self.daily_returns) > 1:
                returns_array = np.array(self.daily_returns)
                volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
                mean_return = np.mean(returns_array) * 252  # Annualized
                
                # Sharpe ratio (assuming 0% risk-free rate)
                sharpe_ratio = mean_return / volatility if volatility > 0 else 0
                
                # Sortino ratio
                negative_returns = returns_array[returns_array < 0]
                downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else volatility
                sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
                
                # Calmar ratio
                calmar_ratio = mean_return / (max_drawdown_percentage / 100) if max_drawdown_percentage > 0 else 0
            else:
                volatility = 0
                sharpe_ratio = 0
                sortino_ratio = 0
                calmar_ratio = 0
            
            # Beta and Alpha (simplified calculation)
            beta = 1.0  # Simplified assumption
            alpha = total_return_percentage - (beta * total_return_percentage)  # Simplified
            
            # Information ratio
            information_ratio = sharpe_ratio  # Simplified
            
            return BacktestMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_return=total_return,
                total_return_percentage=total_return_percentage,
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=self.max_drawdown,
                max_drawdown_percentage=max_drawdown_percentage,
                average_trade_return=average_trade_return,
                average_winning_trade=average_winning_trade,
                average_losing_trade=average_losing_trade,
                largest_winning_trade=largest_winning_trade,
                largest_losing_trade=largest_losing_trade,
                profit_factor=profit_factor,
                recovery_factor=recovery_factor,
                expectancy=expectancy,
                average_trade_duration=average_trade_duration,
                trades_per_month=trades_per_month,
                volatility=volatility,
                beta=beta,
                alpha=alpha,
                information_ratio=information_ratio,
                sortino_ratio=sortino_ratio
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise
    
    async def _generate_reports(self, metrics: BacktestMetrics):
        """Generate comprehensive backtest reports."""
        try:
            self.logger.info("Generating backtest reports...")
            
            # Create reports directory
            reports_dir = Path("data/backtest_results")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save metrics to JSON
            metrics_file = reports_dir / f"backtest_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
            
            # Save trades to CSV
            trades_file = reports_dir / f"backtest_trades_{timestamp}.csv"
            trades_df = pd.DataFrame([asdict(trade) for trade in self.trades])
            trades_df.to_csv(trades_file, index=False)
            
            # Save equity curve to CSV
            equity_file = reports_dir / f"equity_curve_{timestamp}.csv"
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(equity_file, index=False)
            
            # Generate visualizations
            await self._generate_visualizations(reports_dir, timestamp, metrics)
            
            # Generate HTML report
            await self._generate_html_report(reports_dir, timestamp, metrics)
            
            self.logger.info(f"Reports generated in {reports_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
    
    async def _generate_visualizations(self, reports_dir: Path, timestamp: str, metrics: BacktestMetrics):
        """Generate visualization charts."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('ETH/FDUSD Trading Bot - Backtest Results', fontsize=16)
            
            # Equity curve
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
                
                axes[0, 0].plot(equity_df['timestamp'], equity_df['portfolio_value'], label='Portfolio Value')
                axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
                axes[0, 0].set_title('Equity Curve')
                axes[0, 0].set_ylabel('Portfolio Value ($)')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Drawdown chart
            if self.equity_curve:
                axes[0, 1].fill_between(equity_df['timestamp'], 0, -equity_df['drawdown'] * 100, 
                                      color='red', alpha=0.3)
                axes[0, 1].set_title('Drawdown')
                axes[0, 1].set_ylabel('Drawdown (%)')
                axes[0, 1].grid(True)
            
            # Trade P&L distribution
            if self.trades:
                trade_pnls = [trade.pnl for trade in self.trades]
                axes[1, 0].hist(trade_pnls, bins=20, alpha=0.7, color='blue')
                axes[1, 0].axvline(x=0, color='red', linestyle='--')
                axes[1, 0].set_title('Trade P&L Distribution')
                axes[1, 0].set_xlabel('P&L ($)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True)
            
            # Monthly returns
            if self.equity_curve:
                equity_df['month'] = equity_df['timestamp'].dt.to_period('M')
                monthly_returns = equity_df.groupby('month')['portfolio_value'].last().pct_change().dropna() * 100
                
                colors = ['green' if x > 0 else 'red' for x in monthly_returns]
                axes[1, 1].bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
                axes[1, 1].set_title('Monthly Returns')
                axes[1, 1].set_ylabel('Return (%)')
                axes[1, 1].set_xlabel('Month')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save chart
            chart_file = reports_dir / f"backtest_charts_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
    
    async def _generate_html_report(self, reports_dir: Path, timestamp: str, metrics: BacktestMetrics):
        """Generate HTML report."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ETH/FDUSD Trading Bot - Backtest Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #2c3e50; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                    .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                    .metric-label {{ color: #7f8c8d; margin-bottom: 5px; }}
                    .positive {{ color: #27ae60; }}
                    .negative {{ color: #e74c3c; }}
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>ETH/FDUSD Advanced Trading Bot</h1>
                    <h2>Backtest Report</h2>
                    <p>Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}</p>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value {'positive' if metrics.total_return_percentage > 0 else 'negative'}">
                            {metrics.total_return_percentage:.2f}%
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{metrics.sharpe_ratio:.2f}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{metrics.win_rate:.1f}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{metrics.max_drawdown_percentage:.2f}%</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value">{metrics.total_trades}</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value">{metrics.profit_factor:.2f}</div>
                    </div>
                </div>
                
                <h3>Detailed Metrics</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Initial Capital</td><td>${self.initial_capital:,.2f}</td></tr>
                    <tr><td>Final Value</td><td>${self.initial_capital + metrics.total_return:,.2f}</td></tr>
                    <tr><td>Total Return</td><td>${metrics.total_return:,.2f}</td></tr>
                    <tr><td>Winning Trades</td><td>{metrics.winning_trades}</td></tr>
                    <tr><td>Losing Trades</td><td>{metrics.losing_trades}</td></tr>
                    <tr><td>Average Trade Return</td><td>${metrics.average_trade_return:.2f}</td></tr>
                    <tr><td>Average Winning Trade</td><td>${metrics.average_winning_trade:.2f}</td></tr>
                    <tr><td>Average Losing Trade</td><td>${metrics.average_losing_trade:.2f}</td></tr>
                    <tr><td>Largest Winning Trade</td><td>${metrics.largest_winning_trade:.2f}</td></tr>
                    <tr><td>Largest Losing Trade</td><td>${metrics.largest_losing_trade:.2f}</td></tr>
                    <tr><td>Calmar Ratio</td><td>{metrics.calmar_ratio:.2f}</td></tr>
                    <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.2f}</td></tr>
                    <tr><td>Recovery Factor</td><td>{metrics.recovery_factor:.2f}</td></tr>
                    <tr><td>Expectancy</td><td>${metrics.expectancy:.2f}</td></tr>
                    <tr><td>Average Trade Duration</td><td>{metrics.average_trade_duration:.1f} hours</td></tr>
                    <tr><td>Trades per Month</td><td>{metrics.trades_per_month:.1f}</td></tr>
                    <tr><td>Volatility</td><td>{metrics.volatility:.2f}</td></tr>
                </table>
                
                <h3>Risk Analysis</h3>
                <p>The backtest shows {'strong' if metrics.sharpe_ratio > 1.5 else 'moderate' if metrics.sharpe_ratio > 1.0 else 'weak'} 
                risk-adjusted performance with a Sharpe ratio of {metrics.sharpe_ratio:.2f}.</p>
                
                <p>Maximum drawdown of {metrics.max_drawdown_percentage:.2f}% indicates 
                {'low' if metrics.max_drawdown_percentage < 10 else 'moderate' if metrics.max_drawdown_percentage < 20 else 'high'} risk exposure.</p>
                
                <h3>Trading Performance</h3>
                <p>Win rate of {metrics.win_rate:.1f}% with {metrics.total_trades} total trades demonstrates 
                {'excellent' if metrics.win_rate > 70 else 'good' if metrics.win_rate > 60 else 'acceptable' if metrics.win_rate > 50 else 'poor'} 
                signal accuracy.</p>
                
                <p>Profit factor of {metrics.profit_factor:.2f} shows 
                {'excellent' if metrics.profit_factor > 2.0 else 'good' if metrics.profit_factor > 1.5 else 'acceptable' if metrics.profit_factor > 1.0 else 'poor'} 
                profitability relative to losses.</p>
            </body>
            </html>
            """
            
            # Save HTML report
            html_file = reports_dir / f"backtest_report_{timestamp}.html"
            with open(html_file, 'w') as f:
                f.write(html_content)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")


async def main():
    """Main function for running backtests."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define backtest parameters
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    initial_capital = 10000.0
    
    # Create and run backtest
    backtest_runner = BacktestRunner(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    try:
        metrics = await backtest_runner.run_backtest()
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Return: {metrics.total_return_percentage:.2f}%")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"Win Rate: {metrics.win_rate:.1f}%")
        print(f"Max Drawdown: {metrics.max_drawdown_percentage:.2f}%")
        print(f"Total Trades: {metrics.total_trades}")
        print(f"Profit Factor: {metrics.profit_factor:.2f}")
        print("="*60)
        
    except Exception as e:
        print(f"Backtest failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())

