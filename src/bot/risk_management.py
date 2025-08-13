"""
Advanced Risk Management System for ETH/FDUSD Trading Bot
Institutional-grade risk management with dynamic position sizing and portfolio protection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from collections import deque
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_loss: float = 0.05    # 5% daily loss limit
    max_drawdown: float = 0.15      # 15% maximum drawdown
    max_leverage: float = 1.0       # No leverage for spot trading
    max_correlation: float = 0.8    # Maximum correlation between positions
    var_limit_95: float = 0.03      # 3% VaR limit (95% confidence)
    var_limit_99: float = 0.05      # 5% VaR limit (99% confidence)
    max_positions: int = 3          # Maximum number of concurrent positions
    min_liquidity: float = 1000000  # Minimum market liquidity required


@dataclass
class PositionRisk:
    """Risk metrics for individual position"""
    position_id: str
    symbol: str
    entry_price: float
    current_price: float
    quantity: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Risk metrics
    var_95: float
    var_99: float
    expected_shortfall: float
    volatility: float
    beta: float
    maximum_loss: float
    time_to_stop_loss: float
    
    # Position characteristics
    holding_period: timedelta
    entry_confidence: float
    current_confidence: float
    risk_score: float


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    timestamp: datetime
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    
    # Risk metrics
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    current_drawdown: float
    
    # Concentration metrics
    largest_position_pct: float
    concentration_index: float
    correlation_risk: float
    
    # Limit utilization
    position_limit_utilization: float
    risk_limit_utilization: float
    drawdown_limit_utilization: float


@dataclass
class RiskAlert:
    """Risk alert notification"""
    timestamp: datetime
    alert_type: str  # WARNING, CRITICAL, EMERGENCY
    category: str    # POSITION, PORTFOLIO, LIMIT, MARKET
    message: str
    affected_positions: List[str]
    recommended_action: str
    severity_score: float
    auto_action_taken: bool = False


class AdvancedRiskManager:
    """
    Sophisticated risk management system with real-time monitoring
    and dynamic risk adjustment capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Risk limits
        self.risk_limits = RiskLimits(**config.get('risk_limits', {}))
        
        # Portfolio state
        self.positions = {}  # position_id -> position_data
        self.portfolio_history = deque(maxlen=1000)
        self.risk_alerts = deque(maxlen=500)
        
        # Risk calculations
        self.returns_history = deque(maxlen=1000)
        self.volatility_history = deque(maxlen=100)
        self.correlation_matrix = None
        
        # Performance tracking
        self.risk_stats = {
            'total_alerts': 0,
            'critical_alerts': 0,
            'positions_stopped': 0,
            'risk_adjusted_returns': 0.0,
            'max_risk_utilization': 0.0
        }
        
        # Market data for risk calculations
        self.market_data_cache = {}
        
        self.logger.info("Advanced Risk Manager initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for risk manager"""
        logger = logging.getLogger('RiskManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def update_position(self, position_id: str, position_data: Dict[str, Any]):
        """Update position data for risk monitoring"""
        try:
            self.positions[position_id] = position_data
            
            # Calculate position risk metrics
            position_risk = self._calculate_position_risk(position_id, position_data)
            
            # Check position-level risk limits
            self._check_position_limits(position_risk)
            
            # Update portfolio risk
            portfolio_risk = self._calculate_portfolio_risk()
            
            # Check portfolio-level risk limits
            self._check_portfolio_limits(portfolio_risk)
            
            # Store portfolio snapshot
            self.portfolio_history.append(portfolio_risk)
            
            self.logger.debug(f"Updated risk metrics for position {position_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating position risk: {e}")
    
    def _calculate_position_risk(self, position_id: str, position_data: Dict[str, Any]) -> PositionRisk:
        """Calculate comprehensive risk metrics for a position"""
        try:
            # Basic position info
            entry_price = position_data['entry_price']
            current_price = position_data['current_price']
            quantity = position_data['quantity']
            market_value = current_price * quantity
            
            # P&L calculations
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = unrealized_pnl / (entry_price * quantity) if entry_price > 0 else 0
            
            # Get historical price data for risk calculations
            symbol = position_data['symbol']
            price_history = self._get_price_history(symbol)
            
            if len(price_history) < 30:
                # Insufficient data, use conservative estimates
                var_95 = market_value * 0.05
                var_99 = market_value * 0.08
                expected_shortfall = market_value * 0.10
                volatility = 0.05
                beta = 1.0
            else:
                # Calculate returns
                returns = price_history.pct_change().dropna()
                
                # Volatility
                volatility = returns.std() * np.sqrt(365 * 24 * 4)  # Annualized for 15min data
                
                # Value at Risk
                var_95 = np.percentile(returns, 5) * market_value
                var_99 = np.percentile(returns, 1) * market_value
                
                # Expected Shortfall (Conditional VaR)
                tail_returns = returns[returns <= np.percentile(returns, 5)]
                expected_shortfall = tail_returns.mean() * market_value if len(tail_returns) > 0 else var_95
                
                # Beta (simplified - would use market index in practice)
                beta = 1.0  # Placeholder
            
            # Maximum loss calculation
            stop_loss = position_data.get('stop_loss', entry_price * 0.95)
            maximum_loss = abs((stop_loss - entry_price) * quantity)
            
            # Time to stop loss (simplified)
            time_to_stop_loss = abs(current_price - stop_loss) / (volatility * current_price / np.sqrt(365 * 24 * 4))
            
            # Holding period
            entry_time = position_data.get('entry_time', datetime.now())
            holding_period = datetime.now() - entry_time
            
            # Confidence scores
            entry_confidence = position_data.get('entry_confidence', 0.5)
            current_confidence = self._calculate_current_confidence(position_data)
            
            # Risk score (0-100, higher is riskier)
            risk_score = self._calculate_position_risk_score(
                unrealized_pnl_pct, volatility, time_to_stop_loss, current_confidence
            )
            
            return PositionRisk(
                position_id=position_id,
                symbol=symbol,
                entry_price=entry_price,
                current_price=current_price,
                quantity=quantity,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                volatility=volatility,
                beta=beta,
                maximum_loss=maximum_loss,
                time_to_stop_loss=time_to_stop_loss,
                holding_period=holding_period,
                entry_confidence=entry_confidence,
                current_confidence=current_confidence,
                risk_score=risk_score
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk: {e}")
            # Return minimal risk object
            return PositionRisk(
                position_id=position_id,
                symbol=position_data.get('symbol', 'UNKNOWN'),
                entry_price=position_data.get('entry_price', 0),
                current_price=position_data.get('current_price', 0),
                quantity=position_data.get('quantity', 0),
                market_value=0,
                unrealized_pnl=0,
                unrealized_pnl_pct=0,
                var_95=0,
                var_99=0,
                expected_shortfall=0,
                volatility=0,
                beta=1,
                maximum_loss=0,
                time_to_stop_loss=0,
                holding_period=timedelta(0),
                entry_confidence=0,
                current_confidence=0,
                risk_score=100
            )
    
    def _calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate portfolio-level risk metrics"""
        try:
            if not self.positions:
                return self._empty_portfolio_risk()
            
            # Calculate portfolio totals
            total_value = sum(pos['current_price'] * pos['quantity'] for pos in self.positions.values())
            total_pnl = sum((pos['current_price'] - pos['entry_price']) * pos['quantity'] 
                           for pos in self.positions.values())
            total_pnl_pct = total_pnl / total_value if total_value > 0 else 0
            
            # Portfolio returns for risk calculations
            portfolio_returns = self._calculate_portfolio_returns()
            
            if len(portfolio_returns) < 30:
                # Insufficient data
                portfolio_var_95 = total_value * 0.05
                portfolio_var_99 = total_value * 0.08
                portfolio_volatility = 0.05
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                maximum_drawdown = 0.0
                current_drawdown = 0.0
            else:
                # Portfolio VaR
                portfolio_var_95 = np.percentile(portfolio_returns, 5) * total_value
                portfolio_var_99 = np.percentile(portfolio_returns, 1) * total_value
                
                # Portfolio volatility
                portfolio_volatility = portfolio_returns.std() * np.sqrt(365 * 24 * 4)
                
                # Sharpe ratio
                mean_return = portfolio_returns.mean() * 365 * 24 * 4  # Annualized
                sharpe_ratio = mean_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                # Sortino ratio
                downside_returns = portfolio_returns[portfolio_returns < 0]
                downside_volatility = downside_returns.std() * np.sqrt(365 * 24 * 4) if len(downside_returns) > 0 else portfolio_volatility
                sortino_ratio = mean_return / downside_volatility if downside_volatility > 0 else 0
                
                # Drawdown calculations
                cumulative_returns = (1 + portfolio_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                maximum_drawdown = drawdown.min()
                current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
            
            # Concentration metrics
            position_values = [pos['current_price'] * pos['quantity'] for pos in self.positions.values()]
            largest_position_pct = max(position_values) / total_value if total_value > 0 else 0
            
            # Herfindahl concentration index
            position_weights = [val / total_value for val in position_values] if total_value > 0 else []
            concentration_index = sum(w**2 for w in position_weights)
            
            # Correlation risk (simplified)
            correlation_risk = self._calculate_correlation_risk()
            
            # Limit utilization
            position_limit_utilization = len(self.positions) / self.risk_limits.max_positions
            risk_limit_utilization = abs(portfolio_var_95) / (total_value * self.risk_limits.var_limit_95) if total_value > 0 else 0
            drawdown_limit_utilization = abs(current_drawdown) / self.risk_limits.max_drawdown
            
            return PortfolioRisk(
                timestamp=datetime.now(),
                total_value=total_value,
                total_pnl=total_pnl,
                total_pnl_pct=total_pnl_pct,
                portfolio_var_95=portfolio_var_95,
                portfolio_var_99=portfolio_var_99,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                maximum_drawdown=maximum_drawdown,
                current_drawdown=current_drawdown,
                largest_position_pct=largest_position_pct,
                concentration_index=concentration_index,
                correlation_risk=correlation_risk,
                position_limit_utilization=position_limit_utilization,
                risk_limit_utilization=risk_limit_utilization,
                drawdown_limit_utilization=drawdown_limit_utilization
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return self._empty_portfolio_risk()
    
    def _check_position_limits(self, position_risk: PositionRisk):
        """Check position-level risk limits"""
        alerts = []
        
        # Position size limit
        if position_risk.market_value > self.risk_limits.max_position_size * self._get_portfolio_value():
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='WARNING',
                category='POSITION',
                message=f"Position {position_risk.position_id} exceeds size limit",
                affected_positions=[position_risk.position_id],
                recommended_action='Reduce position size',
                severity_score=70
            ))
        
        # Stop loss distance
        if position_risk.time_to_stop_loss < 0.1:  # Very close to stop loss
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='CRITICAL',
                category='POSITION',
                message=f"Position {position_risk.position_id} approaching stop loss",
                affected_positions=[position_risk.position_id],
                recommended_action='Monitor closely or close position',
                severity_score=90
            ))
        
        # High risk score
        if position_risk.risk_score > 80:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='WARNING',
                category='POSITION',
                message=f"Position {position_risk.position_id} has high risk score",
                affected_positions=[position_risk.position_id],
                recommended_action='Review position and consider risk reduction',
                severity_score=position_risk.risk_score
            ))
        
        # Process alerts
        for alert in alerts:
            self._process_risk_alert(alert)
    
    def _check_portfolio_limits(self, portfolio_risk: PortfolioRisk):
        """Check portfolio-level risk limits"""
        alerts = []
        
        # Daily loss limit
        if portfolio_risk.total_pnl_pct < -self.risk_limits.max_daily_loss:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='CRITICAL',
                category='PORTFOLIO',
                message=f"Daily loss limit exceeded: {portfolio_risk.total_pnl_pct:.2%}",
                affected_positions=list(self.positions.keys()),
                recommended_action='Close all positions or reduce exposure',
                severity_score=95
            ))
        
        # Drawdown limit
        if abs(portfolio_risk.current_drawdown) > self.risk_limits.max_drawdown:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='EMERGENCY',
                category='PORTFOLIO',
                message=f"Maximum drawdown exceeded: {portfolio_risk.current_drawdown:.2%}",
                affected_positions=list(self.positions.keys()),
                recommended_action='Emergency position closure required',
                severity_score=100
            ))
        
        # VaR limit
        portfolio_value = portfolio_risk.total_value
        if abs(portfolio_risk.portfolio_var_95) > portfolio_value * self.risk_limits.var_limit_95:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='WARNING',
                category='PORTFOLIO',
                message=f"VaR 95% limit exceeded",
                affected_positions=list(self.positions.keys()),
                recommended_action='Reduce portfolio risk',
                severity_score=75
            ))
        
        # Concentration risk
        if portfolio_risk.largest_position_pct > self.risk_limits.max_position_size:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='WARNING',
                category='PORTFOLIO',
                message=f"Portfolio concentration too high: {portfolio_risk.largest_position_pct:.2%}",
                affected_positions=list(self.positions.keys()),
                recommended_action='Diversify portfolio',
                severity_score=60
            ))
        
        # Process alerts
        for alert in alerts:
            self._process_risk_alert(alert)
    
    def _process_risk_alert(self, alert: RiskAlert):
        """Process and handle risk alerts"""
        try:
            # Add to alert history
            self.risk_alerts.append(alert)
            self.risk_stats['total_alerts'] += 1
            
            if alert.alert_type in ['CRITICAL', 'EMERGENCY']:
                self.risk_stats['critical_alerts'] += 1
            
            # Log alert
            self.logger.warning(f"RISK ALERT [{alert.alert_type}]: {alert.message}")
            
            # Auto-actions for critical alerts
            if alert.alert_type == 'EMERGENCY' and self.config.get('enable_auto_actions', False):
                self._execute_emergency_actions(alert)
                alert.auto_action_taken = True
            
        except Exception as e:
            self.logger.error(f"Error processing risk alert: {e}")
    
    def _execute_emergency_actions(self, alert: RiskAlert):
        """Execute emergency risk management actions"""
        try:
            if alert.category == 'PORTFOLIO' and 'drawdown' in alert.message.lower():
                # Emergency drawdown - close all positions
                self.logger.critical("EMERGENCY: Executing portfolio-wide position closure")
                # This would trigger position closure in the main trading system
                # Implementation would depend on integration with trading engine
                
            elif alert.category == 'POSITION':
                # Emergency position action
                self.logger.critical(f"EMERGENCY: Closing position {alert.affected_positions}")
                # This would trigger specific position closure
                
        except Exception as e:
            self.logger.error(f"Error executing emergency actions: {e}")
    
    def calculate_optimal_position_size(self, signal_data: Dict[str, Any], 
                                      account_balance: float) -> float:
        """
        Calculate optimal position size using advanced risk management
        """
        try:
            # Extract signal parameters
            entry_price = signal_data['entry_price']
            stop_loss = signal_data['stop_loss']
            confidence = signal_data.get('confidence', 50) / 100.0
            
            # Risk per trade (Kelly Criterion adaptation)
            risk_per_trade = self._calculate_kelly_position_size(signal_data)
            
            # Maximum position size based on limits
            max_position_value = account_balance * self.risk_limits.max_position_size
            max_position_size = max_position_value / entry_price
            
            # Risk-based position size
            price_risk = abs(entry_price - stop_loss) / entry_price
            risk_adjusted_size = (account_balance * risk_per_trade) / (entry_price * price_risk)
            
            # Confidence adjustment
            confidence_adjusted_size = risk_adjusted_size * confidence
            
            # Volatility adjustment
            volatility = self._get_current_volatility(signal_data.get('symbol', 'ETHFDUSD'))
            volatility_adjustment = max(0.5, min(1.5, 1.0 / (volatility * 10)))
            volatility_adjusted_size = confidence_adjusted_size * volatility_adjustment
            
            # Portfolio heat adjustment
            portfolio_heat = self._calculate_portfolio_heat()
            heat_adjustment = max(0.3, 1.0 - portfolio_heat)
            final_size = volatility_adjusted_size * heat_adjustment
            
            # Apply maximum limit
            optimal_size = min(final_size, max_position_size)
            
            # Ensure minimum viable size
            min_size = 100 / entry_price  # $100 minimum
            optimal_size = max(optimal_size, min_size) if optimal_size > 0 else 0
            
            self.logger.info(f"Calculated optimal position size: {optimal_size:.6f}")
            
            return optimal_size
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size: {e}")
            return 0.0
    
    def _calculate_kelly_position_size(self, signal_data: Dict[str, Any]) -> float:
        """Calculate Kelly Criterion position size"""
        try:
            # Simplified Kelly calculation
            # In practice, would use historical win rate and average win/loss
            
            confidence = signal_data.get('confidence', 50) / 100.0
            win_rate = 0.55 + (confidence - 0.5) * 0.2  # Adjust based on confidence
            
            entry_price = signal_data['entry_price']
            stop_loss = signal_data['stop_loss']
            take_profit = signal_data.get('take_profit', [entry_price * 1.02])[0]
            
            # Average win and loss
            avg_win = abs(take_profit - entry_price) / entry_price
            avg_loss = abs(entry_price - stop_loss) / entry_price
            
            # Kelly formula: f = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            if avg_loss > 0:
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            else:
                kelly_fraction = 0.02  # Default 2%
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {e}")
            return 0.02  # Default 2%
    
    def _calculate_portfolio_heat(self) -> float:
        """Calculate current portfolio heat (risk utilization)"""
        try:
            if not self.positions:
                return 0.0
            
            # Calculate total risk exposure
            total_risk = 0
            portfolio_value = self._get_portfolio_value()
            
            for pos in self.positions.values():
                position_value = pos['current_price'] * pos['quantity']
                stop_loss = pos.get('stop_loss', pos['entry_price'] * 0.95)
                position_risk = abs(pos['current_price'] - stop_loss) * pos['quantity']
                total_risk += position_risk
            
            # Heat as percentage of portfolio
            heat = total_risk / portfolio_value if portfolio_value > 0 else 0
            return min(1.0, heat)
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio heat: {e}")
            return 0.5  # Conservative default
    
    # Helper methods
    
    def _get_price_history(self, symbol: str, periods: int = 100) -> pd.Series:
        """Get price history for risk calculations"""
        # This would integrate with market data provider
        # For now, return simulated data
        np.random.seed(42)
        base_price = 4200
        returns = np.random.normal(0, 0.02, periods)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.Series(prices[1:])
    
    def _calculate_current_confidence(self, position_data: Dict[str, Any]) -> float:
        """Calculate current confidence in position"""
        # This would integrate with signal analysis
        # For now, return degrading confidence over time
        entry_time = position_data.get('entry_time', datetime.now())
        hours_held = (datetime.now() - entry_time).total_seconds() / 3600
        
        initial_confidence = position_data.get('entry_confidence', 0.5)
        time_decay = max(0.1, 1.0 - hours_held * 0.01)  # 1% decay per hour
        
        return initial_confidence * time_decay
    
    def _calculate_position_risk_score(self, pnl_pct: float, volatility: float, 
                                     time_to_stop: float, confidence: float) -> float:
        """Calculate overall risk score for position"""
        # Combine multiple risk factors
        pnl_risk = max(0, -pnl_pct * 100)  # Higher for losses
        vol_risk = volatility * 100
        time_risk = max(0, 50 - time_to_stop * 10)  # Higher when close to stop
        confidence_risk = (1 - confidence) * 50
        
        total_risk = (pnl_risk + vol_risk + time_risk + confidence_risk) / 4
        return min(100, max(0, total_risk))
    
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate portfolio returns time series"""
        # This would use actual portfolio value history
        # For now, return simulated returns
        if len(self.portfolio_history) < 2:
            return pd.Series([0.0])
        
        values = [portfolio.total_value for portfolio in self.portfolio_history]
        returns = pd.Series(values).pct_change().dropna()
        return returns
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk"""
        # Simplified correlation risk
        # In practice, would calculate correlation matrix of all positions
        if len(self.positions) <= 1:
            return 0.0
        
        # For single asset (ETH), correlation risk is low
        return 0.2
    
    def _get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        if not self.positions:
            return 0.0
        
        return sum(pos['current_price'] * pos['quantity'] for pos in self.positions.values())
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for symbol"""
        # This would integrate with market data
        return 0.05  # 5% default volatility
    
    def _empty_portfolio_risk(self) -> PortfolioRisk:
        """Return empty portfolio risk object"""
        return PortfolioRisk(
            timestamp=datetime.now(),
            total_value=0.0,
            total_pnl=0.0,
            total_pnl_pct=0.0,
            portfolio_var_95=0.0,
            portfolio_var_99=0.0,
            portfolio_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            maximum_drawdown=0.0,
            current_drawdown=0.0,
            largest_position_pct=0.0,
            concentration_index=0.0,
            correlation_risk=0.0,
            position_limit_utilization=0.0,
            risk_limit_utilization=0.0,
            drawdown_limit_utilization=0.0
        )
    
    # Public API methods
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        portfolio_risk = self._calculate_portfolio_risk()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_risk': asdict(portfolio_risk),
            'risk_limits': asdict(self.risk_limits),
            'active_positions': len(self.positions),
            'recent_alerts': len([a for a in self.risk_alerts if 
                                (datetime.now() - a.timestamp).total_seconds() < 3600]),
            'risk_statistics': self.risk_stats
        }
    
    def get_position_risks(self) -> List[Dict[str, Any]]:
        """Get risk metrics for all positions"""
        position_risks = []
        
        for position_id, position_data in self.positions.items():
            position_risk = self._calculate_position_risk(position_id, position_data)
            position_risks.append(asdict(position_risk))
        
        return position_risks
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent risk alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            asdict(alert) for alert in self.risk_alerts
            if alert.timestamp >= cutoff_time
        ]
        
        return recent_alerts
    
    def validate_trade(self, signal_data: Dict[str, Any], account_balance: float) -> Dict[str, Any]:
        """Validate if trade meets risk requirements"""
        try:
            # Calculate position size
            position_size = self.calculate_optimal_position_size(signal_data, account_balance)
            
            # Check if trade is viable
            min_trade_size = 100 / signal_data['entry_price']  # $100 minimum
            
            if position_size < min_trade_size:
                return {
                    'approved': False,
                    'reason': 'Position size too small',
                    'recommended_size': 0
                }
            
            # Check portfolio limits
            if len(self.positions) >= self.risk_limits.max_positions:
                return {
                    'approved': False,
                    'reason': 'Maximum positions reached',
                    'recommended_size': 0
                }
            
            # Check portfolio heat
            portfolio_heat = self._calculate_portfolio_heat()
            if portfolio_heat > 0.8:  # 80% heat limit
                return {
                    'approved': False,
                    'reason': 'Portfolio heat too high',
                    'recommended_size': 0
                }
            
            return {
                'approved': True,
                'reason': 'Trade approved',
                'recommended_size': position_size,
                'risk_metrics': {
                    'portfolio_heat': portfolio_heat,
                    'position_risk': abs(signal_data['entry_price'] - signal_data['stop_loss']) / signal_data['entry_price'],
                    'kelly_fraction': self._calculate_kelly_position_size(signal_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return {
                'approved': False,
                'reason': f'Validation error: {e}',
                'recommended_size': 0
            }


if __name__ == "__main__":
    # Example usage
    print("Advanced Risk Management System for ETH/FDUSD Trading Bot")
    
    # Configuration
    config = {
        'risk_limits': {
            'max_position_size': 0.1,
            'max_daily_loss': 0.05,
            'max_drawdown': 0.15,
            'var_limit_95': 0.03,
            'max_positions': 3
        },
        'enable_auto_actions': False
    }
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager(config)
    
    # Test position update
    test_position = {
        'symbol': 'ETHFDUSD',
        'entry_price': 4200.0,
        'current_price': 4250.0,
        'quantity': 0.1,
        'stop_loss': 4100.0,
        'entry_time': datetime.now() - timedelta(hours=2),
        'entry_confidence': 0.8
    }
    
    risk_manager.update_position('test_pos_1', test_position)
    
    # Test trade validation
    signal_data = {
        'entry_price': 4200.0,
        'stop_loss': 4100.0,
        'take_profit': [4300.0, 4400.0],
        'confidence': 75,
        'symbol': 'ETHFDUSD'
    }
    
    validation = risk_manager.validate_trade(signal_data, 10000.0)
    print(f"Trade validation: {validation}")
    
    # Get risk summary
    risk_summary = risk_manager.get_risk_summary()
    print(f"Risk summary generated with {len(risk_summary)} components")
    
    print("Risk Management System initialized successfully!")

