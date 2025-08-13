"""
Advanced Portfolio Optimization System for ETH/FDUSD Trading Bot
Sophisticated portfolio optimization using modern portfolio theory and advanced algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationObjective:
    """Portfolio optimization objective configuration"""
    primary_objective: str  # 'sharpe', 'return', 'risk', 'calmar', 'sortino'
    risk_tolerance: float   # 0.0 (risk-averse) to 1.0 (risk-seeking)
    return_target: float    # Target return (annualized)
    max_volatility: float   # Maximum acceptable volatility
    max_drawdown: float     # Maximum acceptable drawdown
    rebalance_threshold: float  # Threshold for rebalancing (0.05 = 5%)
    
    # Constraints
    min_position_size: float = 0.01  # 1% minimum
    max_position_size: float = 0.5   # 50% maximum
    max_positions: int = 5
    
    # Preferences
    prefer_momentum: bool = True
    prefer_mean_reversion: bool = False
    consider_sentiment: bool = True
    consider_volatility_regime: bool = True


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    timestamp: datetime
    objective_value: float
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    expected_drawdown: float
    
    # Rebalancing recommendations
    rebalance_required: bool
    position_changes: Dict[str, float]  # symbol -> weight change
    estimated_costs: float
    
    # Risk metrics
    portfolio_beta: float
    diversification_ratio: float
    concentration_index: float
    
    # Optimization details
    optimization_method: str
    convergence_status: str
    iterations: int
    computation_time: float


class AdvancedPortfolioOptimizer:
    """
    Sophisticated portfolio optimizer using multiple optimization techniques
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Optimization settings
        self.objective = OptimizationObjective(**config.get('optimization_objective', {}))
        
        # Historical data for optimization
        self.returns_history = {}  # symbol -> returns series
        self.correlation_matrix = None
        self.covariance_matrix = None
        
        # Current portfolio state
        self.current_weights = {}
        self.current_positions = {}
        
        # Optimization history
        self.optimization_history = []
        self.performance_tracking = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'avg_improvement': 0.0,
            'best_sharpe': 0.0,
            'rebalances_executed': 0
        }
        
        # Market regime detection
        self.current_regime = 'normal'
        self.regime_history = []
        
        self.logger.info("Advanced Portfolio Optimizer initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for portfolio optimizer"""
        logger = logging.getLogger('PortfolioOptimizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def update_market_data(self, symbol: str, price_data: pd.DataFrame):
        """Update market data for optimization"""
        try:
            # Calculate returns
            returns = price_data['close'].pct_change().dropna()
            self.returns_history[symbol] = returns
            
            # Update correlation and covariance matrices
            self._update_correlation_matrices()
            
            self.logger.debug(f"Updated market data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def update_portfolio_state(self, positions: Dict[str, Dict[str, Any]]):
        """Update current portfolio state"""
        try:
            self.current_positions = positions
            
            # Calculate current weights
            total_value = sum(pos['market_value'] for pos in positions.values())
            
            if total_value > 0:
                self.current_weights = {
                    symbol: pos['market_value'] / total_value
                    for symbol, pos in positions.items()
                }
            else:
                self.current_weights = {}
            
            self.logger.debug(f"Updated portfolio state: {len(positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}")
    
    def optimize_portfolio(self, available_symbols: List[str], 
                          portfolio_value: float) -> OptimizationResult:
        """
        Perform comprehensive portfolio optimization
        """
        try:
            start_time = datetime.now()
            
            # Ensure we have sufficient data
            if not self._validate_optimization_data(available_symbols):
                self.logger.warning("Insufficient data for optimization")
                return None
            
            # Detect market regime
            self._detect_market_regime()
            
            # Prepare optimization data
            returns_matrix = self._prepare_returns_matrix(available_symbols)
            
            # Perform optimization based on objective
            if self.objective.primary_objective == 'sharpe':
                result = self._optimize_sharpe_ratio(returns_matrix, available_symbols)
            elif self.objective.primary_objective == 'return':
                result = self._optimize_return(returns_matrix, available_symbols)
            elif self.objective.primary_objective == 'risk':
                result = self._optimize_risk(returns_matrix, available_symbols)
            elif self.objective.primary_objective == 'calmar':
                result = self._optimize_calmar_ratio(returns_matrix, available_symbols)
            else:
                result = self._optimize_multi_objective(returns_matrix, available_symbols)
            
            # Calculate additional metrics
            result = self._enhance_optimization_result(result, returns_matrix, available_symbols)
            
            # Determine rebalancing needs
            result = self._analyze_rebalancing_needs(result, portfolio_value)
            
            # Record optimization
            computation_time = (datetime.now() - start_time).total_seconds()
            result.computation_time = computation_time
            result.timestamp = datetime.now()
            
            self.optimization_history.append(result)
            self._update_performance_tracking(result)
            
            self.logger.info(f"Portfolio optimization completed: {result.objective_value:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return None
    
    def _validate_optimization_data(self, symbols: List[str]) -> bool:
        """Validate that we have sufficient data for optimization"""
        min_periods = 50  # Minimum periods required
        
        for symbol in symbols:
            if symbol not in self.returns_history:
                return False
            if len(self.returns_history[symbol]) < min_periods:
                return False
        
        return True
    
    def _detect_market_regime(self):
        """Detect current market regime for regime-aware optimization"""
        try:
            # Use ETH returns for regime detection (primary asset)
            if 'ETHFDUSD' not in self.returns_history:
                self.current_regime = 'normal'
                return
            
            returns = self.returns_history['ETHFDUSD'].tail(100)
            
            if len(returns) < 50:
                self.current_regime = 'normal'
                return
            
            # Calculate regime indicators
            volatility = returns.std()
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Regime classification (simplified)
            if volatility > returns.rolling(50).std().quantile(0.8):
                if skewness < -0.5:
                    self.current_regime = 'crisis'
                else:
                    self.current_regime = 'high_volatility'
            elif volatility < returns.rolling(50).std().quantile(0.2):
                self.current_regime = 'low_volatility'
            else:
                if abs(skewness) < 0.2 and kurtosis < 3:
                    self.current_regime = 'normal'
                else:
                    self.current_regime = 'transitional'
            
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': self.current_regime,
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis
            })
            
            # Keep only recent history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            self.logger.debug(f"Market regime detected: {self.current_regime}")
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            self.current_regime = 'normal'
    
    def _prepare_returns_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Prepare returns matrix for optimization"""
        returns_data = {}
        
        # Find common date range
        min_length = min(len(self.returns_history[symbol]) for symbol in symbols)
        
        for symbol in symbols:
            returns_data[symbol] = self.returns_history[symbol].tail(min_length)
        
        returns_matrix = pd.DataFrame(returns_data)
        return returns_matrix.dropna()
    
    def _optimize_sharpe_ratio(self, returns_matrix: pd.DataFrame, 
                              symbols: List[str]) -> OptimizationResult:
        """Optimize portfolio for maximum Sharpe ratio"""
        try:
            n_assets = len(symbols)
            
            # Objective function (negative Sharpe ratio for minimization)
            def objective(weights):
                portfolio_return = np.sum(returns_matrix.mean() * weights) * 252  # Annualized
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_matrix.cov() * 252, weights)))
                
                if portfolio_volatility == 0:
                    return -np.inf
                
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio  # Negative for minimization
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Bounds
            bounds = [(self.objective.min_position_size, self.objective.max_position_size) 
                     for _ in range(n_assets)]
            
            # Initial guess (equal weights)
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                
                # Calculate portfolio metrics
                portfolio_return = np.sum(returns_matrix.mean() * result.x) * 252
                portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(returns_matrix.cov() * 252, result.x)))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return OptimizationResult(
                    timestamp=datetime.now(),
                    objective_value=sharpe_ratio,
                    optimal_weights=optimal_weights,
                    expected_return=portfolio_return,
                    expected_volatility=portfolio_volatility,
                    expected_sharpe=sharpe_ratio,
                    expected_drawdown=0.0,  # Will be calculated later
                    rebalance_required=False,
                    position_changes={},
                    estimated_costs=0.0,
                    portfolio_beta=1.0,
                    diversification_ratio=1.0,
                    concentration_index=0.0,
                    optimization_method='sharpe_ratio',
                    convergence_status='success',
                    iterations=result.nit,
                    computation_time=0.0
                )
            else:
                self.logger.warning("Sharpe ratio optimization failed")
                return self._fallback_optimization(symbols)
                
        except Exception as e:
            self.logger.error(f"Error in Sharpe ratio optimization: {e}")
            return self._fallback_optimization(symbols)
    
    def _optimize_return(self, returns_matrix: pd.DataFrame, 
                        symbols: List[str]) -> OptimizationResult:
        """Optimize portfolio for maximum return with volatility constraint"""
        try:
            n_assets = len(symbols)
            
            # Objective function (negative return for minimization)
            def objective(weights):
                portfolio_return = np.sum(returns_matrix.mean() * weights) * 252
                return -portfolio_return
            
            # Volatility constraint
            def volatility_constraint(weights):
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_matrix.cov() * 252, weights)))
                return self.objective.max_volatility - portfolio_volatility
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': volatility_constraint}
            ]
            
            # Bounds
            bounds = [(self.objective.min_position_size, self.objective.max_position_size) 
                     for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                
                portfolio_return = np.sum(returns_matrix.mean() * result.x) * 252
                portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(returns_matrix.cov() * 252, result.x)))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return OptimizationResult(
                    timestamp=datetime.now(),
                    objective_value=portfolio_return,
                    optimal_weights=optimal_weights,
                    expected_return=portfolio_return,
                    expected_volatility=portfolio_volatility,
                    expected_sharpe=sharpe_ratio,
                    expected_drawdown=0.0,
                    rebalance_required=False,
                    position_changes={},
                    estimated_costs=0.0,
                    portfolio_beta=1.0,
                    diversification_ratio=1.0,
                    concentration_index=0.0,
                    optimization_method='max_return',
                    convergence_status='success',
                    iterations=result.nit,
                    computation_time=0.0
                )
            else:
                return self._fallback_optimization(symbols)
                
        except Exception as e:
            self.logger.error(f"Error in return optimization: {e}")
            return self._fallback_optimization(symbols)
    
    def _optimize_risk(self, returns_matrix: pd.DataFrame, 
                      symbols: List[str]) -> OptimizationResult:
        """Optimize portfolio for minimum risk"""
        try:
            n_assets = len(symbols)
            
            # Objective function (portfolio variance)
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(returns_matrix.cov() * 252, weights))
                return portfolio_variance
            
            # Return constraint (if specified)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            if self.objective.return_target > 0:
                def return_constraint(weights):
                    portfolio_return = np.sum(returns_matrix.mean() * weights) * 252
                    return portfolio_return - self.objective.return_target
                
                constraints.append({'type': 'eq', 'fun': return_constraint})
            
            # Bounds
            bounds = [(self.objective.min_position_size, self.objective.max_position_size) 
                     for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                
                portfolio_return = np.sum(returns_matrix.mean() * result.x) * 252
                portfolio_volatility = np.sqrt(result.fun)
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return OptimizationResult(
                    timestamp=datetime.now(),
                    objective_value=portfolio_volatility,
                    optimal_weights=optimal_weights,
                    expected_return=portfolio_return,
                    expected_volatility=portfolio_volatility,
                    expected_sharpe=sharpe_ratio,
                    expected_drawdown=0.0,
                    rebalance_required=False,
                    position_changes={},
                    estimated_costs=0.0,
                    portfolio_beta=1.0,
                    diversification_ratio=1.0,
                    concentration_index=0.0,
                    optimization_method='min_risk',
                    convergence_status='success',
                    iterations=result.nit,
                    computation_time=0.0
                )
            else:
                return self._fallback_optimization(symbols)
                
        except Exception as e:
            self.logger.error(f"Error in risk optimization: {e}")
            return self._fallback_optimization(symbols)
    
    def _optimize_calmar_ratio(self, returns_matrix: pd.DataFrame, 
                              symbols: List[str]) -> OptimizationResult:
        """Optimize portfolio for maximum Calmar ratio (return/max drawdown)"""
        try:
            # For Calmar ratio, we need to estimate maximum drawdown
            # This is computationally intensive, so we'll use a simplified approach
            
            def calculate_max_drawdown(weights):
                # Simulate portfolio returns
                portfolio_returns = returns_matrix.dot(weights)
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                return abs(drawdown.min())
            
            def objective(weights):
                portfolio_return = np.sum(returns_matrix.mean() * weights) * 252
                max_drawdown = calculate_max_drawdown(weights)
                
                if max_drawdown == 0:
                    return -np.inf
                
                calmar_ratio = portfolio_return / max_drawdown
                return -calmar_ratio  # Negative for minimization
            
            # Use differential evolution for global optimization
            bounds = [(self.objective.min_position_size, self.objective.max_position_size) 
                     for _ in range(len(symbols))]
            
            result = differential_evolution(
                objective,
                bounds,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
                maxiter=100,
                seed=42
            )
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                
                portfolio_return = np.sum(returns_matrix.mean() * result.x) * 252
                portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(returns_matrix.cov() * 252, result.x)))
                max_drawdown = calculate_max_drawdown(result.x)
                calmar_ratio = portfolio_return / max_drawdown if max_drawdown > 0 else 0
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return OptimizationResult(
                    timestamp=datetime.now(),
                    objective_value=calmar_ratio,
                    optimal_weights=optimal_weights,
                    expected_return=portfolio_return,
                    expected_volatility=portfolio_volatility,
                    expected_sharpe=sharpe_ratio,
                    expected_drawdown=max_drawdown,
                    rebalance_required=False,
                    position_changes={},
                    estimated_costs=0.0,
                    portfolio_beta=1.0,
                    diversification_ratio=1.0,
                    concentration_index=0.0,
                    optimization_method='calmar_ratio',
                    convergence_status='success',
                    iterations=result.nit,
                    computation_time=0.0
                )
            else:
                return self._fallback_optimization(symbols)
                
        except Exception as e:
            self.logger.error(f"Error in Calmar ratio optimization: {e}")
            return self._fallback_optimization(symbols)
    
    def _optimize_multi_objective(self, returns_matrix: pd.DataFrame, 
                                 symbols: List[str]) -> OptimizationResult:
        """Multi-objective optimization combining multiple criteria"""
        try:
            n_assets = len(symbols)
            
            def objective(weights):
                # Portfolio metrics
                portfolio_return = np.sum(returns_matrix.mean() * weights) * 252
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_matrix.cov() * 252, weights)))
                
                # Sharpe ratio component
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                # Diversification component (negative concentration)
                concentration = np.sum(weights ** 2)  # Herfindahl index
                diversification_score = 1 - concentration
                
                # Risk tolerance adjustment
                risk_penalty = self.objective.risk_tolerance * portfolio_volatility
                
                # Combined objective (maximize)
                combined_score = (
                    0.5 * sharpe_ratio +
                    0.3 * diversification_score +
                    0.2 * (portfolio_return - risk_penalty)
                )
                
                return -combined_score  # Negative for minimization
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Bounds
            bounds = [(self.objective.min_position_size, self.objective.max_position_size) 
                     for _ in range(n_assets)]
            
            # Initial guess
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            
            # Optimization
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                
                portfolio_return = np.sum(returns_matrix.mean() * result.x) * 252
                portfolio_volatility = np.sqrt(np.dot(result.x.T, np.dot(returns_matrix.cov() * 252, result.x)))
                sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
                return OptimizationResult(
                    timestamp=datetime.now(),
                    objective_value=-result.fun,
                    optimal_weights=optimal_weights,
                    expected_return=portfolio_return,
                    expected_volatility=portfolio_volatility,
                    expected_sharpe=sharpe_ratio,
                    expected_drawdown=0.0,
                    rebalance_required=False,
                    position_changes={},
                    estimated_costs=0.0,
                    portfolio_beta=1.0,
                    diversification_ratio=1.0,
                    concentration_index=0.0,
                    optimization_method='multi_objective',
                    convergence_status='success',
                    iterations=result.nit,
                    computation_time=0.0
                )
            else:
                return self._fallback_optimization(symbols)
                
        except Exception as e:
            self.logger.error(f"Error in multi-objective optimization: {e}")
            return self._fallback_optimization(symbols)
    
    def _enhance_optimization_result(self, result: OptimizationResult, 
                                   returns_matrix: pd.DataFrame, 
                                   symbols: List[str]) -> OptimizationResult:
        """Enhance optimization result with additional metrics"""
        try:
            weights = np.array([result.optimal_weights[symbol] for symbol in symbols])
            
            # Portfolio beta (simplified - using ETH as market proxy)
            if 'ETHFDUSD' in symbols:
                market_returns = returns_matrix['ETHFDUSD']
                portfolio_returns = returns_matrix.dot(weights)
                
                if len(market_returns) > 1 and len(portfolio_returns) > 1:
                    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                    market_variance = np.var(market_returns)
                    result.portfolio_beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    result.portfolio_beta = 1.0
            else:
                result.portfolio_beta = 1.0
            
            # Diversification ratio
            individual_volatilities = np.sqrt(np.diag(returns_matrix.cov() * 252))
            weighted_avg_volatility = np.sum(weights * individual_volatilities)
            result.diversification_ratio = weighted_avg_volatility / result.expected_volatility if result.expected_volatility > 0 else 1.0
            
            # Concentration index (Herfindahl)
            result.concentration_index = np.sum(weights ** 2)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error enhancing optimization result: {e}")
            return result
    
    def _analyze_rebalancing_needs(self, result: OptimizationResult, 
                                  portfolio_value: float) -> OptimizationResult:
        """Analyze if rebalancing is needed and calculate costs"""
        try:
            if not self.current_weights:
                result.rebalance_required = True
                result.position_changes = result.optimal_weights.copy()
                result.estimated_costs = portfolio_value * 0.001  # 0.1% estimated cost
                return result
            
            # Calculate weight differences
            position_changes = {}
            max_change = 0
            
            for symbol in result.optimal_weights:
                current_weight = self.current_weights.get(symbol, 0)
                optimal_weight = result.optimal_weights[symbol]
                change = optimal_weight - current_weight
                position_changes[symbol] = change
                max_change = max(max_change, abs(change))
            
            # Check if rebalancing is needed
            result.rebalance_required = max_change > self.objective.rebalance_threshold
            result.position_changes = position_changes
            
            # Estimate transaction costs
            if result.rebalance_required:
                total_turnover = sum(abs(change) for change in position_changes.values())
                result.estimated_costs = portfolio_value * total_turnover * 0.001  # 0.1% per trade
            else:
                result.estimated_costs = 0.0
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing rebalancing needs: {e}")
            result.rebalance_required = False
            result.position_changes = {}
            result.estimated_costs = 0.0
            return result
    
    def _fallback_optimization(self, symbols: List[str]) -> OptimizationResult:
        """Fallback optimization when main optimization fails"""
        # Equal weight portfolio as fallback
        equal_weight = 1.0 / len(symbols)
        optimal_weights = {symbol: equal_weight for symbol in symbols}
        
        return OptimizationResult(
            timestamp=datetime.now(),
            objective_value=0.0,
            optimal_weights=optimal_weights,
            expected_return=0.05,  # 5% default
            expected_volatility=0.20,  # 20% default
            expected_sharpe=0.25,
            expected_drawdown=0.15,
            rebalance_required=True,
            position_changes=optimal_weights.copy(),
            estimated_costs=0.0,
            portfolio_beta=1.0,
            diversification_ratio=1.0,
            concentration_index=equal_weight,
            optimization_method='equal_weight_fallback',
            convergence_status='fallback',
            iterations=0,
            computation_time=0.0
        )
    
    def _update_correlation_matrices(self):
        """Update correlation and covariance matrices"""
        try:
            if len(self.returns_history) < 2:
                return
            
            # Align all return series
            common_symbols = list(self.returns_history.keys())
            min_length = min(len(series) for series in self.returns_history.values())
            
            aligned_returns = {}
            for symbol in common_symbols:
                aligned_returns[symbol] = self.returns_history[symbol].tail(min_length)
            
            returns_df = pd.DataFrame(aligned_returns)
            
            self.correlation_matrix = returns_df.corr()
            self.covariance_matrix = returns_df.cov()
            
        except Exception as e:
            self.logger.error(f"Error updating correlation matrices: {e}")
    
    def _update_performance_tracking(self, result: OptimizationResult):
        """Update performance tracking statistics"""
        self.performance_tracking['total_optimizations'] += 1
        
        if result.convergence_status == 'success':
            self.performance_tracking['successful_optimizations'] += 1
        
        if result.expected_sharpe > self.performance_tracking['best_sharpe']:
            self.performance_tracking['best_sharpe'] = result.expected_sharpe
        
        if result.rebalance_required:
            self.performance_tracking['rebalances_executed'] += 1
    
    # Public API methods
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary and statistics"""
        latest_result = self.optimization_history[-1] if self.optimization_history else None
        
        return {
            'current_regime': self.current_regime,
            'optimization_objective': asdict(self.objective),
            'performance_tracking': self.performance_tracking,
            'latest_optimization': asdict(latest_result) if latest_result else None,
            'portfolio_state': {
                'current_weights': self.current_weights,
                'number_of_positions': len(self.current_positions)
            }
        }
    
    def get_rebalancing_recommendations(self) -> Dict[str, Any]:
        """Get current rebalancing recommendations"""
        if not self.optimization_history:
            return {'recommendations': 'No optimization performed yet'}
        
        latest_result = self.optimization_history[-1]
        
        if not latest_result.rebalance_required:
            return {
                'rebalance_required': False,
                'message': 'Portfolio is within rebalancing threshold'
            }
        
        return {
            'rebalance_required': True,
            'position_changes': latest_result.position_changes,
            'estimated_costs': latest_result.estimated_costs,
            'expected_improvement': {
                'return': latest_result.expected_return,
                'volatility': latest_result.expected_volatility,
                'sharpe': latest_result.expected_sharpe
            }
        }
    
    def simulate_portfolio_performance(self, weights: Dict[str, float], 
                                     periods: int = 252) -> Dict[str, Any]:
        """Simulate portfolio performance with given weights"""
        try:
            symbols = list(weights.keys())
            weight_array = np.array([weights[symbol] for symbol in symbols])
            
            # Get returns matrix
            returns_matrix = self._prepare_returns_matrix(symbols)
            
            if len(returns_matrix) < periods:
                periods = len(returns_matrix)
            
            # Simulate portfolio returns
            portfolio_returns = returns_matrix.tail(periods).dot(weight_array)
            
            # Calculate performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / periods) - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Drawdown analysis
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'periods_simulated': periods
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating portfolio performance: {e}")
            return {}


if __name__ == "__main__":
    # Example usage
    print("Advanced Portfolio Optimization System for ETH/FDUSD Trading Bot")
    
    # Configuration
    config = {
        'optimization_objective': {
            'primary_objective': 'sharpe',
            'risk_tolerance': 0.6,
            'return_target': 0.15,
            'max_volatility': 0.25,
            'max_drawdown': 0.20,
            'rebalance_threshold': 0.05
        }
    }
    
    # Initialize optimizer
    optimizer = AdvancedPortfolioOptimizer(config)
    
    # Simulate market data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    
    # ETH price simulation
    eth_prices = 4000 * (1 + np.random.normal(0.001, 0.03, 252)).cumprod()
    eth_data = pd.DataFrame({'close': eth_prices}, index=dates)
    
    optimizer.update_market_data('ETHFDUSD', eth_data)
    
    # Test optimization
    available_symbols = ['ETHFDUSD']
    portfolio_value = 10000.0
    
    result = optimizer.optimize_portfolio(available_symbols, portfolio_value)
    
    if result:
        print(f"Optimization completed: {result.optimization_method}")
        print(f"Expected Sharpe ratio: {result.expected_sharpe:.4f}")
        print(f"Optimal weights: {result.optimal_weights}")
        print(f"Rebalancing required: {result.rebalance_required}")
    
    print("Portfolio Optimization System initialized successfully!")

