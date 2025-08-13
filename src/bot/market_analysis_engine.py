"""
Core Market Analysis Engine for ETH/FDUSD Trading Bot
Real-time market analysis with advanced mathematical models
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from collections import deque
import threading
import time

from algorithms.mathematical_models import AdvancedMathematicalModels, SignalGenerator


@dataclass
class MarketSnapshot:
    """Real-time market snapshot"""
    timestamp: datetime
    price: float
    volume: float
    high_24h: float
    low_24h: float
    price_change_24h: float
    volume_change_24h: float
    
    
@dataclass
class TradingSignal:
    """Trading signal with confidence and metadata"""
    timestamp: datetime
    signal_type: str  # BUY, SELL, WAIT
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    position_size: float
    risk_reward_ratio: float
    market_regime: str
    signal_metadata: Dict


class MarketDataBuffer:
    """
    Efficient circular buffer for market data storage
    Optimized for real-time processing
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.df_cache = None
        self.cache_valid = False
        self.lock = threading.Lock()
        
    def add_data(self, timestamp: datetime, open_price: float, high: float, 
                 low: float, close: float, volume: float):
        """Add new market data point"""
        with self.lock:
            self.data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            self.cache_valid = False
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get data as pandas DataFrame with caching"""
        with self.lock:
            if not self.cache_valid or self.df_cache is None:
                if len(self.data) == 0:
                    return pd.DataFrame()
                    
                self.df_cache = pd.DataFrame(list(self.data))
                self.df_cache['timestamp'] = pd.to_datetime(self.df_cache['timestamp'])
                self.df_cache.set_index('timestamp', inplace=True)
                self.df_cache.sort_index(inplace=True)
                self.cache_valid = True
                
            return self.df_cache.copy()
    
    def get_latest_price(self) -> Optional[float]:
        """Get latest price"""
        with self.lock:
            if len(self.data) > 0:
                return self.data[-1]['close']
            return None
    
    def get_size(self) -> int:
        """Get current buffer size"""
        return len(self.data)


class AdvancedMarketAnalysisEngine:
    """
    Sophisticated market analysis engine combining multiple mathematical models
    Real-time processing with institutional-grade precision
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize mathematical models
        self.models = AdvancedMathematicalModels(lookback_period=200)
        self.signal_generator = SignalGenerator(self.models)
        
        # Market data management
        self.market_buffer = MarketDataBuffer(max_size=2000)
        self.current_snapshot = None
        
        # Analysis state
        self.last_analysis_time = None
        self.analysis_interval = config.get('analysis_interval', 60)  # seconds
        self.min_data_points = config.get('min_data_points', 100)
        
        # Signal tracking
        self.active_signals = []
        self.signal_history = deque(maxlen=1000)
        
        # Performance metrics
        self.analysis_stats = {
            'total_analyses': 0,
            'signals_generated': 0,
            'avg_analysis_time': 0.0,
            'last_update': None
        }
        
        # Multi-timeframe data (for production, would integrate multiple timeframes)
        self.timeframes = {
            '15m': MarketDataBuffer(max_size=1000),
            '1h': MarketDataBuffer(max_size=500),
            '4h': MarketDataBuffer(max_size=200),
            '1d': MarketDataBuffer(max_size=100)
        }
        
        self.logger.info("Advanced Market Analysis Engine initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('MarketAnalysisEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    async def process_market_data(self, market_data: Dict) -> Optional[TradingSignal]:
        """
        Process incoming market data and generate signals
        Main entry point for real-time analysis
        """
        try:
            start_time = time.time()
            
            # Update market buffer
            timestamp = datetime.fromtimestamp(market_data['timestamp'] / 1000)
            self.market_buffer.add_data(
                timestamp=timestamp,
                open_price=float(market_data['open']),
                high=float(market_data['high']),
                low=float(market_data['low']),
                close=float(market_data['close']),
                volume=float(market_data['volume'])
            )
            
            # Update current snapshot
            self.current_snapshot = MarketSnapshot(
                timestamp=timestamp,
                price=float(market_data['close']),
                volume=float(market_data['volume']),
                high_24h=float(market_data.get('high_24h', market_data['high'])),
                low_24h=float(market_data.get('low_24h', market_data['low'])),
                price_change_24h=float(market_data.get('price_change_24h', 0)),
                volume_change_24h=float(market_data.get('volume_change_24h', 0))
            )
            
            # Check if analysis should be performed
            if not self._should_analyze():
                return None
            
            # Perform comprehensive analysis
            signal = await self._perform_analysis()
            
            # Update performance metrics
            analysis_time = time.time() - start_time
            self._update_performance_metrics(analysis_time)
            
            if signal:
                self.logger.info(f"Generated signal: {signal.signal_type} with confidence {signal.confidence}")
                
            return signal
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return None
    
    def _should_analyze(self) -> bool:
        """Determine if analysis should be performed"""
        # Check minimum data requirements
        if self.market_buffer.get_size() < self.min_data_points:
            return False
        
        # Check time interval
        current_time = datetime.now()
        if (self.last_analysis_time and 
            (current_time - self.last_analysis_time).total_seconds() < self.analysis_interval):
            return False
            
        return True
    
    async def _perform_analysis(self) -> Optional[TradingSignal]:
        """
        Perform comprehensive market analysis
        """
        try:
            self.last_analysis_time = datetime.now()
            self.analysis_stats['total_analyses'] += 1
            
            # Get market data
            df = self.market_buffer.get_dataframe()
            if len(df) < self.min_data_points:
                return None
            
            # Calculate technical indicators
            df = self.models.calculate_technical_indicators(df)
            
            current_index = len(df) - 1
            
            # Generate signals
            bottom_signal = self.signal_generator.generate_bottom_signal(df, current_index)
            top_signal = self.signal_generator.generate_top_signal(df, current_index)
            
            # Determine primary signal
            primary_signal = self._determine_primary_signal(bottom_signal, top_signal)
            
            if primary_signal['signal_type'] != 'WAIT':
                # Calculate position sizing and risk management
                position_details = self._calculate_position_details(df, current_index, primary_signal)
                
                # Create trading signal
                trading_signal = TradingSignal(
                    timestamp=self.current_snapshot.timestamp,
                    signal_type=primary_signal['signal_type'],
                    confidence=primary_signal['confidence'],
                    entry_price=self.current_snapshot.price,
                    stop_loss=position_details['stop_loss'],
                    take_profit=position_details['take_profit'],
                    position_size=position_details['position_size'],
                    risk_reward_ratio=position_details['risk_reward_ratio'],
                    market_regime=primary_signal['market_regime'],
                    signal_metadata={
                        'bottom_signal': bottom_signal,
                        'top_signal': top_signal,
                        'analysis_timestamp': self.last_analysis_time.isoformat(),
                        'data_points_analyzed': len(df)
                    }
                )
                
                # Add to signal history
                self.signal_history.append(trading_signal)
                self.analysis_stats['signals_generated'] += 1
                
                return trading_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
            return None
    
    def _determine_primary_signal(self, bottom_signal: Dict, top_signal: Dict) -> Dict:
        """
        Determine primary signal from bottom and top analysis
        """
        # Priority logic: higher confidence wins
        if bottom_signal['confidence'] > top_signal['confidence']:
            if bottom_signal['signal_type'] == 'BUY':
                return bottom_signal
        elif top_signal['signal_type'] == 'SELL':
            return top_signal
            
        # If no clear signal, return wait
        return {
            'signal_type': 'WAIT',
            'confidence': 0,
            'market_regime': bottom_signal.get('market_regime', 'unknown')
        }
    
    def _calculate_position_details(self, df: pd.DataFrame, index: int, signal: Dict) -> Dict:
        """
        Calculate position sizing and risk management parameters
        """
        current_price = df.iloc[index]['close']
        atr = df.iloc[index]['atr']
        
        if signal['signal_type'] == 'BUY':
            # Calculate stop loss (below recent support)
            stop_loss = current_price - (2.0 * atr)
            
            # Calculate take profit levels (Fibonacci-based)
            risk_amount = current_price - stop_loss
            take_profit = [
                current_price + (risk_amount * 1.618),  # 61.8% extension
                current_price + (risk_amount * 2.618),  # 161.8% extension
                current_price + (risk_amount * 4.236)   # 261.8% extension
            ]
            
        else:  # SELL signal
            # Calculate stop loss (above recent resistance)
            stop_loss = current_price + (2.0 * atr)
            
            # Calculate take profit levels
            risk_amount = stop_loss - current_price
            take_profit = [
                current_price - (risk_amount * 1.618),
                current_price - (risk_amount * 2.618),
                current_price - (risk_amount * 4.236)
            ]
        
        # Position sizing based on Kelly Criterion (simplified)
        risk_per_trade = 0.02  # 2% risk per trade
        position_size = risk_per_trade / (abs(current_price - stop_loss) / current_price)
        position_size = min(position_size, 0.1)  # Max 10% position
        
        # Risk-reward ratio
        risk_reward_ratio = abs(take_profit[0] - current_price) / abs(current_price - stop_loss)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _update_performance_metrics(self, analysis_time: float):
        """Update performance tracking metrics"""
        # Update average analysis time
        total_time = self.analysis_stats['avg_analysis_time'] * (self.analysis_stats['total_analyses'] - 1)
        self.analysis_stats['avg_analysis_time'] = (total_time + analysis_time) / self.analysis_stats['total_analyses']
        self.analysis_stats['last_update'] = datetime.now()
    
    def get_market_summary(self) -> Dict:
        """
        Get comprehensive market summary
        """
        if not self.current_snapshot:
            return {'status': 'No data available'}
        
        df = self.market_buffer.get_dataframe()
        if len(df) < 10:
            return {'status': 'Insufficient data'}
        
        # Calculate recent performance
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_price = self.current_snapshot.price
        
        # Market regime analysis
        current_index = len(df) - 1
        market_regime = self.models.identify_market_regime(df, current_index)
        
        # Volatility analysis
        volatility = df['close'].pct_change().tail(20).std() * np.sqrt(24 * 60 / 15)  # Annualized for 15m data
        
        return {
            'timestamp': self.current_snapshot.timestamp.isoformat(),
            'current_price': current_price,
            'price_change_24h': self.current_snapshot.price_change_24h,
            'volume_24h': self.current_snapshot.volume,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'position_in_range': (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5,
            'market_regime': market_regime,
            'volatility_annualized': volatility,
            'data_points': len(df),
            'analysis_stats': self.analysis_stats,
            'active_signals': len(self.active_signals)
        }
    
    def get_signal_history(self, limit: int = 50) -> List[Dict]:
        """Get recent signal history"""
        signals = list(self.signal_history)[-limit:]
        return [asdict(signal) for signal in signals]
    
    def get_technical_analysis(self) -> Dict:
        """
        Get detailed technical analysis
        """
        df = self.market_buffer.get_dataframe()
        if len(df) < self.min_data_points:
            return {'status': 'Insufficient data for analysis'}
        
        # Calculate indicators
        df = self.models.calculate_technical_indicators(df)
        current_index = len(df) - 1
        
        # Get latest values
        latest = df.iloc[current_index]
        
        # Calculate custom indicators
        cci = self.models.calculate_capitulation_confluence_index(df, current_index)
        ddm = self.models.calculate_distribution_detection_matrix(df, current_index)
        chaos_patterns = self.models.detect_chaos_patterns(df, current_index)
        
        return {
            'timestamp': latest.name.isoformat(),
            'price': latest['close'],
            'technical_indicators': {
                'rsi': latest['rsi'],
                'macd': latest['macd'],
                'macd_signal': latest['macd_signal'],
                'macd_histogram': latest['macd_histogram'],
                'stoch_k': latest['stoch_k'],
                'stoch_d': latest['stoch_d'],
                'williams_r': latest['williams_r'],
                'atr': latest['atr'],
                'bb_upper': latest['bb_upper'],
                'bb_middle': latest['bb_middle'],
                'bb_lower': latest['bb_lower']
            },
            'proprietary_indicators': {
                'capitulation_confluence_index': cci,
                'distribution_detection_matrix': ddm,
                'chaos_patterns': chaos_patterns
            },
            'moving_averages': {
                'sma_20': latest['sma_20'],
                'sma_50': latest['sma_50'],
                'sma_200': latest['sma_200'],
                'ema_12': latest['ema_12'],
                'ema_26': latest['ema_26']
            },
            'volume_analysis': {
                'current_volume': latest['volume'],
                'volume_sma': latest['volume_sma'],
                'volume_ratio': latest['volume_ratio'],
                'vwap': latest['vwap']
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down Market Analysis Engine")
        # Save state if needed
        # Close connections
        # Clean up resources


if __name__ == "__main__":
    # Example usage
    config = {
        'analysis_interval': 60,
        'min_data_points': 100
    }
    
    engine = AdvancedMarketAnalysisEngine(config)
    print("Market Analysis Engine initialized successfully!")

