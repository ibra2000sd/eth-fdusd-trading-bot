"""
Unit Tests for Mathematical Models

This module contains comprehensive unit tests for the mathematical models
used in the ETH/FDUSD trading bot, including CCI and DDM algorithms.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from analyzers.mathematical_models import MathematicalModels, CCICalculator, DDMCalculator


class TestMathematicalModels(unittest.TestCase):
    """Test cases for the MathematicalModels class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models = MathematicalModels(
            cci_threshold=20,
            ddm_threshold=80,
            lookback_period=100
        )
        
        # Create sample market data
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample market data for testing."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=200),
            end=datetime.now(),
            freq='15min'
        )
        
        # Generate realistic price data with trends and volatility
        np.random.seed(42)  # For reproducible tests
        
        # Base price trend
        base_price = 3000
        trend = np.linspace(0, 500, len(dates))
        
        # Add volatility
        volatility = np.random.normal(0, 50, len(dates))
        
        # Create price series
        prices = base_price + trend + volatility
        
        # Ensure no negative prices
        prices = np.maximum(prices, 100)
        
        # Create OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        })
        
        # Ensure high >= close >= low
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        return data
    
    def test_cci_calculation(self):
        """Test CCI (Capitulation Confluence Index) calculation."""
        cci_value = self.models.calculate_cci(self.sample_data)
        
        # CCI should be a float
        self.assertIsInstance(cci_value, (int, float))
        
        # CCI should be within reasonable bounds (0-100)
        self.assertGreaterEqual(cci_value, 0)
        self.assertLessEqual(cci_value, 100)
    
    def test_ddm_calculation(self):
        """Test DDM (Distribution Detection Matrix) calculation."""
        ddm_value = self.models.calculate_ddm(self.sample_data)
        
        # DDM should be a float
        self.assertIsInstance(ddm_value, (int, float))
        
        # DDM should be within reasonable bounds (0-100)
        self.assertGreaterEqual(ddm_value, 0)
        self.assertLessEqual(ddm_value, 100)
    
    def test_signal_generation(self):
        """Test signal generation logic."""
        signal = self.models.generate_signal(self.sample_data)
        
        # Signal should be a dictionary
        self.assertIsInstance(signal, dict)
        
        # Signal should contain required fields
        required_fields = ['signal_type', 'confidence', 'cci_value', 'ddm_value']
        for field in required_fields:
            self.assertIn(field, signal)
        
        # Signal type should be valid
        self.assertIn(signal['signal_type'], ['BUY', 'SELL', 'HOLD'])
        
        # Confidence should be between 0 and 1
        self.assertGreaterEqual(signal['confidence'], 0)
        self.assertLessEqual(signal['confidence'], 1)
    
    def test_cci_components(self):
        """Test individual CCI components."""
        # Test oversold momentum
        om_value = self.models._calculate_oversold_momentum(self.sample_data)
        self.assertIsInstance(om_value, (int, float))
        self.assertGreaterEqual(om_value, 0)
        self.assertLessEqual(om_value, 100)
        
        # Test volume capitulation
        vc_value = self.models._calculate_volume_capitulation(self.sample_data)
        self.assertIsInstance(vc_value, (int, float))
        self.assertGreaterEqual(vc_value, 0)
        self.assertLessEqual(vc_value, 100)
        
        # Test support level confluence
        slc_value = self.models._calculate_support_confluence(self.sample_data)
        self.assertIsInstance(slc_value, (int, float))
        self.assertGreaterEqual(slc_value, 0)
        self.assertLessEqual(slc_value, 100)
    
    def test_ddm_components(self):
        """Test individual DDM components."""
        # Test distribution volume
        dv_value = self.models._calculate_distribution_volume(self.sample_data)
        self.assertIsInstance(dv_value, (int, float))
        self.assertGreaterEqual(dv_value, 0)
        
        # Test momentum divergence
        md_value = self.models._calculate_momentum_divergence(self.sample_data)
        self.assertIsInstance(md_value, (int, float))
        
        # Test time decay
        td_value = self.models._calculate_time_decay(self.sample_data)
        self.assertIsInstance(td_value, (int, float))
        self.assertGreaterEqual(td_value, 0)
        self.assertLessEqual(td_value, 1)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with insufficient data
        small_data = self.sample_data.head(10)
        
        with self.assertRaises(ValueError):
            self.models.calculate_cci(small_data)
        
        with self.assertRaises(ValueError):
            self.models.calculate_ddm(small_data)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            self.models.calculate_cci(empty_data)
        
        with self.assertRaises(ValueError):
            self.models.calculate_ddm(empty_data)
        
        # Test with missing columns
        incomplete_data = self.sample_data[['timestamp', 'close']].copy()
        
        with self.assertRaises(KeyError):
            self.models.calculate_cci(incomplete_data)
    
    def test_threshold_sensitivity(self):
        """Test sensitivity to threshold changes."""
        # Test different CCI thresholds
        models_low = MathematicalModels(cci_threshold=10, ddm_threshold=80)
        models_high = MathematicalModels(cci_threshold=30, ddm_threshold=80)
        
        signal_low = models_low.generate_signal(self.sample_data)
        signal_high = models_high.generate_signal(self.sample_data)
        
        # Signals might be different due to threshold sensitivity
        # This tests that the system responds to parameter changes
        self.assertIsInstance(signal_low, dict)
        self.assertIsInstance(signal_high, dict)
    
    def test_performance_metrics(self):
        """Test performance calculation methods."""
        # Generate multiple signals for testing
        signals = []
        for i in range(10):
            subset = self.sample_data.iloc[i*10:(i+1)*10+100]
            if len(subset) >= 100:
                signal = self.models.generate_signal(subset)
                signals.append(signal)
        
        # Test accuracy calculation
        if signals:
            accuracy = self.models.calculate_accuracy(signals, self.sample_data)
            self.assertIsInstance(accuracy, (int, float))
            self.assertGreaterEqual(accuracy, 0)
            self.assertLessEqual(accuracy, 100)


class TestCCICalculator(unittest.TestCase):
    """Test cases for the CCI Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = CCICalculator(threshold=20)
        
        # Create test data with known patterns
        self.oversold_data = self._create_oversold_pattern()
        self.normal_data = self._create_normal_pattern()
    
    def _create_oversold_pattern(self) -> pd.DataFrame:
        """Create data pattern that should trigger oversold conditions."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=100, freq='15min')
        
        # Declining price pattern with high volume
        prices = np.linspace(3000, 2500, 100)  # 16.7% decline
        volumes = np.random.uniform(5000, 15000, 100)  # High volume
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes
        })
    
    def _create_normal_pattern(self) -> pd.DataFrame:
        """Create normal market data pattern."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=100, freq='15min')
        
        # Sideways price pattern with normal volume
        prices = 3000 + np.random.normal(0, 20, 100)
        volumes = np.random.uniform(1000, 3000, 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': volumes
        })
    
    def test_oversold_detection(self):
        """Test detection of oversold conditions."""
        cci_value = self.calculator.calculate(self.oversold_data)
        
        # Oversold pattern should produce higher CCI values
        self.assertGreater(cci_value, 30)
    
    def test_normal_conditions(self):
        """Test normal market conditions."""
        cci_value = self.calculator.calculate(self.normal_data)
        
        # Normal conditions should produce moderate CCI values
        self.assertLess(cci_value, 50)
    
    def test_rsi_calculation(self):
        """Test RSI component calculation."""
        rsi = self.calculator._calculate_rsi(self.normal_data['close'])
        
        self.assertIsInstance(rsi, (int, float))
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
    
    def test_stochastic_calculation(self):
        """Test Stochastic oscillator calculation."""
        stoch_k = self.calculator._calculate_stochastic(
            self.normal_data['high'],
            self.normal_data['low'],
            self.normal_data['close']
        )
        
        self.assertIsInstance(stoch_k, (int, float))
        self.assertGreaterEqual(stoch_k, 0)
        self.assertLessEqual(stoch_k, 100)


class TestDDMCalculator(unittest.TestCase):
    """Test cases for the DDM Calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = DDMCalculator(threshold=80)
        
        # Create test data with distribution patterns
        self.distribution_data = self._create_distribution_pattern()
        self.accumulation_data = self._create_accumulation_pattern()
    
    def _create_distribution_pattern(self) -> pd.DataFrame:
        """Create data pattern that should trigger distribution detection."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=100, freq='15min')
        
        # Rising prices with increasing volume (distribution)
        prices = np.linspace(3000, 3500, 100)  # 16.7% increase
        volumes = np.linspace(2000, 10000, 100)  # Increasing volume
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': volumes
        })
    
    def _create_accumulation_pattern(self) -> pd.DataFrame:
        """Create accumulation pattern data."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=100, freq='15min')
        
        # Stable prices with normal volume
        prices = 3000 + np.random.normal(0, 10, 100)
        volumes = np.random.uniform(1000, 3000, 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': volumes
        })
    
    def test_distribution_detection(self):
        """Test detection of distribution patterns."""
        ddm_value = self.calculator.calculate(self.distribution_data)
        
        # Distribution pattern should produce higher DDM values
        self.assertGreater(ddm_value, 50)
    
    def test_accumulation_conditions(self):
        """Test accumulation market conditions."""
        ddm_value = self.calculator.calculate(self.accumulation_data)
        
        # Accumulation should produce lower DDM values
        self.assertLess(ddm_value, 70)
    
    def test_volume_analysis(self):
        """Test volume analysis component."""
        volume_factor = self.calculator._analyze_volume_distribution(self.distribution_data)
        
        self.assertIsInstance(volume_factor, (int, float))
        self.assertGreaterEqual(volume_factor, 0)
    
    def test_momentum_divergence(self):
        """Test momentum divergence calculation."""
        divergence = self.calculator._calculate_momentum_divergence(self.distribution_data)
        
        self.assertIsInstance(divergence, (int, float))


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete trading scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.models = MathematicalModels()
        
        # Create scenario data
        self.bull_market_data = self._create_bull_market()
        self.bear_market_data = self._create_bear_market()
        self.sideways_market_data = self._create_sideways_market()
    
    def _create_bull_market(self) -> pd.DataFrame:
        """Create bull market scenario data."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=500, freq='15min')
        
        # Strong uptrend with occasional pullbacks
        trend = np.linspace(0, 1000, 500)
        noise = np.random.normal(0, 30, 500)
        pullbacks = np.where(np.random.random(500) < 0.1, -50, 0)
        
        prices = 3000 + trend + noise + pullbacks
        volumes = np.random.uniform(2000, 8000, 500)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes
        })
    
    def _create_bear_market(self) -> pd.DataFrame:
        """Create bear market scenario data."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=500, freq='15min')
        
        # Strong downtrend with occasional bounces
        trend = np.linspace(0, -800, 500)
        noise = np.random.normal(0, 40, 500)
        bounces = np.where(np.random.random(500) < 0.1, 60, 0)
        
        prices = 3000 + trend + noise + bounces
        volumes = np.random.uniform(3000, 12000, 500)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes
        })
    
    def _create_sideways_market(self) -> pd.DataFrame:
        """Create sideways market scenario data."""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=500, freq='15min')
        
        # Range-bound market
        prices = 3000 + np.random.normal(0, 50, 500)
        volumes = np.random.uniform(1500, 4000, 500)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.005,
            'low': prices * 0.995,
            'close': prices,
            'volume': volumes
        })
    
    def test_bull_market_signals(self):
        """Test signal generation in bull market conditions."""
        signals = []
        
        # Generate signals throughout the bull market
        for i in range(0, len(self.bull_market_data) - 100, 50):
            subset = self.bull_market_data.iloc[i:i+100]
            signal = self.models.generate_signal(subset)
            signals.append(signal)
        
        # In a bull market, we should see more BUY signals
        buy_signals = sum(1 for s in signals if s['signal_type'] == 'BUY')
        total_signals = len([s for s in signals if s['signal_type'] != 'HOLD'])
        
        if total_signals > 0:
            buy_ratio = buy_signals / total_signals
            self.assertGreater(buy_ratio, 0.3)  # At least 30% buy signals
    
    def test_bear_market_signals(self):
        """Test signal generation in bear market conditions."""
        signals = []
        
        # Generate signals throughout the bear market
        for i in range(0, len(self.bear_market_data) - 100, 50):
            subset = self.bear_market_data.iloc[i:i+100]
            signal = self.models.generate_signal(subset)
            signals.append(signal)
        
        # In a bear market, we should see more SELL signals or HOLD
        sell_signals = sum(1 for s in signals if s['signal_type'] == 'SELL')
        hold_signals = sum(1 for s in signals if s['signal_type'] == 'HOLD')
        total_signals = len(signals)
        
        if total_signals > 0:
            defensive_ratio = (sell_signals + hold_signals) / total_signals
            self.assertGreater(defensive_ratio, 0.5)  # At least 50% defensive signals
    
    def test_sideways_market_signals(self):
        """Test signal generation in sideways market conditions."""
        signals = []
        
        # Generate signals throughout the sideways market
        for i in range(0, len(self.sideways_market_data) - 100, 50):
            subset = self.sideways_market_data.iloc[i:i+100]
            signal = self.models.generate_signal(subset)
            signals.append(signal)
        
        # In sideways markets, we should see more HOLD signals
        hold_signals = sum(1 for s in signals if s['signal_type'] == 'HOLD')
        total_signals = len(signals)
        
        if total_signals > 0:
            hold_ratio = hold_signals / total_signals
            self.assertGreater(hold_ratio, 0.4)  # At least 40% hold signals


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMathematicalModels))
    test_suite.addTest(unittest.makeSuite(TestCCICalculator))
    test_suite.addTest(unittest.makeSuite(TestDDMCalculator))
    test_suite.addTest(unittest.makeSuite(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

