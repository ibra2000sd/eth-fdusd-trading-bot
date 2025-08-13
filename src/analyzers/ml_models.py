"""
Machine Learning Models for ETH/FDUSD Trading Bot
Advanced ML algorithms for market prediction and adaptive learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# Deep learning (if available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available, using scikit-learn models only")


@dataclass
class MLPrediction:
    """ML model prediction result"""
    timestamp: datetime
    predicted_price: float
    confidence: float
    prediction_horizon: int  # minutes
    model_name: str
    features_used: List[str]
    additional_metrics: Dict[str, float] = None


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    mse: float
    mae: float
    r2: float
    accuracy_1h: float  # 1-hour prediction accuracy
    accuracy_4h: float  # 4-hour prediction accuracy
    accuracy_24h: float # 24-hour prediction accuracy
    last_updated: datetime


class FeatureEngineering:
    """
    Advanced feature engineering for market data
    Creates sophisticated features for ML models
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from market data
        """
        features_df = df.copy()
        
        # Price-based features
        features_df = self._add_price_features(features_df)
        
        # Volume-based features
        features_df = self._add_volume_features(features_df)
        
        # Technical indicator features
        features_df = self._add_technical_features(features_df)
        
        # Time-based features
        features_df = self._add_time_features(features_df)
        
        # Statistical features
        features_df = self._add_statistical_features(features_df)
        
        # Momentum features
        features_df = self._add_momentum_features(features_df)
        
        # Volatility features
        features_df = self._add_volatility_features(features_df)
        
        # Market microstructure features
        features_df = self._add_microstructure_features(features_df)
        
        return features_df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_15'] = df['close'].pct_change(15)
        df['returns_60'] = df['close'].pct_change(60)
        
        # Log returns
        df['log_returns_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_returns_5'] = np.log(df['close'] / df['close'].shift(5))
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap analysis
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume ratios
        df['volume_sma_5'] = df['volume'].rolling(5).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
        df['volume_ratio_20'] = df['volume'] / df['volume_sma_20']
        
        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['returns_1']
        
        # On-balance volume
        df['obv'] = (df['volume'] * np.sign(df['returns_1'])).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # Moving average convergence/divergence
        df['ma_5_20_diff'] = df['sma_5'] - df['sma_20']
        df['ma_10_50_diff'] = df['sma_10'] - df['sma_50']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['month'] = pd.to_datetime(df['timestamp']).dt.month
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'mean_{window}'] = df['close'].rolling(window).mean()
            df[f'std_{window}'] = df['close'].rolling(window).std()
            df[f'skew_{window}'] = df['close'].rolling(window).skew()
            df[f'kurt_{window}'] = df['close'].rolling(window).kurt()
            df[f'min_{window}'] = df['close'].rolling(window).min()
            df[f'max_{window}'] = df['close'].rolling(window).max()
        
        # Z-scores
        df['zscore_5'] = (df['close'] - df['mean_5']) / df['std_5']
        df['zscore_20'] = (df['close'] - df['mean_20']) / df['std_20']
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features"""
        # Rate of change
        for period in [1, 5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        # Momentum oscillators
        df['momentum_5'] = df['close'] / df['close'].shift(5)
        df['momentum_10'] = df['close'] / df['close'].shift(10)
        
        # Acceleration
        df['acceleration'] = df['returns_1'] - df['returns_1'].shift(1)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        # Realized volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns_1'].rolling(window).std()
        
        # Parkinson volatility (high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * np.log(df['high'] / df['low']) ** 2
        ).rolling(20).mean()
        
        # Volatility ratios
        df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Spread proxy
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Amihud illiquidity
        df['amihud'] = abs(df['returns_1']) / df['volume']
        
        # Price impact
        df['price_impact'] = df['returns_1'] / np.log(df['volume'])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'close', 
                        prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and targets for ML training
        """
        # Create features
        features_df = self.create_features(df)
        
        # Create target (future price)
        features_df['target'] = features_df[target_col].shift(-prediction_horizon)
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        # Select feature columns (exclude original OHLCV and target)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].values
        y = features_df['target'].values
        
        self.feature_names = feature_cols
        
        return X, y, feature_cols


class EnsemblePredictor:
    """
    Ensemble ML predictor combining multiple models
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_engineering = FeatureEngineering()
        self.performance_history = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        # Traditional ML models
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.models['svr'] = SVR(
            kernel='rbf',
            C=1.0,
            gamma='scale'
        )
        
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=42
        )
        
        # Ensemble model
        self.models['ensemble'] = VotingRegressor([
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boosting']),
            ('svr', self.models['svr'])
        ])
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
        
        # Initialize LSTM if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self._initialize_lstm()
    
    def _initialize_lstm(self):
        """Initialize LSTM model"""
        def create_lstm_model(input_shape):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        
        self.lstm_model_creator = create_lstm_model
    
    def train_models(self, df: pd.DataFrame, prediction_horizon: int = 1) -> Dict[str, ModelPerformance]:
        """
        Train all models on historical data
        """
        # Prepare features
        X, y, feature_names = self.feature_engineering.prepare_features(
            df, prediction_horizon=prediction_horizon
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        performance_results = {}
        
        # Train traditional ML models
        for model_name, model in self.models.items():
            try:
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                performance = self._evaluate_model(y_test, y_pred, model_name)
                performance_results[model_name] = performance
                
                print(f"Trained {model_name}: R² = {performance.r2:.4f}, MAE = {performance.mae:.4f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        # Train LSTM if available
        if TENSORFLOW_AVAILABLE:
            try:
                lstm_performance = self._train_lstm(X_train, X_test, y_train, y_test)
                performance_results['lstm'] = lstm_performance
            except Exception as e:
                print(f"Error training LSTM: {e}")
        
        self.performance_history[datetime.now()] = performance_results
        return performance_results
    
    def _train_lstm(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Train LSTM model"""
        # Reshape data for LSTM (samples, timesteps, features)
        # For simplicity, using sequence length of 1
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Create and train model
        lstm_model = self.lstm_model_creator((1, X_train.shape[1]))
        
        # Callbacks
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5)
        
        # Train
        history = lstm_model.fit(
            X_train_lstm, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_lstm, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Predict and evaluate
        y_pred = lstm_model.predict(X_test_lstm).flatten()
        
        # Store model
        self.models['lstm'] = lstm_model
        
        return self._evaluate_model(y_test, y_pred, 'lstm')
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       model_name: str) -> ModelPerformance:
        """Evaluate model performance"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate directional accuracy (simplified)
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) if len(direction_true) > 0 else 0.0
        
        return ModelPerformance(
            model_name=model_name,
            mse=mse,
            mae=mae,
            r2=r2,
            accuracy_1h=directional_accuracy,
            accuracy_4h=directional_accuracy,  # Simplified
            accuracy_24h=directional_accuracy,  # Simplified
            last_updated=datetime.now()
        )
    
    def predict(self, df: pd.DataFrame, prediction_horizon: int = 1) -> Dict[str, MLPrediction]:
        """
        Generate predictions from all models
        """
        # Prepare features for latest data point
        features_df = self.feature_engineering.create_features(df)
        
        # Get feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Get latest features
        latest_features = features_df[feature_cols].iloc[-1:].values
        
        predictions = {}
        
        # Generate predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                    # LSTM prediction
                    features_scaled = latest_features.reshape((1, 1, latest_features.shape[1]))
                    pred = model.predict(features_scaled)[0][0]
                else:
                    # Traditional ML prediction
                    features_scaled = self.scalers[model_name].transform(latest_features)
                    pred = model.predict(features_scaled)[0]
                
                # Calculate confidence (simplified)
                confidence = self._calculate_confidence(model_name, latest_features)
                
                predictions[model_name] = MLPrediction(
                    timestamp=datetime.now(),
                    predicted_price=pred,
                    confidence=confidence,
                    prediction_horizon=prediction_horizon,
                    model_name=model_name,
                    features_used=feature_cols
                )
                
            except Exception as e:
                print(f"Error generating prediction for {model_name}: {e}")
        
        return predictions
    
    def _calculate_confidence(self, model_name: str, features: np.ndarray) -> float:
        """Calculate prediction confidence (simplified)"""
        # This is a simplified confidence calculation
        # In practice, you might use prediction intervals, ensemble variance, etc.
        
        if model_name in self.performance_history:
            latest_performance = list(self.performance_history.values())[-1]
            if model_name in latest_performance:
                r2 = latest_performance[model_name].r2
                return max(0, min(100, r2 * 100))
        
        return 50.0  # Default confidence
    
    def get_ensemble_prediction(self, predictions: Dict[str, MLPrediction]) -> MLPrediction:
        """
        Create ensemble prediction from individual model predictions
        """
        if not predictions:
            return None
        
        # Weight predictions by confidence
        weighted_sum = 0
        total_weight = 0
        
        for pred in predictions.values():
            weight = pred.confidence / 100.0
            weighted_sum += pred.predicted_price * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        ensemble_price = weighted_sum / total_weight
        ensemble_confidence = np.mean([pred.confidence for pred in predictions.values()])
        
        return MLPrediction(
            timestamp=datetime.now(),
            predicted_price=ensemble_price,
            confidence=ensemble_confidence,
            prediction_horizon=list(predictions.values())[0].prediction_horizon,
            model_name='ensemble',
            features_used=list(predictions.values())[0].features_used,
            additional_metrics={
                'individual_predictions': {name: pred.predicted_price for name, pred in predictions.items()},
                'prediction_variance': np.var([pred.predicted_price for pred in predictions.values()])
            }
        )
    
    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': {},
            'scalers': self.scalers,
            'feature_names': self.feature_engineering.feature_names,
            'performance_history': self.performance_history
        }
        
        # Save traditional ML models
        for name, model in self.models.items():
            if name != 'lstm':
                model_data['models'][name] = model
        
        # Save to file
        joblib.dump(model_data, filepath)
        
        # Save LSTM separately if it exists
        if 'lstm' in self.models and TENSORFLOW_AVAILABLE:
            lstm_path = filepath.replace('.pkl', '_lstm.h5')
            self.models['lstm'].save(lstm_path)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            model_data = joblib.load(filepath)
            
            self.models.update(model_data['models'])
            self.scalers = model_data['scalers']
            self.feature_engineering.feature_names = model_data['feature_names']
            self.performance_history = model_data['performance_history']
            
            # Load LSTM if it exists
            lstm_path = filepath.replace('.pkl', '_lstm.h5')
            if os.path.exists(lstm_path) and TENSORFLOW_AVAILABLE:
                self.models['lstm'] = tf.keras.models.load_model(lstm_path)
            
            print(f"Models loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading models: {e}")


class AdaptiveLearning:
    """
    Adaptive learning system that continuously improves model performance
    """
    
    def __init__(self, predictor: EnsemblePredictor):
        self.predictor = predictor
        self.prediction_history = []
        self.performance_tracker = {}
        self.retraining_threshold = 0.1  # Retrain if performance drops by 10%
        
    def track_prediction(self, prediction: MLPrediction, actual_price: float):
        """Track prediction accuracy"""
        error = abs(prediction.predicted_price - actual_price) / actual_price
        
        self.prediction_history.append({
            'timestamp': prediction.timestamp,
            'model_name': prediction.model_name,
            'predicted_price': prediction.predicted_price,
            'actual_price': actual_price,
            'error': error,
            'confidence': prediction.confidence
        })
        
        # Update performance tracker
        if prediction.model_name not in self.performance_tracker:
            self.performance_tracker[prediction.model_name] = []
        
        self.performance_tracker[prediction.model_name].append(error)
        
        # Keep only recent predictions (last 1000)
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def should_retrain(self, model_name: str) -> bool:
        """Determine if model should be retrained"""
        if model_name not in self.performance_tracker:
            return False
        
        recent_errors = self.performance_tracker[model_name][-100:]  # Last 100 predictions
        
        if len(recent_errors) < 50:  # Need minimum data
            return False
        
        recent_avg_error = np.mean(recent_errors)
        historical_avg_error = np.mean(self.performance_tracker[model_name][:-100])
        
        # Retrain if recent performance is significantly worse
        return recent_avg_error > historical_avg_error * (1 + self.retraining_threshold)
    
    def get_model_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all models"""
        summary = {}
        
        for model_name, errors in self.performance_tracker.items():
            if len(errors) > 0:
                summary[model_name] = {
                    'avg_error': np.mean(errors),
                    'std_error': np.std(errors),
                    'min_error': np.min(errors),
                    'max_error': np.max(errors),
                    'recent_avg_error': np.mean(errors[-50:]) if len(errors) >= 50 else np.mean(errors)
                }
        
        return summary


if __name__ == "__main__":
    # Example usage
    print("Machine Learning Models for ETH/FDUSD Trading Bot")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='15T')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': 4000 + np.cumsum(np.random.randn(len(dates)) * 10),
        'high': 4000 + np.cumsum(np.random.randn(len(dates)) * 10) + 20,
        'low': 4000 + np.cumsum(np.random.randn(len(dates)) * 10) - 20,
        'close': 4000 + np.cumsum(np.random.randn(len(dates)) * 10),
        'volume': np.random.exponential(1000, len(dates))
    })
    
    # Initialize predictor
    config = {'prediction_horizons': [1, 4, 16]}  # 15min, 1h, 4h
    predictor = EnsemblePredictor(config)
    
    print("Training models...")
    performance = predictor.train_models(sample_data, prediction_horizon=1)
    
    print("\nModel Performance:")
    for model_name, perf in performance.items():
        print(f"{model_name}: R² = {perf.r2:.4f}, MAE = {perf.mae:.4f}")
    
    print("\nGenerating predictions...")
    predictions = predictor.predict(sample_data, prediction_horizon=1)
    
    for model_name, pred in predictions.items():
        print(f"{model_name}: ${pred.predicted_price:.2f} (confidence: {pred.confidence:.1f}%)")
    
    # Ensemble prediction
    ensemble_pred = predictor.get_ensemble_prediction(predictions)
    if ensemble_pred:
        print(f"\nEnsemble: ${ensemble_pred.predicted_price:.2f} (confidence: {ensemble_pred.confidence:.1f}%)")
    
    print("\nML models initialized successfully!")

