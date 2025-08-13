"""
Sentiment Analysis Module for ETH/FDUSD Trading Bot
Advanced sentiment analysis from multiple data sources
"""

import numpy as np
import pandas as pd
import re
import json
import requests
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import threading
from collections import deque
import logging

# Text processing
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available, using simplified sentiment analysis")

# Machine learning for sentiment
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available for advanced sentiment models")


@dataclass
class SentimentData:
    """Sentiment data point"""
    timestamp: datetime
    source: str
    text: str
    sentiment_score: float  # -1 (very negative) to 1 (very positive)
    confidence: float
    volume: int  # Number of mentions/interactions
    keywords: List[str]
    metadata: Dict[str, Any] = None


@dataclass
class SentimentSummary:
    """Aggregated sentiment summary"""
    timestamp: datetime
    overall_sentiment: float
    confidence: float
    bullish_ratio: float
    bearish_ratio: float
    neutral_ratio: float
    volume_weighted_sentiment: float
    trending_keywords: List[str]
    source_breakdown: Dict[str, float]
    sentiment_momentum: float  # Change in sentiment over time


class CryptoSentimentLexicon:
    """
    Cryptocurrency-specific sentiment lexicon
    Enhanced vocabulary for crypto market sentiment
    """
    
    def __init__(self):
        self.positive_words = {
            # Price movement
            'moon', 'mooning', 'pump', 'pumping', 'rally', 'surge', 'breakout', 'bullish', 'bull',
            'green', 'gains', 'profit', 'up', 'rise', 'rising', 'climb', 'climbing', 'soar', 'soaring',
            
            # Market sentiment
            'hodl', 'diamond hands', 'buy the dip', 'accumulate', 'accumulating', 'strong hands',
            'institutional adoption', 'mainstream adoption', 'mass adoption', 'adoption',
            
            # Technical analysis
            'support', 'resistance broken', 'golden cross', 'bullish divergence', 'higher highs',
            'higher lows', 'uptrend', 'momentum', 'volume spike',
            
            # Fundamental
            'upgrade', 'partnership', 'integration', 'development', 'innovation', 'breakthrough',
            'milestone', 'achievement', 'success', 'progress', 'growth', 'expansion',
            
            # General positive
            'amazing', 'awesome', 'fantastic', 'excellent', 'great', 'good', 'positive',
            'optimistic', 'confident', 'excited', 'bullish', 'promising', 'potential'
        }
        
        self.negative_words = {
            # Price movement
            'dump', 'dumping', 'crash', 'crashing', 'fall', 'falling', 'drop', 'dropping',
            'bearish', 'bear', 'red', 'loss', 'losses', 'down', 'decline', 'declining',
            'plummet', 'plummeting', 'collapse', 'collapsing',
            
            # Market sentiment
            'panic', 'fear', 'fud', 'weak hands', 'paper hands', 'sell off', 'selling pressure',
            'capitulation', 'despair', 'bottom', 'oversold',
            
            # Technical analysis
            'resistance', 'support broken', 'death cross', 'bearish divergence', 'lower highs',
            'lower lows', 'downtrend', 'breakdown', 'volume decline',
            
            # Fundamental
            'hack', 'exploit', 'vulnerability', 'regulation', 'ban', 'restriction', 'concern',
            'issue', 'problem', 'delay', 'postpone', 'cancel', 'failure', 'risk',
            
            # General negative
            'terrible', 'awful', 'bad', 'negative', 'pessimistic', 'worried', 'concerned',
            'disappointed', 'frustrated', 'angry', 'scared', 'uncertain', 'doubt'
        }
        
        self.neutral_words = {
            'stable', 'sideways', 'consolidation', 'range', 'waiting', 'watching',
            'analysis', 'technical', 'fundamental', 'chart', 'pattern', 'indicator'
        }
        
        # Create scoring dictionary
        self.word_scores = {}
        
        for word in self.positive_words:
            self.word_scores[word] = 1.0
        
        for word in self.negative_words:
            self.word_scores[word] = -1.0
        
        for word in self.neutral_words:
            self.word_scores[word] = 0.0
    
    def get_word_sentiment(self, word: str) -> float:
        """Get sentiment score for a word"""
        word_lower = word.lower()
        return self.word_scores.get(word_lower, 0.0)


class TextPreprocessor:
    """
    Text preprocessing for sentiment analysis
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
        
        # Add crypto-specific stop words
        self.stop_words.update({
            'eth', 'ethereum', 'crypto', 'cryptocurrency', 'blockchain', 'btc', 'bitcoin'
        })
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep the content)
        text = re.sub(r'@\w+|#', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        if not NLTK_AVAILABLE:
            return text.split()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key terms from text"""
        tokens = self.tokenize_and_lemmatize(text)
        
        # Simple frequency-based keyword extraction
        word_freq = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]


class SentimentAnalyzer:
    """
    Advanced sentiment analyzer for cryptocurrency content
    """
    
    def __init__(self):
        self.crypto_lexicon = CryptoSentimentLexicon()
        self.preprocessor = TextPreprocessor()
        self.vader_analyzer = SentimentIntensityAnalyzer() if NLTK_AVAILABLE else None
        
        # Custom sentiment model (would be trained on crypto-specific data)
        self.custom_model = None
        self.vectorizer = TfidfVectorizer(max_features=1000) if SKLEARN_AVAILABLE else None
        
    def analyze_text(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text
        Returns (sentiment_score, confidence)
        """
        if not text or len(text.strip()) == 0:
            return 0.0, 0.0
        
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Multiple sentiment analysis approaches
        scores = []
        confidences = []
        
        # 1. Crypto-specific lexicon analysis
        lexicon_score, lexicon_conf = self._lexicon_sentiment(cleaned_text)
        scores.append(lexicon_score)
        confidences.append(lexicon_conf)
        
        # 2. VADER sentiment (if available)
        if self.vader_analyzer:
            vader_score, vader_conf = self._vader_sentiment(text)
            scores.append(vader_score)
            confidences.append(vader_conf)
        
        # 3. Custom model (if trained)
        if self.custom_model and self.vectorizer:
            custom_score, custom_conf = self._custom_model_sentiment(cleaned_text)
            scores.append(custom_score)
            confidences.append(custom_conf)
        
        # Combine scores (weighted average)
        if scores:
            weights = np.array(confidences)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
            
            final_score = np.average(scores, weights=weights)
            final_confidence = np.mean(confidences)
            
            return final_score, final_confidence
        
        return 0.0, 0.0
    
    def _lexicon_sentiment(self, text: str) -> Tuple[float, float]:
        """Sentiment analysis using crypto lexicon"""
        tokens = self.preprocessor.tokenize_and_lemmatize(text)
        
        if not tokens:
            return 0.0, 0.0
        
        scores = []
        for token in tokens:
            score = self.crypto_lexicon.get_word_sentiment(token)
            if score != 0.0:  # Only count words with sentiment
                scores.append(score)
        
        if not scores:
            return 0.0, 0.0
        
        # Calculate sentiment score
        sentiment_score = np.mean(scores)
        
        # Calculate confidence based on number of sentiment words found
        confidence = min(1.0, len(scores) / len(tokens))
        
        return sentiment_score, confidence
    
    def _vader_sentiment(self, text: str) -> Tuple[float, float]:
        """VADER sentiment analysis"""
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Convert compound score to our scale (-1 to 1)
        sentiment_score = scores['compound']
        
        # Use the absolute value of compound score as confidence
        confidence = abs(sentiment_score)
        
        return sentiment_score, confidence
    
    def _custom_model_sentiment(self, text: str) -> Tuple[float, float]:
        """Custom model sentiment analysis"""
        # This would use a trained model on crypto-specific data
        # For now, return neutral
        return 0.0, 0.0
    
    def batch_analyze(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Analyze sentiment for multiple texts"""
        results = []
        for text in texts:
            sentiment, confidence = self.analyze_text(text)
            results.append((sentiment, confidence))
        return results


class SentimentDataCollector:
    """
    Collects sentiment data from various sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analyzer = SentimentAnalyzer()
        self.logger = logging.getLogger('SentimentCollector')
        
        # Data storage
        self.sentiment_buffer = deque(maxlen=10000)
        self.buffer_lock = threading.Lock()
        
        # Collection threads
        self.collection_threads = []
        self.is_collecting = False
    
    def start_collection(self):
        """Start sentiment data collection"""
        self.is_collecting = True
        
        # Start collection threads for different sources
        if self.config.get('collect_reddit', False):
            reddit_thread = threading.Thread(target=self._collect_reddit_sentiment)
            reddit_thread.daemon = True
            reddit_thread.start()
            self.collection_threads.append(reddit_thread)
        
        if self.config.get('collect_twitter', False):
            twitter_thread = threading.Thread(target=self._collect_twitter_sentiment)
            twitter_thread.daemon = True
            twitter_thread.start()
            self.collection_threads.append(twitter_thread)
        
        if self.config.get('collect_news', False):
            news_thread = threading.Thread(target=self._collect_news_sentiment)
            news_thread.daemon = True
            news_thread.start()
            self.collection_threads.append(news_thread)
        
        self.logger.info("Sentiment collection started")
    
    def stop_collection(self):
        """Stop sentiment data collection"""
        self.is_collecting = False
        self.logger.info("Sentiment collection stopped")
    
    def _collect_reddit_sentiment(self):
        """Collect sentiment from Reddit (simplified)"""
        while self.is_collecting:
            try:
                # This would integrate with Reddit API
                # For now, simulate data collection
                time.sleep(300)  # Collect every 5 minutes
                
                # Simulate Reddit posts about Ethereum
                sample_posts = [
                    "ETH is looking bullish, great momentum!",
                    "Ethereum upgrade is going to be huge",
                    "Not sure about this market, seems risky",
                    "Diamond hands on ETH, hodling strong",
                    "Market looks bearish, might be time to sell"
                ]
                
                for post in sample_posts:
                    sentiment, confidence = self.analyzer.analyze_text(post)
                    keywords = self.analyzer.preprocessor.extract_keywords(post, 5)
                    
                    sentiment_data = SentimentData(
                        timestamp=datetime.now(),
                        source='reddit',
                        text=post,
                        sentiment_score=sentiment,
                        confidence=confidence,
                        volume=1,
                        keywords=keywords
                    )
                    
                    with self.buffer_lock:
                        self.sentiment_buffer.append(sentiment_data)
                
            except Exception as e:
                self.logger.error(f"Error collecting Reddit sentiment: {e}")
                time.sleep(60)
    
    def _collect_twitter_sentiment(self):
        """Collect sentiment from Twitter (simplified)"""
        while self.is_collecting:
            try:
                # This would integrate with Twitter API
                time.sleep(180)  # Collect every 3 minutes
                
                # Simulate Twitter data
                sample_tweets = [
                    "ETH to the moon! 🚀",
                    "Ethereum network congestion is concerning",
                    "Just bought more ETH on the dip",
                    "Bearish on crypto right now",
                    "ETH staking rewards looking good"
                ]
                
                for tweet in sample_tweets:
                    sentiment, confidence = self.analyzer.analyze_text(tweet)
                    keywords = self.analyzer.preprocessor.extract_keywords(tweet, 3)
                    
                    sentiment_data = SentimentData(
                        timestamp=datetime.now(),
                        source='twitter',
                        text=tweet,
                        sentiment_score=sentiment,
                        confidence=confidence,
                        volume=1,
                        keywords=keywords
                    )
                    
                    with self.buffer_lock:
                        self.sentiment_buffer.append(sentiment_data)
                
            except Exception as e:
                self.logger.error(f"Error collecting Twitter sentiment: {e}")
                time.sleep(60)
    
    def _collect_news_sentiment(self):
        """Collect sentiment from news sources (simplified)"""
        while self.is_collecting:
            try:
                # This would integrate with news APIs
                time.sleep(600)  # Collect every 10 minutes
                
                # Simulate news headlines
                sample_news = [
                    "Ethereum sees institutional adoption surge",
                    "Regulatory concerns impact crypto markets",
                    "Major DeFi protocol launches on Ethereum",
                    "Market volatility continues amid uncertainty",
                    "Ethereum upgrade promises improved scalability"
                ]
                
                for headline in sample_news:
                    sentiment, confidence = self.analyzer.analyze_text(headline)
                    keywords = self.analyzer.preprocessor.extract_keywords(headline, 5)
                    
                    sentiment_data = SentimentData(
                        timestamp=datetime.now(),
                        source='news',
                        text=headline,
                        sentiment_score=sentiment,
                        confidence=confidence,
                        volume=1,
                        keywords=keywords
                    )
                    
                    with self.buffer_lock:
                        self.sentiment_buffer.append(sentiment_data)
                
            except Exception as e:
                self.logger.error(f"Error collecting news sentiment: {e}")
                time.sleep(60)
    
    def get_recent_sentiment(self, hours: int = 24) -> List[SentimentData]:
        """Get recent sentiment data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.buffer_lock:
            recent_data = [
                data for data in self.sentiment_buffer
                if data.timestamp >= cutoff_time
            ]
        
        return recent_data


class SentimentAggregator:
    """
    Aggregates and analyzes sentiment data
    """
    
    def __init__(self):
        self.logger = logging.getLogger('SentimentAggregator')
    
    def aggregate_sentiment(self, sentiment_data: List[SentimentData], 
                          time_window: int = 60) -> SentimentSummary:
        """
        Aggregate sentiment data over time window (minutes)
        """
        if not sentiment_data:
            return self._empty_summary()
        
        # Filter data within time window
        cutoff_time = datetime.now() - timedelta(minutes=time_window)
        recent_data = [data for data in sentiment_data if data.timestamp >= cutoff_time]
        
        if not recent_data:
            return self._empty_summary()
        
        # Calculate overall sentiment
        sentiments = [data.sentiment_score for data in recent_data]
        confidences = [data.confidence for data in recent_data]
        volumes = [data.volume for data in recent_data]
        
        overall_sentiment = np.mean(sentiments)
        confidence = np.mean(confidences)
        
        # Calculate sentiment distribution
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        total_count = len(sentiments)
        bullish_ratio = positive_count / total_count if total_count > 0 else 0
        bearish_ratio = negative_count / total_count if total_count > 0 else 0
        neutral_ratio = neutral_count / total_count if total_count > 0 else 0
        
        # Volume-weighted sentiment
        if sum(volumes) > 0:
            volume_weighted_sentiment = np.average(sentiments, weights=volumes)
        else:
            volume_weighted_sentiment = overall_sentiment
        
        # Extract trending keywords
        all_keywords = []
        for data in recent_data:
            all_keywords.extend(data.keywords)
        
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        trending_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        trending_keywords = [keyword for keyword, freq in trending_keywords]
        
        # Source breakdown
        source_sentiments = {}
        for data in recent_data:
            if data.source not in source_sentiments:
                source_sentiments[data.source] = []
            source_sentiments[data.source].append(data.sentiment_score)
        
        source_breakdown = {
            source: np.mean(sentiments)
            for source, sentiments in source_sentiments.items()
        }
        
        # Calculate sentiment momentum
        sentiment_momentum = self._calculate_momentum(sentiment_data, time_window)
        
        return SentimentSummary(
            timestamp=datetime.now(),
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            bullish_ratio=bullish_ratio,
            bearish_ratio=bearish_ratio,
            neutral_ratio=neutral_ratio,
            volume_weighted_sentiment=volume_weighted_sentiment,
            trending_keywords=trending_keywords,
            source_breakdown=source_breakdown,
            sentiment_momentum=sentiment_momentum
        )
    
    def _calculate_momentum(self, sentiment_data: List[SentimentData], 
                           time_window: int) -> float:
        """Calculate sentiment momentum (change over time)"""
        if len(sentiment_data) < 2:
            return 0.0
        
        # Split data into two halves
        cutoff_time = datetime.now() - timedelta(minutes=time_window)
        mid_time = cutoff_time + timedelta(minutes=time_window // 2)
        
        early_data = [data for data in sentiment_data if cutoff_time <= data.timestamp < mid_time]
        late_data = [data for data in sentiment_data if data.timestamp >= mid_time]
        
        if not early_data or not late_data:
            return 0.0
        
        early_sentiment = np.mean([data.sentiment_score for data in early_data])
        late_sentiment = np.mean([data.sentiment_score for data in late_data])
        
        return late_sentiment - early_sentiment
    
    def _empty_summary(self) -> SentimentSummary:
        """Return empty sentiment summary"""
        return SentimentSummary(
            timestamp=datetime.now(),
            overall_sentiment=0.0,
            confidence=0.0,
            bullish_ratio=0.0,
            bearish_ratio=0.0,
            neutral_ratio=1.0,
            volume_weighted_sentiment=0.0,
            trending_keywords=[],
            source_breakdown={},
            sentiment_momentum=0.0
        )
    
    def get_sentiment_signal(self, summary: SentimentSummary) -> Dict[str, Any]:
        """
        Generate trading signal based on sentiment
        """
        signal_strength = 0
        signal_type = 'NEUTRAL'
        
        # Strong positive sentiment
        if summary.overall_sentiment > 0.3 and summary.confidence > 0.6:
            if summary.bullish_ratio > 0.6 and summary.sentiment_momentum > 0.1:
                signal_strength = 75
                signal_type = 'BULLISH'
            elif summary.bullish_ratio > 0.5:
                signal_strength = 50
                signal_type = 'BULLISH'
        
        # Strong negative sentiment
        elif summary.overall_sentiment < -0.3 and summary.confidence > 0.6:
            if summary.bearish_ratio > 0.6 and summary.sentiment_momentum < -0.1:
                signal_strength = 75
                signal_type = 'BEARISH'
            elif summary.bearish_ratio > 0.5:
                signal_strength = 50
                signal_type = 'BEARISH'
        
        # Extreme sentiment (contrarian signal)
        if summary.overall_sentiment > 0.7 or summary.overall_sentiment < -0.7:
            signal_type = 'CONTRARIAN_' + signal_type
            signal_strength = min(signal_strength + 25, 100)
        
        return {
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'sentiment_score': summary.overall_sentiment,
            'confidence': summary.confidence,
            'momentum': summary.sentiment_momentum,
            'bullish_ratio': summary.bullish_ratio,
            'bearish_ratio': summary.bearish_ratio
        }


if __name__ == "__main__":
    # Example usage
    print("Sentiment Analysis Module for ETH/FDUSD Trading Bot")
    
    # Initialize components
    analyzer = SentimentAnalyzer()
    aggregator = SentimentAggregator()
    
    # Test sentiment analysis
    test_texts = [
        "ETH is mooning! Great bullish momentum, diamond hands!",
        "Market crash incoming, bearish signals everywhere",
        "Ethereum upgrade looks promising for long-term growth",
        "FUD spreading, but I'm still hodling strong",
        "Neutral market conditions, waiting for breakout"
    ]
    
    print("\nSentiment Analysis Results:")
    for text in test_texts:
        sentiment, confidence = analyzer.analyze_text(text)
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {sentiment:.3f}, Confidence: {confidence:.3f}")
        print()
    
    # Test data collection (simulation)
    config = {
        'collect_reddit': True,
        'collect_twitter': True,
        'collect_news': True
    }
    
    collector = SentimentDataCollector(config)
    print("Sentiment data collection configured")
    
    print("\nSentiment analysis module initialized successfully!")

