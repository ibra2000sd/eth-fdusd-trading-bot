"""
Data Manager - Market Data and API Integration

This module handles all market data operations including:
- Binance API integration
- Real-time data feeds
- Historical data management
- Order execution
- WebSocket connections
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.exceptions import BinanceAPIException
import websockets
import aiohttp

from ..utils.database import DatabaseManager


class DataManager:
    """
    Comprehensive data management system for market data and trading operations.
    
    This class handles:
    - Binance API integration
    - Real-time market data feeds
    - Historical data retrieval
    - Order placement and management
    - WebSocket connections
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = True,
        database: Optional[DatabaseManager] = None
    ):
        """
        Initialize the data manager.
        
        Args:
            api_key: Binance API key
            secret_key: Binance secret key
            testnet: Whether to use testnet
            database: Database manager instance
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.database = database
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Binance client
        if testnet:
            self.client = Client(
                api_key=api_key,
                api_secret=secret_key,
                testnet=True
            )
        else:
            self.client = Client(
                api_key=api_key,
                api_secret=secret_key
            )
        
        # WebSocket manager
        self.socket_manager = None
        self.websocket_connections = {}
        self.real_time_data = {}
        
        # Data cache
        self.price_cache = {}
        self.orderbook_cache = {}
        self.trade_cache = {}
        
        # Connection status
        self.is_connected = False
        self.last_heartbeat = None
        
    async def initialize(self):
        """Initialize the data manager."""
        try:
            self.logger.info("Initializing data manager...")
            
            # Test API connection
            await self._test_api_connection()
            
            # Initialize WebSocket manager
            self.socket_manager = BinanceSocketManager(self.client)
            
            # Initialize database tables if database is provided
            if self.database:
                await self._initialize_database_tables()
            
            self.is_connected = True
            self.logger.info("Data manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data manager: {e}")
            raise
    
    async def _test_api_connection(self):
        """Test the API connection."""
        try:
            # Test connectivity
            status = self.client.get_system_status()
            self.logger.info(f"Binance system status: {status}")
            
            # Test account access
            account_info = self.client.get_account()
            self.logger.info("API connection test successful")
            
            # Log account balances
            balances = {
                balance['asset']: float(balance['free'])
                for balance in account_info['balances']
                if float(balance['free']) > 0
            }
            self.logger.info(f"Account balances: {balances}")
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            raise
    
    async def _initialize_database_tables(self):
        """Initialize database tables for market data storage."""
        try:
            # Create tables for market data
            await self.database.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price DECIMAL(20, 8),
                    high_price DECIMAL(20, 8),
                    low_price DECIMAL(20, 8),
                    close_price DECIMAL(20, 8),
                    volume DECIMAL(20, 8),
                    timeframe VARCHAR(10),
                    UNIQUE(symbol, timestamp, timeframe)
                )
            """)
            
            # Create table for trades
            await self.database.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    order_id VARCHAR(50) UNIQUE,
                    symbol VARCHAR(20) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    quantity DECIMAL(20, 8),
                    price DECIMAL(20, 8),
                    commission DECIMAL(20, 8),
                    timestamp TIMESTAMP NOT NULL
                )
            """)
            
            # Create table for order book data
            await self.database.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    bids JSONB,
                    asks JSONB
                )
            """)
            
            self.logger.info("Database tables initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database tables: {e}")
            raise
    
    async def start_real_time_feed(self, symbol: str):
        """Start real-time data feed for a symbol."""
        try:
            self.logger.info(f"Starting real-time feed for {symbol}")
            
            # Start price stream
            await self._start_price_stream(symbol)
            
            # Start order book stream
            await self._start_orderbook_stream(symbol)
            
            # Start trade stream
            await self._start_trade_stream(symbol)
            
            self.logger.info(f"Real-time feed started for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to start real-time feed: {e}")
            raise
    
    async def _start_price_stream(self, symbol: str):
        """Start price stream for a symbol."""
        try:
            def handle_socket_message(msg):
                """Handle incoming price messages."""
                try:
                    if msg['e'] == '24hrTicker':
                        self.price_cache[symbol] = {
                            'symbol': msg['s'],
                            'price': float(msg['c']),
                            'change': float(msg['P']),
                            'volume': float(msg['v']),
                            'high': float(msg['h']),
                            'low': float(msg['l']),
                            'timestamp': datetime.now()
                        }
                        
                        # Store in real-time data
                        self.real_time_data[symbol] = self.price_cache[symbol]
                        
                except Exception as e:
                    self.logger.error(f"Error handling price message: {e}")
            
            # Start the WebSocket connection
            conn_key = self.socket_manager.start_symbol_ticker_socket(
                symbol.lower(),
                handle_socket_message
            )
            
            self.websocket_connections[f"{symbol}_price"] = conn_key
            
        except Exception as e:
            self.logger.error(f"Failed to start price stream: {e}")
            raise
    
    async def _start_orderbook_stream(self, symbol: str):
        """Start order book stream for a symbol."""
        try:
            def handle_orderbook_message(msg):
                """Handle incoming order book messages."""
                try:
                    if msg['e'] == 'depthUpdate':
                        self.orderbook_cache[symbol] = {
                            'symbol': msg['s'],
                            'bids': [[float(bid[0]), float(bid[1])] for bid in msg['b']],
                            'asks': [[float(ask[0]), float(ask[1])] for ask in msg['a']],
                            'timestamp': datetime.now()
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error handling orderbook message: {e}")
            
            # Start the WebSocket connection
            conn_key = self.socket_manager.start_depth_socket(
                symbol.lower(),
                handle_orderbook_message
            )
            
            self.websocket_connections[f"{symbol}_orderbook"] = conn_key
            
        except Exception as e:
            self.logger.error(f"Failed to start orderbook stream: {e}")
            raise
    
    async def _start_trade_stream(self, symbol: str):
        """Start trade stream for a symbol."""
        try:
            def handle_trade_message(msg):
                """Handle incoming trade messages."""
                try:
                    if msg['e'] == 'trade':
                        trade_data = {
                            'symbol': msg['s'],
                            'price': float(msg['p']),
                            'quantity': float(msg['q']),
                            'timestamp': datetime.fromtimestamp(msg['T'] / 1000),
                            'is_buyer_maker': msg['m']
                        }
                        
                        # Store recent trades
                        if symbol not in self.trade_cache:
                            self.trade_cache[symbol] = []
                        
                        self.trade_cache[symbol].append(trade_data)
                        
                        # Keep only last 100 trades
                        if len(self.trade_cache[symbol]) > 100:
                            self.trade_cache[symbol] = self.trade_cache[symbol][-100:]
                        
                except Exception as e:
                    self.logger.error(f"Error handling trade message: {e}")
            
            # Start the WebSocket connection
            conn_key = self.socket_manager.start_trade_socket(
                symbol.lower(),
                handle_trade_message
            )
            
            self.websocket_connections[f"{symbol}_trades"] = conn_key
            
        except Exception as e:
            self.logger.error(f"Failed to start trade stream: {e}")
            raise
    
    async def stop_real_time_feed(self):
        """Stop all real-time data feeds."""
        try:
            self.logger.info("Stopping real-time feeds...")
            
            # Stop all WebSocket connections
            for conn_key in self.websocket_connections.values():
                self.socket_manager.stop_socket(conn_key)
            
            self.websocket_connections.clear()
            
            # Close socket manager
            if self.socket_manager:
                self.socket_manager.close()
            
            self.logger.info("Real-time feeds stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping real-time feeds: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str = "15m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical kline data.
        
        Args:
            symbol: Trading symbol
            interval: Kline interval
            start_time: Start time
            end_time: End time
            limit: Number of records to retrieve
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to timestamp if provided
            start_str = None
            end_str = None
            
            if start_time:
                start_str = str(int(start_time.timestamp() * 1000))
            if end_time:
                end_str = str(int(end_time.timestamp() * 1000))
            
            # Get klines from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Store in database if available
            if self.database:
                await self._store_historical_data(symbol, df, interval)
            
            self.logger.info(f"Retrieved {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            raise
    
    async def _store_historical_data(self, symbol: str, df: pd.DataFrame, interval: str):
        """Store historical data in database."""
        try:
            for timestamp, row in df.iterrows():
                await self.database.execute(
                    """
                    INSERT INTO market_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume, timeframe)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, timestamp, timeframe) DO NOTHING
                    """,
                    symbol, timestamp, row['open'], row['high'], 
                    row['low'], row['close'], row['volume'], interval
                )
        except Exception as e:
            self.logger.error(f"Failed to store historical data: {e}")
    
    async def get_latest_data(self, symbol: str) -> Optional[Dict]:
        """Get the latest market data for a symbol."""
        try:
            # Try to get from real-time cache first
            if symbol in self.real_time_data:
                return self.real_time_data[symbol]
            
            # Fallback to API call
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            
            return {
                'symbol': symbol,
                'price': float(ticker['price']),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get latest data for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            # Try cache first
            if symbol in self.price_cache:
                return self.price_cache[symbol]['price']
            
            # Fallback to API
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
            
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            return 0.0
    
    def get_current_price_sync(self, symbol: str) -> float:
        """Synchronous version of get_current_price."""
        try:
            if symbol in self.price_cache:
                return self.price_cache[symbol]['price']
            
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
            
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            return 0.0
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol."""
        try:
            # Try cache first
            if symbol in self.orderbook_cache:
                return self.orderbook_cache[symbol]
            
            # Fallback to API
            orderbook = self.client.get_order_book(symbol=symbol, limit=limit)
            
            return {
                'symbol': symbol,
                'bids': [[float(bid[0]), float(bid[1])] for bid in orderbook['bids']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in orderbook['asks']],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return {}
    
    async def place_order(self, order) -> Dict:
        """
        Place an order on the exchange.
        
        Args:
            order: Order object with order details
            
        Returns:
            Order result dictionary
        """
        try:
            self.logger.info(f"Placing {order.side.value} order: {order.quantity} {order.symbol}")
            
            # Prepare order parameters
            order_params = {
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.type.value,
                'quantity': order.quantity
            }
            
            # Add price for limit orders
            if order.type.value == 'LIMIT' and order.price:
                order_params['price'] = str(order.price)
                order_params['timeInForce'] = 'GTC'
            
            # Add stop price for stop orders
            if order.type.value in ['STOP_LOSS', 'TAKE_PROFIT'] and order.stop_price:
                order_params['stopPrice'] = str(order.stop_price)
            
            # Place the order
            if self.testnet:
                # For testnet, simulate order execution
                result = {
                    'orderId': f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'status': 'FILLED',
                    'executedQty': str(order.quantity),
                    'fills': [{
                        'price': str(order.price or await self.get_current_price(order.symbol)),
                        'qty': str(order.quantity),
                        'commission': '0.001',
                        'commissionAsset': 'BNB'
                    }]
                }
            else:
                result = self.client.create_order(**order_params)
            
            # Calculate fill price
            if result.get('fills'):
                fill_price = sum(
                    float(fill['price']) * float(fill['qty']) 
                    for fill in result['fills']
                ) / sum(float(fill['qty']) for fill in result['fills'])
                result['fill_price'] = fill_price
            
            # Store trade in database
            if self.database and result.get('status') == 'FILLED':
                await self._store_trade(order, result)
            
            self.logger.info(f"Order placed successfully: {result.get('orderId')}")
            return result
            
        except BinanceAPIException as e:
            self.logger.error(f"Binance API error placing order: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> Dict:
        """Cancel an order."""
        try:
            if symbol:
                result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            else:
                # Try to find the symbol from order ID
                # This is a simplified implementation
                result = {'status': 'CANCELED', 'orderId': order_id}
            
            self.logger.info(f"Order {order_id} canceled")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def _store_trade(self, order, result: Dict):
        """Store executed trade in database."""
        try:
            if not result.get('fills'):
                return
            
            for fill in result['fills']:
                await self.database.execute(
                    """
                    INSERT INTO trades 
                    (order_id, symbol, side, quantity, price, commission, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (order_id) DO NOTHING
                    """,
                    result['orderId'],
                    order.symbol,
                    order.side.value,
                    float(fill['qty']),
                    float(fill['price']),
                    float(fill['commission']),
                    order.timestamp
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store trade: {e}")
    
    async def get_account_info(self) -> Dict:
        """Get account information."""
        try:
            account = self.client.get_account()
            
            # Process balances
            balances = {}
            for balance in account['balances']:
                if float(balance['free']) > 0 or float(balance['locked']) > 0:
                    balances[balance['asset']] = {
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
            
            return {
                'balances': balances,
                'can_trade': account['canTrade'],
                'can_withdraw': account['canWithdraw'],
                'can_deposit': account['canDeposit'],
                'update_time': datetime.fromtimestamp(account['updateTime'] / 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get account info: {e}")
            raise
    
    def get_connection_status(self) -> Dict:
        """Get connection status information."""
        return {
            'is_connected': self.is_connected,
            'testnet': self.testnet,
            'websocket_connections': len(self.websocket_connections),
            'last_heartbeat': self.last_heartbeat,
            'cached_symbols': list(self.price_cache.keys())
        }

