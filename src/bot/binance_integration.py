"""
Binance API Integration for ETH/FDUSD Trading Bot
Production-ready integration with comprehensive error handling
"""

import asyncio
import logging
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import websocket
import threading
from urllib.parse import urlencode
import os
from decimal import Decimal, ROUND_DOWN


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: str
    timestamp: datetime
    commission: float = 0.0
    commission_asset: str = ""


@dataclass
class AccountBalance:
    """Account balance information"""
    asset: str
    free: float
    locked: float
    total: float


@dataclass
class MarketTicker:
    """Market ticker information"""
    symbol: str
    price: float
    price_change: float
    price_change_percent: float
    volume: float
    high: float
    low: float
    timestamp: datetime


class BinanceAPIError(Exception):
    """Custom exception for Binance API errors"""
    def __init__(self, message: str, error_code: int = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class BinanceIntegration:
    """
    Professional Binance API integration with advanced features
    Supports both REST API and WebSocket for real-time data
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        # API credentials
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API credentials not provided")
        
        # API endpoints
        if testnet:
            self.base_url = "https://testnet.binance.vision"
            self.ws_base_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com"
            self.ws_base_url = "wss://stream.binance.com:9443/ws"
        
        # Configuration
        self.symbol = "ETHFDUSD"
        self.testnet = testnet
        self.recv_window = 5000
        
        # State management
        self.logger = self._setup_logging()
        self.session = requests.Session()
        self.session.headers.update({'X-MBX-APIKEY': self.api_key})
        
        # WebSocket management
        self.ws = None
        self.ws_thread = None
        self.ws_callbacks = {}
        self.is_connected = False
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # seconds
        self.max_requests_per_minute = 1200
        
        # Market data cache
        self.market_data_cache = {}
        self.last_price_update = None
        
        self.logger.info(f"Binance Integration initialized (Testnet: {testnet})")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Binance integration"""
        logger = logging.getLogger('BinanceIntegration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature for API requests"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds"""
        return int(time.time() * 1000)
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Check if rate limit exceeded
        if self.request_count >= self.max_requests_per_minute:
            sleep_time = self.rate_limit_window - (current_time - self.last_request_time)
            if sleep_time > 0:
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """Make HTTP request to Binance API with error handling"""
        self._check_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if signed:
            params['timestamp'] = self._get_timestamp()
            params['recvWindow'] = self.recv_window
            query_string = urlencode(params)
            params['signature'] = self._generate_signature(query_string)
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, params=params, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise BinanceAPIError(f"Request failed: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response: {e}")
            raise BinanceAPIError(f"Invalid JSON response: {e}")
    
    # Market Data Methods
    
    def get_symbol_info(self) -> Dict:
        """Get symbol information and trading rules"""
        try:
            response = self._make_request('GET', '/api/v3/exchangeInfo')
            
            for symbol_info in response['symbols']:
                if symbol_info['symbol'] == self.symbol:
                    return symbol_info
            
            raise BinanceAPIError(f"Symbol {self.symbol} not found")
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            raise
    
    def get_ticker(self) -> MarketTicker:
        """Get 24hr ticker price change statistics"""
        try:
            response = self._make_request('GET', '/api/v3/ticker/24hr', {'symbol': self.symbol})
            
            return MarketTicker(
                symbol=response['symbol'],
                price=float(response['lastPrice']),
                price_change=float(response['priceChange']),
                price_change_percent=float(response['priceChangePercent']),
                volume=float(response['volume']),
                high=float(response['highPrice']),
                low=float(response['lowPrice']),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting ticker: {e}")
            raise
    
    def get_orderbook(self, limit: int = 100) -> Dict:
        """Get order book depth"""
        try:
            params = {'symbol': self.symbol, 'limit': limit}
            response = self._make_request('GET', '/api/v3/depth', params)
            
            return {
                'bids': [[float(price), float(qty)] for price, qty in response['bids']],
                'asks': [[float(price), float(qty)] for price, qty in response['asks']],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting orderbook: {e}")
            raise
    
    def get_klines(self, interval: str = '15m', limit: int = 500) -> List[Dict]:
        """Get kline/candlestick data"""
        try:
            params = {
                'symbol': self.symbol,
                'interval': interval,
                'limit': limit
            }
            response = self._make_request('GET', '/api/v3/klines', params)
            
            klines = []
            for kline in response:
                klines.append({
                    'timestamp': datetime.fromtimestamp(kline[0] / 1000),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': datetime.fromtimestamp(kline[6] / 1000),
                    'quote_volume': float(kline[7]),
                    'trades': int(kline[8])
                })
            
            return klines
            
        except Exception as e:
            self.logger.error(f"Error getting klines: {e}")
            raise
    
    # Account Methods
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        try:
            response = self._make_request('GET', '/api/v3/account', signed=True)
            
            balances = []
            for balance in response['balances']:
                if float(balance['free']) > 0 or float(balance['locked']) > 0:
                    balances.append(AccountBalance(
                        asset=balance['asset'],
                        free=float(balance['free']),
                        locked=float(balance['locked']),
                        total=float(balance['free']) + float(balance['locked'])
                    ))
            
            return {
                'balances': balances,
                'can_trade': response['canTrade'],
                'can_withdraw': response['canWithdraw'],
                'can_deposit': response['canDeposit'],
                'update_time': datetime.fromtimestamp(response['updateTime'] / 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise
    
    def get_balance(self, asset: str) -> Optional[AccountBalance]:
        """Get balance for specific asset"""
        try:
            account_info = self.get_account_info()
            
            for balance in account_info['balances']:
                if balance.asset == asset:
                    return balance
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting balance for {asset}: {e}")
            raise
    
    # Trading Methods
    
    def place_market_order(self, side: str, quantity: float) -> OrderResult:
        """Place market order"""
        try:
            # Get symbol info for precision
            symbol_info = self.get_symbol_info()
            
            # Find quantity precision
            qty_precision = 0
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    step_size = float(filter_info['stepSize'])
                    qty_precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    break
            
            # Round quantity to proper precision
            quantity = float(Decimal(str(quantity)).quantize(
                Decimal('0.' + '0' * qty_precision), rounding=ROUND_DOWN
            ))
            
            params = {
                'symbol': self.symbol,
                'side': side.upper(),
                'type': 'MARKET',
                'quantity': quantity
            }
            
            response = self._make_request('POST', '/api/v3/order', params, signed=True)
            
            return OrderResult(
                order_id=str(response['orderId']),
                symbol=response['symbol'],
                side=response['side'],
                quantity=float(response['executedQty']),
                price=float(response.get('price', 0)),
                status=response['status'],
                timestamp=datetime.fromtimestamp(response['transactTime'] / 1000)
            )
            
        except Exception as e:
            self.logger.error(f"Error placing market order: {e}")
            raise
    
    def place_limit_order(self, side: str, quantity: float, price: float) -> OrderResult:
        """Place limit order"""
        try:
            # Get symbol info for precision
            symbol_info = self.get_symbol_info()
            
            # Find precision requirements
            qty_precision = 0
            price_precision = 0
            
            for filter_info in symbol_info['filters']:
                if filter_info['filterType'] == 'LOT_SIZE':
                    step_size = float(filter_info['stepSize'])
                    qty_precision = len(str(step_size).split('.')[-1].rstrip('0'))
                elif filter_info['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter_info['tickSize'])
                    price_precision = len(str(tick_size).split('.')[-1].rstrip('0'))
            
            # Round to proper precision
            quantity = float(Decimal(str(quantity)).quantize(
                Decimal('0.' + '0' * qty_precision), rounding=ROUND_DOWN
            ))
            price = float(Decimal(str(price)).quantize(
                Decimal('0.' + '0' * price_precision), rounding=ROUND_DOWN
            ))
            
            params = {
                'symbol': self.symbol,
                'side': side.upper(),
                'type': 'LIMIT',
                'timeInForce': 'GTC',
                'quantity': quantity,
                'price': price
            }
            
            response = self._make_request('POST', '/api/v3/order', params, signed=True)
            
            return OrderResult(
                order_id=str(response['orderId']),
                symbol=response['symbol'],
                side=response['side'],
                quantity=float(response['origQty']),
                price=float(response['price']),
                status=response['status'],
                timestamp=datetime.fromtimestamp(response['transactTime'] / 1000)
            )
            
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order"""
        try:
            params = {
                'symbol': self.symbol,
                'orderId': order_id
            }
            
            response = self._make_request('DELETE', '/api/v3/order', params, signed=True)
            return response
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            raise
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status"""
        try:
            params = {
                'symbol': self.symbol,
                'orderId': order_id
            }
            
            response = self._make_request('GET', '/api/v3/order', params, signed=True)
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            raise
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        try:
            params = {'symbol': self.symbol}
            response = self._make_request('GET', '/api/v3/openOrders', params, signed=True)
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            raise
    
    # WebSocket Methods
    
    def start_websocket(self, streams: List[str], callback: callable):
        """Start WebSocket connection for real-time data"""
        try:
            if self.is_connected:
                self.logger.warning("WebSocket already connected")
                return
            
            # Create WebSocket URL
            stream_names = '/'.join(streams)
            ws_url = f"{self.ws_base_url}/{stream_names}"
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error processing WebSocket message: {e}")
            
            def on_error(ws, error):
                self.logger.error(f"WebSocket error: {error}")
                self.is_connected = False
            
            def on_close(ws, close_status_code, close_msg):
                self.logger.info("WebSocket connection closed")
                self.is_connected = False
            
            def on_open(ws):
                self.logger.info("WebSocket connection opened")
                self.is_connected = True
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket in separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            self.logger.info(f"WebSocket started for streams: {streams}")
            
        except Exception as e:
            self.logger.error(f"Error starting WebSocket: {e}")
            raise
    
    def stop_websocket(self):
        """Stop WebSocket connection"""
        try:
            if self.ws:
                self.ws.close()
                self.is_connected = False
                self.logger.info("WebSocket stopped")
                
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket: {e}")
    
    # Utility Methods
    
    def test_connectivity(self) -> bool:
        """Test API connectivity"""
        try:
            self._make_request('GET', '/api/v3/ping')
            self.logger.info("API connectivity test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"API connectivity test failed: {e}")
            return False
    
    def get_server_time(self) -> datetime:
        """Get server time"""
        try:
            response = self._make_request('GET', '/api/v3/time')
            return datetime.fromtimestamp(response['serverTime'] / 1000)
            
        except Exception as e:
            self.logger.error(f"Error getting server time: {e}")
            raise
    
    def calculate_position_size(self, account_balance: float, risk_percentage: float, 
                              entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        try:
            risk_amount = account_balance * (risk_percentage / 100)
            price_difference = abs(entry_price - stop_loss)
            position_size = risk_amount / price_difference
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0


if __name__ == "__main__":
    # Example usage
    print("Binance Integration Module")
    print("Professional-grade API integration for ETH/FDUSD trading")
    
    # Test with environment variables
    try:
        binance = BinanceIntegration(testnet=True)
        if binance.test_connectivity():
            print("✓ Binance API connection successful")
        else:
            print("✗ Binance API connection failed")
    except Exception as e:
        print(f"✗ Error initializing Binance integration: {e}")
        print("Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")

