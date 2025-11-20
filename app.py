import os
from flask import Flask, jsonify
import fetch_candles

app = Flask(__name__)

@app.route('/')
def get_candles():
    """Fetch and return 3000 Binance candles"""
    symbol = "BTCUSDT"
    interval = "1m"
    limit = 3000
    
    candles = fetch_candles.fetch_binance_candles(symbol=symbol, interval=interval, limit=limit)
    
    if candles is not None and len(candles) > 0:
        # Convert DataFrame to JSON-serializable format
        result = {
            'status': 'success',
            'candles_fetched': len(candles),
            'symbol': symbol,
            'interval': interval,
            'latest_price': float(candles['close'].iloc[0]),
            'time_range': {
                'start': candles['open_time'].iloc[-1].isoformat(),
                'end': candles['open_time'].iloc[0].isoformat()
            }
        }
        return jsonify(result)
    else:
        return jsonify({'status': 'error', 'message': 'Failed to fetch candle data'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
from flask import Flask, jsonify
import fetch_candles

app = Flask(__name__)

@app.route('/')
def get_candles():
    """Fetch and return 3000 Binance candles"""
    symbol = "BTCUSDT"
    interval = "1m"
    limit = 3000
    
    candles = fetch_candles.fetch_binance_candles(symbol=symbol, interval=interval, limit=limit)
    
    if candles is not None and len(candles) > 0:
        # Convert DataFrame to JSON-serializable format
        result = {
            'status': 'success',
            'candles_fetched': len(candles),
            'symbol': symbol,
            'interval': interval,
            'latest_price': float(candles['close'].iloc[0]),
            'time_range': {
                'start': candles['open_time'].iloc[-1].isoformat(),
                'end': candles['open_time'].iloc[0].isoformat()
            }
        }
        return jsonify(result)
    else:
        return jsonify({'status': 'error', 'message': 'Failed to fetch candle data'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)