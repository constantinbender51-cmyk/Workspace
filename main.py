import fetch_candles

if __name__ == "__main__":
    print("Fetching 3000 candles from Binance...")
    
    symbol = "BTCUSDT"
    interval = "1m"
    limit = 3000
    
    candles = fetch_candles.fetch_binance_candles(symbol=symbol, interval=interval, limit=limit)
    
    if candles is not None and len(candles) > 0:
        fetch_candles.display_candles(candles)
        print(f"\n✅ Successfully fetched {len(candles)} {symbol} {interval} candles!")
    else:
        print("❌ Failed to fetch candle data. Please check your internet connection.")