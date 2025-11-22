from binance.client import Client
import pandas as pd

def fetch_top_assets_by_volume(limit=10):
    """
    Fetch the top crypto assets by 24h trading volume from Binance.
    
    Parameters:
    limit (int): Number of top assets to return, default is 10.
    
    Returns:
    list: List of symbol strings for the top assets.
    """
    try:
        client = Client()
        if not client.ping():
            print("Error: Unable to connect to Binance API")
            return []
        # Get 24h ticker data for all symbols
        tickers = client.get_ticker()
        # Filter for USDT pairs and sort by quoteVolume (trading volume in quote asset)
        usdt_tickers = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
        sorted_tickers = sorted(usdt_tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
        top_symbols = [ticker['symbol'] for ticker in sorted_tickers[:limit]]
        return top_symbols
    except Exception as e:
        print(f"An error occurred while fetching top assets: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    top_assets = fetch_top_assets_by_volume(10)
    print("Top 10 assets by volume:", top_assets)