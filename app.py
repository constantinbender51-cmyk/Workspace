import requests
import pandas as pd
from datetime import datetime, timezone
import time
from flask import Flask, render_template
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# --- Configuration ---
# API Base URL for Blockchain.com Charts & Statistics
BASE_URL = "https://api.blockchain.info/charts/"

# Dictionary mapping desired metric names to their respective Blockchain.com chart endpoints
METRICS = {
    'Active_Addresses': 'unique-addresses-used',
    'Net_Transaction_Count': 'n-transactions',
    'Transaction_Volume_USD': 'estimated-transaction-volume-usd',
}

# The requested date range
START_DATE = '2022-01-01'
END_DATE = '2023-09-30' # We will use this date to filter the final DataFrame




def fetch_binance_price_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches daily Bitcoin price data from Binance API for the specified date range.
    Returns a DataFrame with 'Date' as index and 'Price' column (closing price).
    """
    # Convert dates to milliseconds for Binance API
    start_timestamp = int(pd.to_datetime(start_date).timestamp() * 1000)
    end_timestamp = int(pd.to_datetime(end_date).timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'startTime': start_timestamp,
        'endTime': end_timestamp,
        'limit': 1000
    }
    
    print("-> Fetching Bitcoin price data from Binance...")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        if not data:
            print("   Warning: No price data found from Binance.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        # Binance returns: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        
        # Convert timestamp to datetime and set as index
        df['Date'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('Date')
        
        # Use closing price and rename column
        df_price = df['close'].astype(float).rename('Price_USD')
        
        print(f"   Success! Fetched {len(df_price)} price data points.")
        return df_price
        
    except requests.exceptions.RequestException as e:
        print(f"   Error fetching price data from Binance: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"   An unexpected error occurred with Binance API: {e}")
        return pd.DataFrame()

def fetch_chart_data(chart_name: str, start_date: str) -> pd.DataFrame:
    """
    Fetches historical data for a single chart from the Blockchain.com API.

    The 'sampled=false' parameter is crucial to ensure daily granularity
    over long time spans, as the API defaults to sampling the data.
    The 'timespan=all' is used in conjunction with the 'start' date to retrieve 
    data up to the latest available date, which we will later filter.
    """
    params = {
        'format': 'json',
        'start': start_date,
        'timespan': 'all',  # Request all data from the start date onwards
        'sampled': 'false'  # Crucial for daily data retrieval
    }
    
    url = f"{BASE_URL}{chart_name}"
    
    print(f"-> Fetching data for {chart_name}...")
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        
        data = response.json()
        
        if 'values' not in data or not data['values']:
            print(f"   Warning: No data found for {chart_name}.")
            return pd.DataFrame()

        # Convert the list of dicts into a pandas DataFrame
        df = pd.DataFrame(data['values'])
        
        # 'x' is the Unix timestamp (seconds), 'y' is the metric value
        # Convert Unix timestamp to datetime objects and set as index
        df['Date'] = pd.to_datetime(df['x'], unit='s', utc=True).dt.tz_localize(None)
        df = df.set_index('Date')['y'].rename(chart_name)

        print(f"   Success! Fetched {len(df)} data points.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"   Error fetching data for {chart_name}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"   An unexpected error occurred for {chart_name}: {e}")
        return pd.DataFrame()


def get_bitcoin_data():
    """
    Fetch and process Bitcoin on-chain data and price data.
    """
    print("--- Starting Bitcoin Data Fetcher ---")
    
    all_data = []
    
    # Fetch on-chain metrics
    for metric_name, chart_endpoint in METRICS.items():
        # Use a short sleep to be polite to the public API
        time.sleep(1.5) 
        
        # Fetch the data for the current metric
        df_metric = fetch_chart_data(chart_endpoint, START_DATE)
        
        if not df_metric.empty:
            # Rename the column to the user-friendly metric name
            df_metric = df_metric.rename(metric_name)
            all_data.append(df_metric)

    # Fetch price data from Binance
    df_price = fetch_binance_price_data(START_DATE, END_DATE)
    if not df_price.empty:
        all_data.append(df_price)

    if not all_data:
        print("\n--- Failed to retrieve any data. Aborting. ---")
        return pd.DataFrame()

    # Combine all fetched Series into a single DataFrame
    df_combined = pd.concat(all_data, axis=1)

    # --- Filtering and Cleanup ---
    
    # 1. Filter the DataFrame to the exact requested end date
    df_final = df_combined.loc[START_DATE:END_DATE]
    
    # 2. Fill potential missing values (if any metric missed a day)
    # Forward-fill (ffill) is usually better than filling with 0 for time-series data
    df_final = df_final.ffill()
    
    # 3. Handle potential multiple measurements on the same day (shouldn't happen with this API, but good practice)
    df_final = df_final[~df_final.index.duplicated(keep='first')]

    print("\n--- Fetching Complete ---")
    print(f"Requested Period: {START_DATE} to {END_DATE}")
    print(f"Final DataFrame Shape: {df_final.shape}")
    
    return df_final


@app.route('/')
def index():
    """
    Main route that fetches data and displays interactive charts.
    """
    df = get_bitcoin_data()
    
    if df.empty:
        return "<h1>Bitcoin On-Chain Data</h1><p>Failed to fetch data. Please try again later.</p>"
    
    # Reset index to have Date as a column for plotting
    df_plot = df.reset_index()
    
    # Create individual charts for each metric
    charts_html = ""
    
    for column in df_plot.columns[1:]:  # Skip the Date column
        fig = px.line(df_plot, x='Date', y=column, title=f'Bitcoin {column.replace("_", " ")}')
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=column.replace('_', ' '),
            height=400
        )
        charts_html += pio.to_html(fig, full_html=False)
    
    # Create HTML page
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bitcoin On-Chain Data Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .chart {{ margin-bottom: 40px; }}
            h1 {{ color: #333; }}
            .info {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Bitcoin On-Chain Data Dashboard</h1>
        <div class="info">
            <p><strong>Period:</strong> {START_DATE} to {END_DATE}</p>
            <p><strong>Data Points:</strong> {len(df)}</p>
            <p><strong>Metrics:</strong> {', '.join(df.columns)}</p>
        </div>
        {charts_html}
    </body>
    </html>
    """
    
    return html_content


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)