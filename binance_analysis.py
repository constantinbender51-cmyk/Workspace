import requests
import pandas as pd
import json
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Get your free API key at: https://fredaccount.stlouisfed.org/apikeys
# 2. Paste it inside the quotes below.
API_KEY = "8005f92c424c0503df32084af3e66daf" 

# Series IDs
# FEDFUNDS: Effective Federal Funds Rate (Monthly)
# WALCL: Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level
SERIES_IDS = {
    "Interest Rate (Fed Funds)": "FEDFUNDS",
    "Fed Balance Sheet (Total Assets)": "WALCL"
}

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_series_data(series_id, api_key):
    """
    Fetches observations for a specific FRED series ID.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc" # Oldest to newest
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # FRED returns data in a 'observations' list
        # Fields: realtime_start, realtime_end, date, value
        observations = data.get("observations", [])
        
        # Convert to DataFrame
        df = pd.DataFrame(observations)
        
        # Clean data
        # 'value' comes as string and can contain '.' for missing data
        df = df[df['value'] != '.'] 
        df['value'] = pd.to_numeric(df['value'])
        df['date'] = pd.to_datetime(df['date'])
        
        return df[['date', 'value']]

    except requests.exceptions.HTTPError as err:
        print(f"Error fetching {series_id}: {err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def calculate_metrics(name, df):
    """
    Calculates key metrics from the series DataFrame.
    """
    if df is None or df.empty:
        return f"No data available for {name}"

    latest_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
    start_date = df['date'].iloc[0].strftime('%Y-%m-%d')
    
    metrics = {
        "Series Name": name,
        "Start Date": start_date,
        "End Date": latest_date,
        "Total Observations": len(df),
        "Latest Value": f"{df['value'].iloc[-1]:,.2f}",
        "Min Value": f"{df['value'].min():,.2f}",
        "Max Value": f"{df['value'].max():,.2f}",
        "Average Value": f"{df['value'].mean():,.2f}"
    }
    return metrics

def main():
    if not API_KEY:
        print("❌ ERROR: API Key is missing.")
        print("Please open the script and paste your FRED API key into the 'API_KEY' variable.")
        return

    print(f"Fetching data from FRED API...\n")
    
    all_metrics = []

    for name, series_id in SERIES_IDS.items():
        print(f"Processing: {name} ({series_id})...")
        df = fetch_series_data(series_id, API_KEY)
        
        if df is not None:
            metric = calculate_metrics(name, df)
            all_metrics.append(metric)

    # Display Results
    print("\n" + "="*80)
    print(f"{'FRED DATA METRICS REPORT':^80}")
    print("="*80 + "\n")

    # Create a summary DataFrame for pretty printing
    if all_metrics:
        results_df = pd.DataFrame(all_metrics)
        
        # Transpose for a card-like view per series or just print the table
        # Here we iterate to print a readable list
        for item in all_metrics:
            print(f"🔹 {item['Series Name']}")
            print("-" * 40)
            for key, value in item.items():
                if key != "Series Name":
                    print(f"{key:<20}: {value}")
            print("\n")
    else:
        print("No metrics could be generated.")

if __name__ == "__main__":
    main()