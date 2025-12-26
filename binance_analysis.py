import requests
import pandas as pd
from datetime import timedelta

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = "8005f92c424c0503df32084af3e66daf" 
BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def fetch_data(series_id, api_key):
    """Fetches data from FRED and returns a clean DataFrame."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc"
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json().get("observations", [])
        
        df = pd.DataFrame(data)
        if df.empty: return pd.DataFrame()

        df = df[df['value'] != '.'] 
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'])
        return df[['date', 'value']].sort_values('date')
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.DataFrame()

def main():
    if not API_KEY:
        print("Please insert your API Key in the script.")
        return

    print("Fetching data...")
    # 1. Fetch Data
    df_rate = fetch_data("FEDFUNDS", API_KEY)
    df_rate.rename(columns={'value': 'interest_rate'}, inplace=True)
    
    df_bs = fetch_data("WALCL", API_KEY) 
    df_bs.rename(columns={'value': 'balance_sheet'}, inplace=True)

    if df_rate.empty or df_bs.empty:
        print("Failed to fetch data.")
        return

    # 2. Merge Monthly Rate into Weekly Balance Sheet
    print("Processing data...")
    df_merged = pd.merge_asof(
        df_bs, 
        df_rate, 
        on='date', 
        direction='backward'
    )

    # 3. Calculate Weekly Balance Sheet Change
    df_merged['bs_change'] = df_merged['balance_sheet'].diff()
    
    # 4. Filter for Last 10 Years ONLY
    # We find the latest date in the dataset and subtract 365*10 days
    latest_date = df_merged['date'].max()
    cutoff_date = latest_date - timedelta(days=365 * 10)
    
    df_10y = df_merged[df_merged['date'] >= cutoff_date].dropna().copy()

    # 5. Calculate Average Rate (Based on this 10-year window)
    avg_rate_10y = df_10y['interest_rate'].mean()

    # 6. Categorize
    def get_category(row):
        high_rate = row['interest_rate'] > avg_rate_10y
        growing_bs = row['bs_change'] >= 0
        
        if high_rate and growing_bs:
            return "1. High Rate / Growing BS"
        elif high_rate and not growing_bs:
            return "2. High Rate / Shrinking BS"
        elif not high_rate and growing_bs:
            return "3. Low Rate / Growing BS"
        elif not high_rate and not growing_bs:
            return "4. Low Rate / Shrinking BS"

    df_10y['category'] = df_10y.apply(get_category, axis=1)

    # ==========================================
    # OUTPUT REPORT
    # ==========================================
    print("\n" + "="*60)
    print(f"FRED ANALYSIS: LAST 10 YEARS")
    print("="*60)
    print(f"Date Range: {df_10y['date'].iloc[0].date()} to {df_10y['date'].iloc[-1].date()}")
    print(f"Total Weeks: {len(df_10y)}")
    print(f"10-Year Avg Interest Rate Threshold: {avg_rate_10y:.2f}%")
    print("-" * 60)
    
    summary = df_10y['category'].value_counts().reset_index()
    summary.columns = ['Category', 'Weeks']
    summary['Percentage'] = (summary['Weeks'] / len(df_10y) * 100).round(1)
    
    # Sort by Category Name for consistent display
    summary = summary.sort_values('Category')
    
    print(summary.to_string(index=False))
    print("-" * 60)

if __name__ == "__main__":
    main()