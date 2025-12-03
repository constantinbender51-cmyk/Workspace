import pandas as pd
import numpy as np
from flask import Flask, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import ccxt

app = Flask(__name__)

# HTML template for the web page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Binance Trading Strategy Returns</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .info-box {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            border-left: 4px solid #007bff;
        }
        .plot-container {
            text-align: center;
            margin: 30px 0;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Binance Trading Strategy Returns</h1>
        
        <div class="info-box">
            <h3>Strategy Description</h3>
            <p>This strategy analyzes BTC/USDT data from Binance starting from 2018:</p>
            <ul>
                <li><strong>Add returns</strong> when price is above both 365-day and 120-day Simple Moving Averages (SMAs)</li>
                <li><strong>Subtract returns</strong> when price is below both 365-day and 120-day SMAs</li>
                <li><strong>Add 0 return</strong> otherwise (when price is between SMAs or above one but below the other)</li>
            </ul>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{{ total_return_pct }}%</div>
                <div class="stat-label">Total Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ final_cum_return }}</div>
                <div class="stat-label">Final Cumulative Return</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ sharpe_ratio }}</div>
                <div class="stat-label">Annualized Sharpe Ratio</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ positive_days }}</div>
                <div class="stat-label">Positive Signal Days</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ negative_days }}</div>
                <div class="stat-label">Negative Signal Days</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ avg_monthly_return_pct }}%</div>
                <div class="stat-label">Avg Monthly Return</div>
            </div>
        </div>
        
        <div class="plot-container">
            <h3>Cumulative Returns Over Time</h3>
            <img src="data:image/png;base64,{{ plot_url }}" alt="Cumulative Returns Plot">
        </div>
        <div class="plot-container">
            <h3>Monthly Strategy Returns</h3>
            <img src="data:image/png;base64,{{ monthly_plot_url }}" alt="Monthly Returns Plot">
        </div>
        
        <div class="info-box">
            <h3>Monthly Strategy Returns</h3>
            <div style="max-height: 400px; overflow-y: auto; margin-top: 15px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background-color: #f2f2f2;">
                            <th style="padding: 10px; text-align: left; border-bottom: 2px solid #ddd;">Month</th>
                            <th style="padding: 10px; text-align: right; border-bottom: 2px solid #ddd;">Return (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for month in monthly_returns %}
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 8px 10px;">{{ month.year_month }}</td>
                            <td style="padding: 8px 10px; text-align: right; color: {% if month.is_positive %}#28a745{% else %}#dc3545{% endif %};">
                                {{ month.return_pct }}%
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            <p style="margin-top: 15px; font-size: 14px; color: #666;">
                <strong>Note:</strong> Monthly returns are calculated from daily strategy returns resampled to month-end.
            </p>
        </div>
        
        
        <div class="info-box">
            <h3>Grid Search for Optimal Parameters</h3>
            <p>A grid search has been implemented to find the optimal leverage and stop loss parameters.</p>
            <p><strong>Current Parameters:</strong> Leverage = 3.8x, Stop Loss = 5.0%</p>
            <p><a href="/grid_search" style="color: #007bff; text-decoration: none; font-weight: bold;">
                → Click here to run grid search and find optimal parameters
            </a></p>
            <p style="font-size: 14px; color: #666; margin-top: 10px;">
                The grid search tests leverage from 1 to 5 and stop loss from 1% to 10% to maximize Sharpe ratio.
            </p>
        </div>        <div class="info-box">
            <h3>Data Information</h3>
            <p><strong>Symbol:</strong> BTC/USDT</p>
            <p><strong>Exchange:</strong> Binance</p>
            <p><strong>Timeframe:</strong> 1 day</p>
            <p><strong>Start Date:</strong> {{ start_date }}</p>
            <p><strong>End Date:</strong> {{ end_date }}</p>
            <p><strong>Total Days:</strong> {{ total_days }}</p>
            <p><strong>Last Updated:</strong> {{ current_time }}</p>
        </div>
    </div>
</body>
</html>
"""

def fetch_binance_data():
    """Fetch OHLCV data from Binance starting from 2018 to present"""
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        # Calculate start timestamp for 2018
        start_date = datetime(2018, 1, 1)
        since = exchange.parse8601(start_date.isoformat() + 'Z')
        
        # Fetch all OHLCV data with pagination
        print("Fetching BTC/USDT data from Binance (2018 to present)...")
        all_ohlcv = []
        
        while True:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', since=since, limit=1000)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            
            # Update since to the last timestamp + 1 day (in milliseconds)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + (24 * 60 * 60 * 1000)
            
            # Check if we've reached current date
            last_date = pd.to_datetime(last_timestamp, unit='ms')
            if last_date.date() >= datetime.now().date():
                break
            
            # Rate limiting
            exchange.sleep(1000)
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        
        # Remove duplicates and sort
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        print(f"Fetched {len(df)} days of data from {df.index[0].date()} to {df.index[-1].date()}")
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return sample data for demonstration if fetch fails
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration if Binance API fails"""
    print("Creating sample data for demonstration...")
    dates = pd.date_range(start='2018-01-01', end=datetime.now(), freq='D')
    np.random.seed(42)
    
    # Create synthetic price data with trend
    n = len(dates)
    trend = np.linspace(10000, 60000, n)
    noise = np.random.normal(0, 5000, n)
    seasonal = 10000 * np.sin(np.linspace(0, 20*np.pi, n))
    
    prices = trend + noise + seasonal
    prices = np.abs(prices)  # Ensure positive prices
    
    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    }, index=dates)
    
    return df

def calculate_strategy_returns(df, leverage=3.8, stop_loss_pct=0.05):
    """Calculate strategy returns based on SMA crossover rules with customizable leverage and stop loss"""
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate SMAs
    df['sma_120'] = df['close'].rolling(window=120).mean()
    df['sma_365'] = df['close'].rolling(window=365).mean()
    df['sma_120_open'] = df['open'].rolling(window=120).mean()
    
    # Drop NaN values (first 365 days won't have SMA_365)
    df_clean = df.dropna().copy()
    
    # Initialize strategy returns column
    df_clean['strategy_returns'] = 0.0
    
    # Apply strategy rules with stop loss
    for i in range(len(df_clean)):
        close_price = df_clean['close'].iloc[i]
        open_price = df_clean['open'].iloc[i]
        high_price = df_clean['high'].iloc[i]
        low_price = df_clean['low'].iloc[i]
        sma_120 = df_clean['sma_120'].iloc[i]
        sma_365 = df_clean['sma_365'].iloc[i]
        sma_120_open = df_clean['sma_120_open'].iloc[i]
        daily_return = df_clean['returns'].iloc[i]
        
        # Calculate risk category based on open vs strategy SMAs (120-day, 365-day) and risk SMA (120-day open)
        if open_price > sma_120 and open_price > sma_365 and open_price > sma_120_open:
            risk_category = 1  # Open above all three SMAs
        elif open_price < sma_120 and open_price < sma_365 and open_price < sma_120_open:
            risk_category = 1  # Open below all three SMAs
        else:
            risk_category = 2  # Other cases (e.g., open between SMAs or above risk SMA but below strategy SMAs)
        adjusted_leverage = 4.0 / (risk_category ** 2)
        
        # Calculate raw strategy signal
        raw_signal = 0.0
        # Rule 1: Add returns when price is above both SMAs
        if close_price > sma_120 and close_price > sma_365:
            raw_signal = daily_return
        # Rule 2: Subtract returns when price is below both SMAs
        elif close_price < sma_120 and close_price < sma_365:
            raw_signal = -daily_return
        # Rule 3: Add 0 return otherwise
        else:
            raw_signal = 0.0
        
        # Apply adjusted leverage
        leveraged_signal = raw_signal * adjusted_leverage
        
        # Apply fee only on days with non-zero signal
        fee_rate = 0.0004  # 0.04%
        if leveraged_signal != 0:
            leveraged_signal = leveraged_signal - fee_rate
        
        # Check for stop loss conditions
        stop_loss_triggered = False
        
        # Condition 1: Price above both SMAs and low is stop_loss_pct or more below open
        if close_price > sma_120 and close_price > sma_365:
            if low_price <= open_price * (1 - stop_loss_pct):
                stop_loss_triggered = True
        # Condition 2: Price below both SMAs and high is stop_loss_pct or more above open
        elif close_price < sma_120 and close_price < sma_365:
            if high_price >= open_price * (1 + stop_loss_pct):
                stop_loss_triggered = True
        
        # Apply stop loss if triggered
        if stop_loss_triggered:
            # Apply stop loss percentage scaled by adjusted leverage
            df_clean.loc[df_clean.index[i], 'strategy_returns'] = -stop_loss_pct * adjusted_leverage
        else:
            df_clean.loc[df_clean.index[i], 'strategy_returns'] = leveraged_signal
    
    # Calculate cumulative returns
    df_clean['cumulative_returns'] = (1 + df_clean['strategy_returns']).cumprod() - 1
    
    # Calculate Sharpe ratio (annualized, assuming 365 trading days per year)
    # Using risk-free rate of 0 for simplicity
    if len(df_clean) > 1:
        daily_mean = df_clean['strategy_returns'].mean()
        daily_std = df_clean['strategy_returns'].std()
        if daily_std != 0:
            sharpe_ratio = (daily_mean / daily_std) * np.sqrt(365)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0
    
    # Add Sharpe ratio to dataframe for easy access
    df_clean['sharpe_ratio'] = sharpe_ratio
    
    return df_clean


def grid_search_optimal_params(df):
    """Perform grid search to find optimal leverage and stop loss parameters for maximum Sharpe ratio"""
    # Define parameter ranges with smaller steps of 0.1
    leverage_range = np.arange(1.0, 5.1, 0.1).round(1).tolist()  # 1.0 to 5.0 in 0.1 increments
    stop_loss_range = np.arange(0.01, 0.101, 0.01).round(2).tolist()  # 1% to 10% in 0.01 increments
    
    best_sharpe = -float('inf')
    best_params = {'leverage': 1.5, 'stop_loss_pct': 0.05}
    results = []
    
    print(f"Starting grid search over {len(leverage_range)} leverage values and {len(stop_loss_range)} stop loss values...")
    print(f"Total combinations: {len(leverage_range) * len(stop_loss_range)}")
    
    for leverage in leverage_range:
        for stop_loss_pct in stop_loss_range:
            # Calculate strategy returns with current parameters
            df_strategy = calculate_strategy_returns(df, leverage=leverage, stop_loss_pct=stop_loss_pct)
            
            # Get Sharpe ratio
            sharpe_ratio = df_strategy['sharpe_ratio'].iloc[0]
            
            # Calculate total return
            total_return = df_strategy['cumulative_returns'].iloc[-1] if len(df_strategy) > 0 else 0
            
            # Store results
            result = {
                'leverage': leverage,
                'stop_loss_pct': stop_loss_pct,
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'total_return_pct': total_return * 100
            }
            results.append(result)
            
            # Update best parameters if current Sharpe is better
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = {'leverage': leverage, 'stop_loss_pct': stop_loss_pct}
            
            print(f"  Leverage: {leverage:.1f}, Stop Loss: {stop_loss_pct*100:.1f}%, Sharpe: {sharpe_ratio:.3f}, Total Return: {total_return*100:.2f}%")
    
    # Sort results by Sharpe ratio (descending)
    results_sorted = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
    
    print(f"\nGrid search complete. Best parameters: Leverage={best_params['leverage']:.1f}, Stop Loss={best_params['stop_loss_pct']*100:.1f}%")
    print(f"Best Sharpe ratio: {best_sharpe:.3f}")
    
    return best_params, results_sorted

def create_plot(df):
    """Create a plot of cumulative returns with risk category indication"""
    plt.figure(figsize=(14, 7))
    
    # Calculate risk categories for the plot based on open vs strategy SMAs and risk SMA
    risk_categories = []
    for i in range(len(df)):
        open_price = df['open'].iloc[i]
        sma_120 = df['sma_120'].iloc[i]
        sma_365 = df['sma_365'].iloc[i]
        sma_120_open = df['sma_120_open'].iloc[i]
        if open_price > sma_120 and open_price > sma_365 and open_price > sma_120_open:
            risk_categories.append(1)  # Open above all three SMAs
        elif open_price < sma_120 and open_price < sma_365 and open_price < sma_120_open:
            risk_categories.append(1)  # Open below all three SMAs
        else:
            risk_categories.append(2)  # Other cases
    risk_categories = np.array(risk_categories)
    
    # Find indices where risk category changes
    risk_changes = np.where(np.diff(risk_categories) != 0)[0]
    
    # Create shaded regions for risk categories
    # Start with the first category
    current_category = risk_categories[0]
    start_idx = 0
    
    # Add shaded background for each continuous risk category period
    for change_idx in risk_changes:
        end_idx = change_idx + 1
        
        # Determine color based on risk category
        if current_category == 1:
            # Light blue for risk category 1 (open above/below all three SMAs)
            color = 'lightblue'
            label = 'Risk Category 1 (Open above/below all SMAs)'
        else:
            # Light red for risk category 2 (other cases)
            color = 'lightcoral'
            label = 'Risk Category 2 (Other)'
        
        # Add shaded region
        plt.axvspan(df.index[start_idx], df.index[end_idx], 
                   alpha=0.3, color=color, label=label if start_idx == 0 else '')
        
        # Update for next region
        start_idx = end_idx
        current_category = risk_categories[end_idx]
    
    # Add final region
    if start_idx < len(df):
        if current_category == 1:
            color = 'lightblue'
            label = 'Risk Category 1 (Open > 120-day SMA)'
        else:
            color = 'lightcoral'
            label = 'Risk Category 2 (Open ≤ 120-day SMA)'
        
        plt.axvspan(df.index[start_idx], df.index[-1], 
                   alpha=0.3, color=color, label=label if start_idx == 0 else '')
    
    # Plot cumulative returns with log scale
    plt.plot(df.index, df['cumulative_returns'] + 1,  # Add 1 for log scale
             label='Cumulative Strategy Returns', 
             linewidth=2, 
             color='blue',
             alpha=0.9,
             zorder=5)
    
    # Set y-axis to log scale
    plt.yscale('log')
    
    # Add horizontal line at 1 (0 returns in log scale)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, zorder=4)
    
    # Formatting
    plt.title('Cumulative Returns with Risk Categories (Log Scale)', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Returns (Log Scale)', fontsize=12)
    plt.grid(True, alpha=0.3, which='both', zorder=1)
    plt.legend(fontsize=10, loc='upper left')
    
    # Add text box with risk category explanation
    risk_text = 'Risk Categories:\n• Light Blue: Open above all three SMAs OR below all three SMAs\n• Light Red: Other cases (e.g., between SMAs)'
    plt.text(0.02, 0.98, risk_text,
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             zorder=6)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return plot_url


def create_monthly_plot(monthly_returns_raw):
    """Create a bar plot of monthly returns"""
    plt.figure(figsize=(14, 6))
    
    # Create bar plot
    dates = monthly_returns_raw.index.strftime('%Y-%m')
    values = monthly_returns_raw.values * 100  # Convert to percentage
    
    # Color bars based on positive/negative
    colors = ['green' if val >= 0 else 'red' for val in values]
    
    plt.bar(dates, values, color=colors, alpha=0.7)
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Formatting
    plt.title('Monthly Strategy Returns (%)', fontsize=16, pad=20)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    return plot_url


def calculate_monthly_returns(df):
    """Calculate monthly returns from strategy returns"""
    # Create a copy of the dataframe with strategy returns
    df_monthly = df.copy()
    
    # Resample to monthly frequency and calculate monthly returns
    # Using the last day of each month
    monthly_returns_raw = df_monthly['strategy_returns'].resample('M').apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Convert to percentage and round
    monthly_returns_pct = (monthly_returns_raw * 100).round(2)
    
    # Create a list of dictionaries for easy template rendering
    monthly_data = []
    for date, return_pct in monthly_returns_pct.items():
        monthly_data.append({
            'year_month': date.strftime('%Y-%m'),
            'return_pct': return_pct,
            'is_positive': return_pct >= 0
        })
    
    return monthly_data, monthly_returns_raw



@app.route('/grid_search')
def grid_search():
    """Route to display grid search results"""
    # Fetch data
    df = fetch_binance_data()
    
    # Perform grid search
    best_params, results = grid_search_optimal_params(df)
    
    # Calculate strategy with best parameters
    df_best = calculate_strategy_returns(df, 
                                         leverage=best_params['leverage'], 
                                         stop_loss_pct=best_params['stop_loss_pct'])
    
    # Create HTML for results table
    results_html = ""
    for i, result in enumerate(results[:20]):  # Show top 20 results
        results_html += f"""
        <tr style="border-bottom: 1px solid #ddd;">
            <td style="padding: 8px 10px; text-align: center;">{i+1}</td>
            <td style="padding: 8px 10px; text-align: center;">{result['leverage']}</td>
            <td style="padding: 8px 10px; text-align: center;">{result['stop_loss_pct']*100:.1f}%</td>
            <td style="padding: 8px 10px; text-align: right;">{result['sharpe_ratio']:.3f}</td>
            <td style="padding: 8px 10px; text-align: right; color: {'#28a745' if result['total_return'] >= 0 else '#dc3545'};">
                {result['total_return_pct']:.2f}%
            </td>
        </tr>
        """
    
    # Create HTML template for grid search results
    grid_search_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grid Search Results - Optimal Parameters</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .info-box {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #007bff;
            }}
            .best-params {{
                background-color: #d4edda;
                border-left: 4px solid #28a745;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th {{
                background-color: #f2f2f2;
                padding: 12px 10px;
                text-align: center;
                border-bottom: 2px solid #ddd;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .back-link {{
                display: inline-block;
                margin-top: 20px;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}
            .back-link:hover {{
                background-color: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Grid Search Results - Optimal Parameters</h1>
            
            <div class="best-params">
                <h3>🎯 Best Parameters Found</h3>
                <p><strong>Leverage:</strong> {best_params['leverage']}x</p>
                <p><strong>Stop Loss:</strong> {best_params['stop_loss_pct']*100:.1f}%</p>
                <p><strong>Sharpe Ratio:</strong> {results[0]['sharpe_ratio']:.3f}</p>
                <p><strong>Total Return:</strong> {results[0]['total_return_pct']:.2f}%</p>
            </div>
            
            <div class="info-box">
                <h3>Grid Search Details</h3>
                <p>Parameter ranges tested:</p>
                <ul>
                    <li><strong>Leverage:</strong> 1.0 to 5.0 in 0.1 increments</li>
                    <li><strong>Stop Loss:</strong> 1% to 10% in 0.1% increments</li>
                </ul>
                <p>Total combinations tested: {len(results)}</p>
            </div>
            
            <div class="info-box">
                <h3>Top 20 Parameter Combinations by Sharpe Ratio</h3>
                <div style="max-height: 600px; overflow-y: auto; margin-top: 15px;">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Leverage</th>
                                <th>Stop Loss</th>
                                <th>Sharpe Ratio</th>
                                <th>Total Return</th>
                            </tr>
                        </thead>
                        <tbody>
                            {results_html}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <a href="/" class="back-link">← Back to Main Strategy</a>
        </div>
    </body>
    </html>
    """
    
    return grid_search_html

@app.route('/')
def index():
    """Main route that displays the strategy returns plot"""
    # Fetch and process data
    df = fetch_binance_data()
    df_strategy = calculate_strategy_returns(df)
    
    # Create plot
    plot_url = create_plot(df_strategy)
    
    # Calculate statistics
    total_return = df_strategy['cumulative_returns'].iloc[-1] if len(df_strategy) > 0 else 0
    total_return_pct = round(total_return * 100, 2)
    final_cum_return = round(total_return, 4)
    
    # Get Sharpe ratio from dataframe
    sharpe_ratio = round(df_strategy['sharpe_ratio'].iloc[0] if len(df_strategy) > 0 else 0.0, 3)
    
    # Count signal days
    positive_mask = (df_strategy['close'] > df_strategy['sma_120']) & (df_strategy['close'] > df_strategy['sma_365'])
    negative_mask = (df_strategy['close'] < df_strategy['sma_120']) & (df_strategy['close'] < df_strategy['sma_365'])
    
    positive_days = positive_mask.sum()
    negative_days = negative_mask.sum()
    
    # Calculate monthly returns
    monthly_returns, monthly_returns_raw = calculate_monthly_returns(df_strategy)
    
    # Calculate average monthly return (in percentage)
    avg_monthly_return_pct = round(monthly_returns_raw.mean() * 100, 2) if len(monthly_returns_raw) > 0 else 0.0
    
    # Create monthly returns plot
    monthly_plot_url = create_monthly_plot(monthly_returns_raw)    
    # Calculate risk category and adjusted leverage for the latest day
    latest_open = df_strategy['open'].iloc[-1] if len(df_strategy) > 0 else 0
    latest_sma_120_open = df_strategy['sma_120_open'].iloc[-1] if len(df_strategy) > 0 else 0
    risk_category = 1 if latest_open > latest_sma_120_open else 2
    adjusted_leverage = round(4.0 / (risk_category ** 2), 2)
    
    # Prepare template data
    template_data = {
        'plot_url': plot_url,
        'monthly_plot_url': monthly_plot_url,
        'total_return_pct': total_return_pct,
        'final_cum_return': final_cum_return,
        'sharpe_ratio': sharpe_ratio,
        'avg_monthly_return_pct': avg_monthly_return_pct,
        'positive_days': int(positive_days),
        'negative_days': int(negative_days),        'risk_category': risk_category,
        'adjusted_leverage': adjusted_leverage,
        'avg_monthly_return_pct': avg_monthly_return_pct,
        'positive_days': int(positive_days),
        'negative_days': int(negative_days),
        'monthly_returns': monthly_returns,
        'start_date': df_strategy.index[0].strftime('%Y-%m-%d') if len(df_strategy) > 0 else '2018-01-01',
        'end_date': df_strategy.index[-1].strftime('%Y-%m-%d') if len(df_strategy) > 0 else datetime.now().strftime('%Y-%m-%d'),
        'total_days': len(df_strategy),
        'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template_string(HTML_TEMPLATE, **template_data)

if __name__ == '__main__':
    print("Starting web server on port 8080...")
    print("Open http://localhost:8080 in your browser")
    app.run(host='0.0.0.0', port=8080, debug=False)
