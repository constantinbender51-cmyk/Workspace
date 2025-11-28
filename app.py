import gdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from flask import Flask, render_template_string, send_file
import io
import base64

# Download file from Google Drive
file_id = "16gXMCEX5WxcpEaR0xg-Wmr0dsDPniSkg"
url = f"https://drive.google.com/uc?id={file_id}"
output = "daily_ohlcv.csv"

print("Downloading data...")
gdown.download(url, output, quiet=False)

# Load data
print("Loading data...")
df = pd.read_csv(output)
print(f"Loaded {len(df)} rows")
print(df.head())

# Calculate optimal positions
print("\nCalculating optimal positions...")

TRANSACTION_COST = 0.032  # 3.2% total (0.8% + 0.8%)

# Calculate daily returns
df['return'] = df['close'].pct_change()

# Determine optimal position for each day (what position to hold TODAY to profit from TOMORROW)
# Note: We need returns > 2x TRANSACTION_COST to account for entry + exit fees
df['optimal_position'] = 0  # 0 = FLAT, 1 = LONG, -1 = SHORT

for i in range(len(df) - 1):
    next_return = df.loc[i + 1, 'return']
    
    if next_return > (2 * TRANSACTION_COST):
        df.loc[i, 'optimal_position'] = 1  # LONG
    elif next_return < -(2 * TRANSACTION_COST):
        df.loc[i, 'optimal_position'] = -1  # SHORT
    else:
        df.loc[i, 'optimal_position'] = 0  # FLAT

# Calculate capital development
capital = [1.0]  # Start with $1
position = 0  # Start FLAT
position_changes = 0

for i in range(1, len(df)):
    current_capital = capital[-1]
    
    target_position = df.loc[i - 1, 'optimal_position']
    daily_return = df.loc[i, 'return']
    
    # Check if position change
    if target_position != position:
        current_capital *= (1 - TRANSACTION_COST)  # Pay transaction cost
        position = target_position
        position_changes += 1
    
    # Apply daily return based on position
    if position == 1:  # LONG
        current_capital *= (1 + daily_return)
    elif position == -1:  # SHORT
        current_capital *= (1 - daily_return)
    # FLAT: no change
    
    capital.append(current_capital)

df['capital'] = capital

# Calculate statistics
total_return = (capital[-1] - 1) * 100
buy_hold_return = ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100
days_long = (df['optimal_position'] == 1).sum()
days_short = (df['optimal_position'] == -1).sum()
days_flat = (df['optimal_position'] == 0).sum()

print(f"\n{'='*60}")
print(f"OPTIMAL STRATEGY RESULTS")
print(f"{'='*60}")
print(f"Total Return:        {total_return:.2f}%")
print(f"Buy & Hold Return:   {buy_hold_return:.2f}%")
print(f"Position Changes:    {position_changes}")
print(f"Days Long:           {days_long} ({days_long/len(df)*100:.1f}%)")
print(f"Days Short:          {days_short} ({days_short/len(df)*100:.1f}%)")
print(f"Days Flat:           {days_flat} ({days_flat/len(df)*100:.1f}%)")
print(f"Final Capital:       ${capital[-1]:.4f}")
print(f"{'='*60}\n")

# Create visualization
print("Creating visualization...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('Optimal Trading Strategy with Perfect Foresight', fontsize=16, fontweight='bold')

# Plot 1: Price with position background
ax1 = axes[0]
ax1.plot(df.index, df['close'], color='black', linewidth=1.5, label='Close Price')
ax1.set_ylabel('Price', fontsize=12)
ax1.set_title('Price Chart with Optimal Positions', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add colored background for positions
ymin, ymax = ax1.get_ylim()
for i in range(len(df)):
    pos = df.loc[i, 'optimal_position']
    if pos == 1:  # LONG
        color = 'green'
        alpha = 0.1
    elif pos == -1:  # SHORT
        color = 'red'
        alpha = 0.1
    else:  # FLAT
        continue
    
    ax1.axvspan(i, i + 1, color=color, alpha=alpha)

# Plot 2: Position indicator
ax2 = axes[1]
ax2.fill_between(df.index, 0, df['optimal_position'], 
                  where=(df['optimal_position'] > 0), color='green', alpha=0.5, label='LONG')
ax2.fill_between(df.index, 0, df['optimal_position'], 
                  where=(df['optimal_position'] < 0), color='red', alpha=0.5, label='SHORT')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Position', fontsize=12)
ax2.set_ylim(-1.5, 1.5)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['SHORT', 'FLAT', 'LONG'])
ax2.set_title('Optimal Position Over Time', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Capital development
ax3 = axes[2]
ax3.plot(df.index, df['capital'], color='blue', linewidth=2, label='Optimal Strategy')
buy_hold_capital = df['close'] / df['close'].iloc[0]
ax3.plot(df.index, buy_hold_capital, color='orange', linewidth=2, 
         linestyle='--', label='Buy & Hold', alpha=0.7)
ax3.set_ylabel('Capital (Starting = $1)', fontsize=12)
ax3.set_xlabel('Day', fontsize=12)
ax3.set_title('Capital Development', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend()

# Add statistics text box
stats_text = f"Total Return: {total_return:.2f}%\n"
stats_text += f"Position Changes: {position_changes}\n"
stats_text += f"Transaction Cost: {TRANSACTION_COST*100}%"
ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save plot to bytes
img_bytes = io.BytesIO()
plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
img_bytes.seek(0)
img_base64 = base64.b64encode(img_bytes.read()).decode()

# Create Flask app
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Optimal Trading Strategy</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-box h3 {
            margin: 0;
            font-size: 14px;
            opacity: 0.9;
        }
        .stat-box p {
            margin: 10px 0 0 0;
            font-size: 24px;
            font-weight: bold;
        }
        img {
            width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .download-link {
            display: block;
            text-align: center;
            margin: 20px 0;
        }
        .download-link a {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-size: 16px;
        }
        .download-link a:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Optimal Trading Strategy Analysis</h1>
        
        <div class="download-link">
            <a href="/download">Download OHLCV + Optimal Positions CSV</a>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Total Return</h3>
                <p>{{ total_return }}%</p>
            </div>
            <div class="stat-box">
                <h3>Buy & Hold Return</h3>
                <p>{{ buy_hold_return }}%</p>
            </div>
            <div class="stat-box">
                <h3>Position Changes</h3>
                <p>{{ position_changes }}</p>
            </div>
            <div class="stat-box">
                <h3>Days Long</h3>
                <p>{{ days_long }}</p>
            </div>
            <div class="stat-box">
                <h3>Days Short</h3>
                <p>{{ days_short }}</p>
            </div>
            <div class="stat-box">
                <h3>Days Flat</h3>
                <p>{{ days_flat }}</p>
            </div>
        </div>
        
        <img src="data:image/png;base64,{{ img_data }}" alt="Trading Strategy Visualization">
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_TEMPLATE,
        total_return=f"{total_return:.2f}",
        buy_hold_return=f"{buy_hold_return:.2f}",
        position_changes=position_changes,
        days_long=days_long,
        days_short=days_short,
        days_flat=days_flat,
        img_data=img_base64
    )

@app.route('/download')
def download_csv():
    # Filter DataFrame to include only datetime, OHLCV, and optimal_position columns
    # Check if 'datetime' column exists, otherwise use the first column as datetime
    available_columns = df.columns.tolist()
    datetime_column = 'datetime' if 'datetime' in available_columns else available_columns[0]
    columns_to_include = [datetime_column, 'open', 'high', 'low', 'close', 'volume', 'optimal_position']
    # Ensure all columns exist in the DataFrame
    columns_to_include = [col for col in columns_to_include if col in available_columns]
    filtered_df = df[columns_to_include]
    
    # Create a CSV in memory
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Send as file download
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='ohlcv_optimal_positions.csv'
    )

print("Starting web server on http://0.0.0.0:8080")
print("Press Ctrl+C to stop")
app.run(host='0.0.0.0', port=8080, debug=False)
