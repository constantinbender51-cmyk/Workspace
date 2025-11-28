import gdown
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import threading

# Download file from Google Drive
def download_data():
    print("Downloading data from Google Drive...")
    url = "https://drive.google.com/uc?id=1Bn7Bv1Z4Evxl3N4Ep_wYwaMc45VpOvtc"
    output = "data.csv"
    gdown.download(url, output, quiet=False)
    print(f"Downloaded to {output}")
    return output

# Create features using 30 days of OHLCV data
def create_features(df, lookback=30):
    features = []
    targets = []
    
    # Ensure we have the required columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Check which columns exist (case-insensitive)
    df.columns = df.columns.str.lower()
    available_cols = [col for col in ohlcv_cols if col in df.columns]
    
    print(f"Available OHLCV columns: {available_cols}")
    print(f"Checking for optimal_position column...")
    
    if 'optimal_position' not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise ValueError("optimal_position column not found in dataset")
    
    # Create lagged features
    for i in range(lookback, len(df)):
        feature_vector = []
        
        # Add lagged OHLCV values for the past 'lookback' days
        for col in available_cols:
            for lag in range(lookback):
                feature_vector.append(df[col].iloc[i - lookback + lag])
        
        features.append(feature_vector)
        targets.append(df['optimal_position'].iloc[i])
    
    return np.array(features), np.array(targets)

# Train model and generate predictions
def train_model(csv_file):
    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("Creating features...")
    X, y = create_features(df, lookback=30)
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y.astype(int) + 1)}")  # Assuming -1, 0, 1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=4000, random_state=42, multi_class='multinomial')
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Get prediction probabilities
    y_pred_proba_train = model.predict_proba(X_train)
    y_pred_proba_test = model.predict_proba(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nModel Performance:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=['Short (-1)', 'Neutral (0)', 'Long (1)']))
    
    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    
    return {
        'y_train': y_train,
        'y_pred_train': y_pred_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cm_train': cm_train,
        'cm_test': cm_test,
        'classes': model.classes_
    }

# Create visualization
def create_plot(results):
    print("Creating visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Logistic Regression Model: Optimal Position Classification', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(results['cm_train'], annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Short', 'Neutral', 'Long'],
                yticklabels=['Short', 'Neutral', 'Long'])
    ax1.set_title(f"Training Confusion Matrix\nAccuracy: {results['train_acc']:.4f}")
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    
    # Plot 2: Test Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(results['cm_test'], annot=True, fmt='d', cmap='Oranges', ax=ax2,
                xticklabels=['Short', 'Neutral', 'Long'],
                yticklabels=['Short', 'Neutral', 'Long'])
    ax2.set_title(f"Test Confusion Matrix\nAccuracy: {results['test_acc']:.4f}")
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    # Plot 3: Class Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    unique_train, counts_train = np.unique(results['y_train'], return_counts=True)
    unique_test, counts_test = np.unique(results['y_test'], return_counts=True)
    
    x = np.arange(len(unique_train))
    width = 0.35
    ax3.bar(x - width/2, counts_train, width, label='Train', alpha=0.8)
    ax3.bar(x + width/2, counts_test, width, label='Test', alpha=0.8)
    ax3.set_xlabel('Position Class')
    ax3.set_ylabel('Count')
    ax3.set_title('Class Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Short (-1)', 'Neutral (0)', 'Long (1)'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Training Time Series
    ax4 = fig.add_subplot(gs[1, :])
    x_train = np.arange(len(results['y_train']))
    ax4.plot(x_train, results['y_train'], label='Actual', alpha=0.6, linewidth=1.5, color='blue')
    ax4.plot(x_train, results['y_pred_train'], label='Predicted', alpha=0.6, linewidth=1.5, color='red', linestyle='--')
    ax4.set_xlabel('Time Index')
    ax4.set_ylabel('Position')
    ax4.set_title('Training Set: Actual vs Predicted Positions')
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['Short (-1)', 'Neutral (0)', 'Long (1)'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Test Time Series
    ax5 = fig.add_subplot(gs[2, :])
    x_test = np.arange(len(results['y_test']))
    ax5.plot(x_test, results['y_test'], label='Actual', alpha=0.6, linewidth=1.5, color='blue')
    ax5.plot(x_test, results['y_pred_test'], label='Predicted', alpha=0.6, linewidth=1.5, color='orange', linestyle='--')
    ax5.set_xlabel('Time Index')
    ax5.set_ylabel('Position')
    ax5.set_title('Test Set: Actual vs Predicted Positions')
    ax5.set_yticks([-1, 0, 1])
    ax5.set_yticklabels(['Short (-1)', 'Neutral (0)', 'Long (1)'])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved as prediction_results.png")
    
    # Create HTML page
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>OHLCV Logistic Regression Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            img {{
                width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .metrics {{
                background-color: #e8f4f8;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .metrics h2 {{
                margin-top: 0;
                color: #2c3e50;
            }}
            .metric-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 15px;
            }}
            .metric-box {{
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }}
            .metric-label {{
                font-weight: bold;
                color: #555;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 24px;
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OHLCV Logistic Regression Model Results</h1>
            <div class="metrics">
                <h2>Model Performance</h2>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-label">Training Accuracy</div>
                        <div class="metric-value">{:.2%}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Test Accuracy</div>
                        <div class="metric-value">{:.2%}</div>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <p><strong>Model Type:</strong> Multinomial Logistic Regression</p>
                    <p><strong>Features:</strong> 30 days of OHLCV data (lagged features)</p>
                    <p><strong>Target Classes:</strong> -1 (Short), 0 (Neutral), 1 (Long)</p>
                    <p><strong>Train/Test Split:</strong> 80% / 20% (time-series split)</p>
                </div>
            </div>
            <img src="prediction_results.png" alt="Prediction Results">
        </div>
    </body>
    </html>
    """.format(results['train_acc'], results['test_acc'])
    
    with open('index.html', 'w') as f:
        f.write(html_content)
    
    print("HTML page created as index.html")

# Start web server
def start_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
    
    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    print(f"\n{'='*60}")
    print(f"Web server running at http://0.0.0.0:8080")
    print(f"Access from your browser at http://localhost:8080")
    print(f"Press Ctrl+C to stop the server")
    print(f"{'='*60}\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

# Main execution
if __name__ == "__main__":
    try:
        # Download data
        csv_file = download_data()
        
        # Train model and get predictions
        results = train_model(csv_file)
        
        # Create visualization
        create_plot(results)
        
        # Start web server
        start_server()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
