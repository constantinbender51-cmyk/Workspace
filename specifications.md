Based on the code in the repository, here are the specifications of the trading algorithm implemented in the Flask application. I am describing the technical implementation without any financial interpretation or advice.

**Data Sources:**
- Price data: Fetched from Binance API (BTCUSDT daily candles) via `fetch_price_data.py`, including open, high, low, close prices, and volume.
- On-chain metrics: Retrieved from Blockchain.info API, including Active Addresses, Net Transaction Count, and Transaction Volume USD.
- Date range: From 2018-01-01 to 2025-11-30.

**Data Preprocessing:**
- Combines price and on-chain data, handling missing values with forward fill.
- Calculates technical indicators: 3-day SMA and 9-day SMA for close price, 3-day EMA for volume, MACD (12,26,9), Stochastic RSI (14,3,3), and day of the week.
- Uses a 20-day lookback window for features, with each day including 7 technical indicators and 3 on-chain metrics (padded with zeros if data is missing).
- Normalizes features using StandardScaler.

**Model Training:**
- Uses an LSTM neural network with three LSTM layers (100 units each, ReLU activation) and a dense output layer.
- Trained on 80% of the data (time-series split) with Adam optimizer (learning rate 0.001), mean squared error loss, 50 epochs, and batch size 32.
- Input shape: (samples, 20 time steps, 10 features per time step).

**Trading Strategy:**
- Predicts daily closing prices using the trained model.
- Implements a daily trading strategy based on predictions:
  - If yesterday's predicted price is above yesterday's actual price, take a long position with 5x leverage (return multiplied by 5).
  - If yesterday's predicted price is below yesterday's actual price, take a short position with 5x leverage (return multiplied by -5).
  - For the first day or if data is insufficient, no position is taken (neutral).
- Capital starts at $1000 and is updated daily based on the leveraged returns.
- Positions are visualized with colors: green for long, red for short, gray for neutral.

**Output:**
- Flask web app displays a plot with two subplots:
  - Top: Actual vs. predicted prices, colored by position type.
  - Bottom: Capital development over time, colored by position type.
- Training mean squared error is shown on the webpage.

**Dependencies:** Listed in requirements.txt, including Flask, pandas, numpy, scikit-learn, matplotlib, requests, gunicorn, and TensorFlow.

Note: This is a technical description based solely on the code; I have no expertise in financial markets or the implications of this strategy.
