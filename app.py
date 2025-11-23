#!/usr/bin/env python3
"""
lr_live_bi.py - LSTM-based BTC Trading Strategy
Uses LSTM neural network with 20-day lookback and on-chain metrics
Trades daily at 00:01 UTC based on predicted vs actual price comparison
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple
import subprocess
import numpy as np
import pandas as pd
import requests

import kraken_futures as kf
import kraken_ohlc
import binance_ohlc

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam

dry = os.getenv("DRY_RUN", "false").lower() in {"1", "true", "yes"}
RUN_TRADE_NOW = os.getenv("RUN_TRADE_NOW", "false").lower() in {"1", "true", "yes"}

SYMBOL_FUTS_UC = "PF_XBTUSD"
SYMBOL_FUTS_LC = "pf_xbtusd"
SYMBOL_OHLC_KRAKEN = "XBTUSD"
SYMBOL_OHLC_BINANCE = "BTCUSDT"
INTERVAL_KRAKEN = 1440
INTERVAL_BINANCE = "1d"
LOOKBACK = 20
LEV = 5.0
STOP_LOSS_PCT = 0.05  # 5% stop loss
STATE_FILE = Path("lstm_state.json")
TEST_SIZE_BTC = 0.0001

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("lstm_live")


def fetch_onchain_metrics(start_date: str = "2022-08-10") -> pd.DataFrame:
    """Fetch on-chain metrics from Blockchain.info API"""
    BASE_URL = "https://api.blockchain.info/charts/"
    METRICS = {
        'Active_Addresses': 'n-unique-addresses',
        'Net_Transaction_Count': 'n-transactions',
        'Transaction_Volume_USD': 'estimated-transaction-volume-usd',
    }
    
    all_data = []
    for metric_name, chart_endpoint in METRICS.items():
        log.info(f"Fetching {metric_name}...")
        params = {
            'format': 'json',
            'start': start_date,
            'timespan': '2years',
            'rollingAverage': '1d'
        }
        url = f"{BASE_URL}{chart_endpoint}"
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'values' in data and data['values']:
                df = pd.DataFrame(data['values'])
                df['date'] = pd.to_datetime(df['x'], unit='s', utc=True).dt.tz_localize(None)
                df = df.set_index('date')['y'].rename(metric_name)
                all_data.append(df)
                log.info(f"Fetched {len(df)} rows for {metric_name}")
            time.sleep(2)  # Rate limiting
        except Exception as e:
            log.warning(f"Failed to fetch {metric_name}: {e}")
            # Return empty series if fetch fails
            all_data.append(pd.Series(name=metric_name, dtype=float))
    
    if all_data:
        return pd.concat(all_data, axis=1)
    return pd.DataFrame()


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators matching the app.py specification"""
    df = df.copy()
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            raise ValueError("DataFrame must have a datetime index or 'date'/'timestamp' column")
    
    # 3-day SMA for close price
    df['sma_3_close'] = df['close'].rolling(window=3).mean()
    
    # 9-day SMA for close price
    df['sma_9_close'] = df['close'].rolling(window=9).mean()
    
    # 3-day EMA for volume
    df['ema_3_volume'] = df['volume'].ewm(span=3, adjust=False).mean()
    
    # MACD (12,26,9)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema_12 - ema_26
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    
    # Stochastic RSI (14,3,3)
    rsi_period = 14
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    df['stoch_rsi'] = 100 * (rsi - rsi_min) / (rsi_max - rsi_min)
    
    # Day of week (1-7)
    df['day_of_week'] = df.index.dayofweek + 1
    
    return df


def prepare_lstm_features(df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Prepare features for LSTM model with 20-day lookback"""
    df = calculate_technical_indicators(df)
    
    # Merge with on-chain data if available
    if onchain_df is not None and not onchain_df.empty:
        df = df.join(onchain_df, how='left')
        df = df.ffill()
    
    # Ensure on-chain columns exist (fill with 0 if not available)
    for col in ['Active_Addresses', 'Net_Transaction_Count', 'Transaction_Volume_USD']:
        if col not in df.columns:
            df[col] = 0
    
    df_clean = df.dropna()
    
    features = []
    targets = []
    
    for i in range(len(df_clean)):
        if i >= 40:  # Ensure enough history
            feature = []
            # Add features from the last 20 days (t-20 to t-1)
            for lookback in range(1, 21):
                if i - lookback >= 0:
                    feature.append(df_clean['sma_3_close'].iloc[i - lookback])
                    feature.append(df_clean['sma_9_close'].iloc[i - lookback])
                    feature.append(df_clean['ema_3_volume'].iloc[i - lookback])
                    feature.append(df_clean['macd_line'].iloc[i - lookback])
                    feature.append(df_clean['signal_line'].iloc[i - lookback])
                    feature.append(df_clean['stoch_rsi'].iloc[i - lookback])
                    feature.append(df_clean['day_of_week'].iloc[i - lookback])
                    feature.append(df_clean['Net_Transaction_Count'].iloc[i - lookback])
                    feature.append(df_clean['Transaction_Volume_USD'].iloc[i - lookback])
                    feature.append(df_clean['Active_Addresses'].iloc[i - lookback])
                else:
                    feature.extend([0] * 10)
            
            features.append(feature)
            targets.append(df_clean['close'].iloc[i])
    
    features = np.array(features)
    targets = np.array(targets)
    
    # Remove rows with NaN
    valid_indices = ~np.isnan(features).any(axis=1)
    features = features[valid_indices]
    targets = targets[valid_indices]
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    return features_normalized, targets, scaler


class LSTMModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.train_mse = None
        self.last_trained = None
        
    def build_model(self):
        """Build LSTM model matching app.py specification"""
        model = Sequential()
        model.add(Input(shape=(20, 10)))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def fit(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None):
        """Train LSTM model on historical data"""
        log.info("Preparing features for LSTM training...")
        features, targets, scaler = prepare_lstm_features(df, onchain_df)
        self.scaler = scaler
        
        # 80/20 split for training
        split_idx = int(len(features) * 0.8)
        X_train = features[:split_idx]
        y_train = targets[:split_idx]
        
        # Reshape for LSTM: (samples, time_steps, features)
        X_train_reshaped = X_train.reshape(X_train.shape[0], 20, 10)
        
        log.info(f"Training LSTM on {len(X_train)} samples...")
        self.model = self.build_model()
        self.model.fit(X_train_reshaped, y_train, epochs=50, batch_size=32, verbose=0)
        
        # Calculate training MSE
        train_predictions = self.model.predict(X_train_reshaped, verbose=0).flatten()
        self.train_mse = np.mean((y_train - train_predictions) ** 2)
        self.last_trained = datetime.now(timezone.utc).isoformat()
        
        log.info(f"Training complete. MSE: {self.train_mse:.2f}")
    
    def predict_last(self, df: pd.DataFrame, onchain_df: pd.DataFrame = None) -> float:
        """Predict closing price for the last available data point"""
        df_calc = calculate_technical_indicators(df)
        
        if onchain_df is not None and not onchain_df.empty:
            df_calc = df_calc.join(onchain_df, how='left')
            df_calc = df_calc.ffill()
        
        for col in ['Active_Addresses', 'Net_Transaction_Count', 'Transaction_Volume_USD']:
            if col not in df_calc.columns:
                df_calc[col] = 0
        
        df_calc = df_calc.dropna()
        
        if len(df_calc) < 40:
            raise ValueError("Not enough data for prediction")
        
        # Build feature for last row
        i = len(df_calc) - 1
        feature = []
        for lookback in range(1, 21):
            if i - lookback >= 0:
                feature.append(df_calc['sma_3_close'].iloc[i - lookback])
                feature.append(df_calc['sma_9_close'].iloc[i - lookback])
                feature.append(df_calc['ema_3_volume'].iloc[i - lookback])
                feature.append(df_calc['macd_line'].iloc[i - lookback])
                feature.append(df_calc['signal_line'].iloc[i - lookback])
                feature.append(df_calc['stoch_rsi'].iloc[i - lookback])
                feature.append(df_calc['day_of_week'].iloc[i - lookback])
                feature.append(df_calc['Net_Transaction_Count'].iloc[i - lookback])
                feature.append(df_calc['Transaction_Volume_USD'].iloc[i - lookback])
                feature.append(df_calc['Active_Addresses'].iloc[i - lookback])
            else:
                feature.extend([0] * 10)
        
        feature = np.array([feature])
        feature_normalized = self.scaler.transform(feature)
        feature_reshaped = feature_normalized.reshape(1, 20, 10)
        
        prediction = self.model.predict(feature_reshaped, verbose=0)[0][0]
        return float(prediction)


def portfolio_usd(api: kf.KrakenFuturesApi) -> float:
    return float(api.get_accounts()["accounts"]["flex"]["portfolioValue"])


def mark_price(api: kf.KrakenFuturesApi) -> float:
    tk = api.get_tickers()
    for t in tk["tickers"]:
        if t["symbol"] == SYMBOL_FUTS_UC:
            return float(t["markPrice"])
    raise RuntimeError("Mark-price for PF_XBTUSD not found")


def cancel_all(api: kf.KrakenFuturesApi):
    log.info("Cancelling all orders")
    try:
        api.cancel_all_orders()
    except Exception as e:
        log.warning("cancel_all_orders failed: %s", e)


def flatten_position(api: kf.KrakenFuturesApi):
    pos = api.get_open_positions()
    for p in pos.get("openPositions", []):
        if p["symbol"] != SYMBOL_FUTS_UC:
            continue
        side = "sell" if p["side"] == "long" else "buy"
        size = abs(float(p["size"]))
        log.info("Flatten %s position %.4f BTC", p["side"], size)
        api.send_order({
            "orderType": "mkt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": round(size, 4),
        })


def place_stop_loss(api: kf.KrakenFuturesApi, side: str, size_btc: float, fill_price: float):
    """Place 5% stop loss"""
    if side == "buy":
        stop_price = fill_price * (1 - STOP_LOSS_PCT)
        stop_side = "sell"
    else:
        stop_price = fill_price * (1 + STOP_LOSS_PCT)
        stop_side = "buy"
    
    limit_price = stop_price * (0.9999 if stop_side == "sell" else 1.0001)
    
    log.info("Placing %d%% stop loss: %s at %.2f", int(STOP_LOSS_PCT*100), stop_side, stop_price)
    api.send_order({
        "orderType": "stp",
        "symbol": SYMBOL_FUTS_LC,
        "side": stop_side,
        "size": round(size_btc, 4),
        "stopPrice": int(round(stop_price)),
        "limitPrice": int(round(limit_price)),
    })


def smoke_test(api: kf.KrakenFuturesApi):
    """Run smoke test to verify API connectivity"""
    log.info("=== Smoke-test start ===")
    
    try:
        # Test portfolio access
        usd = portfolio_usd(api)
        log.info(f"Portfolio value: ${usd:.2f} USD")
        
        # Test market data
        mp = mark_price(api)
        log.info(f"BTC mark price: ${mp:.2f}")
        
        # Check open positions
        current_pos = get_current_position(api)
        if current_pos:
            log.info(f"Open position: {current_pos['signal']} {current_pos['size_btc']:.4f} BTC @ ${current_pos['fill_price']:.2f}")
        else:
            log.info("No open positions")
        
        log.info("=== Smoke-test complete ===")
        return True
    except Exception as e:
        log.error(f"Smoke test failed: {e}")
        return False


def update_state_with_current_position(api: kf.KrakenFuturesApi):
    """Update state file with current position from Kraken"""
    state = load_state()
    
    # Get current position
    current_pos = get_current_position(api)
    portfolio_value = portfolio_usd(api)
    
    # Update state with current info
    state["current_position"] = current_pos
    state["current_portfolio_value"] = portfolio_value
    
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
    
    # Calculate performance if we have starting capital
    if state["starting_capital"]:
        total_return = (portfolio_value - state["starting_capital"]) / state["starting_capital"] * 100
        state["performance"] = {
            "current_value": portfolio_value,
            "starting_capital": state["starting_capital"],
            "total_return_pct": total_return,
            "total_trades": len(state.get("trades", [])),
        }
    
    save_state(state)
    log.info(f"Updated state with current position and portfolio value: ${portfolio_value:.2f}")


def load_state() -> Dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {
        "trades": [],
        "predictions": [],
        "starting_capital": None,
        "performance": {}
    }


def save_state(st: Dict):
    STATE_FILE.write_text(json.dumps(st, indent=2))


def generate_signal(model: LSTMModel, df_current: pd.DataFrame, onchain_df: pd.DataFrame) -> Tuple[str, float, float, float]:
    """
    Generate trading signal based on yesterday's prediction vs actual
    Returns: (signal, today_prediction, yesterday_prediction, yesterday_actual)
    """
    # Need at least 2 days to compare yesterday's prediction vs actual
    if len(df_current) < 2:
        raise ValueError("Need at least 2 days of data")
    
    # Get yesterday's actual price
    yesterday_actual = float(df_current['close'].iloc[-2])
    
    # Get yesterday's data and predict
    df_yesterday = df_current.iloc[:-1]
    yesterday_prediction = model.predict_last(df_yesterday, onchain_df)
    
    # Get today's prediction
    today_prediction = model.predict_last(df_current, onchain_df)
    
    # Generate signal
    if yesterday_prediction > yesterday_actual:
        signal = "LONG"
    else:
        signal = "SHORT"
    
    log.info(f"Yesterday: predicted={yesterday_prediction:.2f}, actual={yesterday_actual:.2f}")
    log.info(f"Today: predicted={today_prediction:.2f}")
    log.info(f"Signal: {signal}")
    
    return signal, today_prediction, yesterday_prediction, yesterday_actual


def daily_trade(api: kf.KrakenFuturesApi, model: LSTMModel, onchain_df: pd.DataFrame):
    """Execute daily trading strategy"""
    state = load_state()
    
    # Get current market data
    df = kraken_ohlc.get_ohlc(SYMBOL_OHLC_KRAKEN, INTERVAL_KRAKEN)
    current_price = mark_price(api)
    portfolio_value = portfolio_usd(api)
    
    # Set starting capital on first run
    if state["starting_capital"] is None:
        state["starting_capital"] = portfolio_value
    
    # Generate signal
    signal, today_pred, yesterday_pred, yesterday_actual = generate_signal(model, df, onchain_df)
    
    # Close existing position
    log.info("Closing any existing positions...")
    cancel_all(api)
    flatten_position(api)
    time.sleep(2)
    
    # Calculate position size
    collateral = portfolio_usd(api)  # Get fresh portfolio value after flatten
    notional = collateral * LEV
    size_btc = round(notional / current_price, 4)
    
    side = "buy" if signal == "LONG" else "sell"
    
    log.info(f"Opening {signal} position: {side} {size_btc} BTC at ~{current_price:.2f}")
    
    if dry:
        log.info(f"DRY-RUN: {signal} {size_btc} BTC at {current_price:.2f}")
        fill_price = current_price
    else:
        ord = api.send_order({
            "orderType": "mkt",
            "symbol": SYMBOL_FUTS_LC,
            "side": side,
            "size": size_btc,
        })
        fill_price = float(ord.get("price", current_price))
        
        # Place stop loss
        place_stop_loss(api, side, size_btc, fill_price)
    
    # Record trade
    trade_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal": signal,
        "side": side,
        "size_btc": size_btc,
        "fill_price": fill_price,
        "portfolio_value": collateral,
        "today_prediction": today_pred,
        "yesterday_prediction": yesterday_pred,
        "yesterday_actual": yesterday_actual,
    }
    
    state["trades"].append(trade_record)
    state["predictions"].append({
        "date": df.index[-1].isoformat(),
        "predicted": today_pred,
        "actual": None,  # Will be filled next day
    })
    
    # Update yesterday's actual if exists
    if len(state["predictions"]) > 1:
        state["predictions"][-2]["actual"] = yesterday_actual
    
    # Calculate performance
    if state["starting_capital"]:
        total_return = (collateral - state["starting_capital"]) / state["starting_capital"] * 100
        state["performance"] = {
            "current_value": collateral,
            "starting_capital": state["starting_capital"],
            "total_return_pct": total_return,
            "total_trades": len(state["trades"]),
        }
    
    save_state(state)
    log.info(f"Trade executed and logged. Portfolio: ${collateral:.2f}")


def wait_until_00_01_utc():
    """Wait until 00:01 UTC for daily execution"""
    now = datetime.now(timezone.utc)
    next_run = now.replace(hour=0, minute=1, second=0, microsecond=0)
    if now >= next_run:
        next_run += timedelta(days=1)
    wait_sec = (next_run - now).total_seconds()
    log.info("Next run at 00:01 UTC (%s), sleeping %.0f s", next_run.strftime("%Y-%m-%d"), wait_sec)
    time.sleep(wait_sec)


def main():
    api_key = os.getenv("KRAKEN_API_KEY")
    api_sec = os.getenv("KRAKEN_API_SECRET")
    if not api_key or not api_sec:
        log.error("Env vars KRAKEN_API_KEY / KRAKEN_API_SECRET missing")
        sys.exit(1)

    api = kf.KrakenFuturesApi(api_key, api_sec)

    # Fetch training data
    log.info("Fetching Binance historical data for training...")
    df_train = binance_ohlc.get_ohlc_for_training(
        symbol=SYMBOL_OHLC_BINANCE,
        interval=INTERVAL_BINANCE
    )
    log.info(f"Training on {len(df_train)} days of Binance data")
    
    # Fetch on-chain metrics
    log.info("Fetching on-chain metrics...")
    onchain_df = fetch_onchain_metrics()
    
    # Train model
    log.info("Training LSTM model...")
    model = LSTMModel()
    model.fit(df_train, onchain_df)
    
    # Save model info to state
    state = load_state()
    state["model_info"] = {
        "train_mse": model.train_mse,
        "last_trained": model.last_trained,
        "lookback": LOOKBACK,
        "leverage": LEV,
    }
    save_state(state)
    
    # Run smoke test
    log.info("Running smoke test...")
    smoke_test(api)
    
    # Update state with current position
    update_state_with_current_position(api)

    if RUN_TRADE_NOW:
        log.info("RUN_TRADE_NOW=true – executing trade now")
        try:
            daily_trade(api, model, onchain_df)
        except Exception as exc:
            log.exception("Immediate trade failed: %s", exc)

    log.info("Starting web dashboard on port %s", os.getenv("PORT", 8080))
    subprocess.Popen([sys.executable, "web_state.py"])

    while True:
        wait_until_00_01_utc()
        try:
            daily_trade(api, model, onchain_df)
        except KeyboardInterrupt:
            log.info("Interrupted")
            break
        except Exception as exc:
            log.exception("Daily trade failed: %s", exc)


if __name__ == "__main__":
    main()
