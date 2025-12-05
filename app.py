import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import http.server
import socketserver
import os
import threading
import time
import logging

# ==========================================
# 1. Advanced Data Generation (Regimes)
# ==========================================
def generate_complex_data(n_days=2000, start_price=100, volatility=0.02):
    """
    Generates data with distinct Bull, Bear, and Sideways regimes.
    This forces the agent to actually learn switching behavior.
    """
    np.random.seed(42)
    
    # Create regimes
    # 0: Bull, 1: Bear, 2: Sideways
    regime_len = n_days // 4
    regimes = [0] * regime_len + [2] * regime_len + [1] * regime_len + [0] * regime_len
    
    # If n_days doesn't divide perfectly, fill the rest
    if len(regimes) < n_days:
        regimes.extend([0] * (n_days - len(regimes)))
        
    price = start_price
    data = []
    
    for i in range(n_days):
        regime = regimes[i]
        
        # Base return noise
        noise = np.random.normal(0, volatility)
        
        # Trend component based on regime
        if regime == 0:   # Bull
            trend = 0.0015 # +0.15% daily drift
        elif regime == 1: # Bear
            trend = -0.0015 # -0.15% daily drift
        else:             # Sideways
            trend = 0.0
            
        ret = trend + noise
        price = price * (1 + ret)
        
        # OHLC Simulation
        high = price * (1 + abs(np.random.normal(0, volatility/2)))
        low = price * (1 - abs(np.random.normal(0, volatility/2)))
        open_p = price * (1 + np.random.normal(0, volatility/4))
        
        data.append([open_p, high, low, price])
        
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
    return df

# ==========================================
# 2. Technical Indicators (Feature Engineering)
# ==========================================
def add_indicators(df):
    """
    Manually calculate RSI, MACD, and SMA to avoid external TA-Lib dependency.
    """
    df = df.copy()
    
    # 1. Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Fill NaN from calculations
    df.fillna(0, inplace=True)
    return df

# ==========================================
# 3. Enhanced Trading Environment
# ==========================================
class AdvancedTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(AdvancedTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        
        # Actions: Continuous [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        
        # Observations:
        # 1. Returns (Last 5 days)
        # 2. RSI / 100 (Normalized)
        # 3. (Price / SMA_50) - 1 (Trend distance)
        # 4. MACD Histogram (Normalized approx)
        # 5. Current Position Ratio
        # Shape = 5 + 4 = 9 inputs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.max_steps = len(df) - 1
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 60 # Start after enough data for indicators
        self.balance = self.initial_balance
        self.position = 0 # Units
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.history = {'net_worth': []}
        return self._next_observation(), {}

    def _next_observation(self):
        # Window of data
        current_idx = self.current_step
        
        # 1. Returns history (last 5)
        closes = self.df['Close'].values[current_idx-5:current_idx+1]
        returns = np.diff(np.log(closes)) # Length 5
        
        # 2. Technical Indicators (Pre-calculated)
        rsi = self.df.iloc[current_idx]['RSI'] / 100.0 # Normalize 0-1
        
        sma_ratio = (self.df.iloc[current_idx]['Close'] / self.df.iloc[current_idx]['SMA_50']) - 1.0
        
        macd_hist = self.df.iloc[current_idx]['MACD'] - self.df.iloc[current_idx]['Signal_Line']
        # Rough normalization for MACD (assuming price ~100)
        macd_norm = macd_hist 
        
        # 3. Position State
        pos_ratio = 0
        if self.net_worth > 0:
            pos_ratio = (self.position * self.df.iloc[current_idx]['Close']) / self.net_worth
            
        obs = np.concatenate((
            returns, 
            [rsi, sma_ratio, macd_norm, pos_ratio]
        ))
        
        return np.nan_to_num(obs).astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        prev_price = self.df.iloc[self.current_step - 1]['Close']
        
        # --- EXECUTION ---
        target_ratio = float(action[0])
        target_ratio = np.clip(target_ratio, -1, 1)
        
        # Calculate value
        current_val = self.balance + (self.position * current_price)
        
        # Transaction Cost (0.1% per trade to discourage infinite switching)
        current_exposure = self.position * current_price
        target_exposure = current_val * target_ratio
        trade_amount = abs(target_exposure - current_exposure)
        cost = trade_amount * 0.001
        
        # Update Balances
        # To simplify: We deduct cost from 'current_val' then rebalance
        current_val -= cost
        
        units_needed = (current_val * target_ratio) / current_price
        self.position = units_needed
        self.balance = current_val - (self.position * current_price)
        
        # --- STATE UPDATE ---
        self.net_worth = self.balance + (self.position * current_price)
        self.history['net_worth'].append(self.net_worth)
        
        # --- REWARD ENGINEERING ---
        # Calculate Strategy Return
        strategy_ret = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        
        # Calculate Benchmark Return (Buy and Hold)
        benchmark_ret = (current_price - prev_price) / prev_price
        
        # Reward = Alpha (Excess Return) + Small penalty for staying flat?
        # If we beat the market, positive reward. If we trail it, negative.
        reward = (strategy_ret - benchmark_ret) * 100
        
        # Small penalty for extreme leverage changes to encourage smoothness? 
        # (Optional, keeping simple for now)

        self.prev_net_worth = self.net_worth
        self.current_step += 1
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        if self.net_worth < self.initial_balance * 0.1:
            terminated = True
            reward = -100 # Die

        return self._next_observation(), reward, terminated, truncated, {}

# ==========================================
# 4. Web & Training Utils
# ==========================================
def start_web_server(port=8080):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>RL Advanced Trading</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f4f4f9; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .stats { margin-top: 20px; padding: 15px; background: #e8f4f8; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Advanced RL Agent Results</h1>
        <p>This agent uses RSI, MACD, and Moving Averages to trade across Bull, Bear, and Sideways markets.</p>
        <img src="rl_trading_results_v2.png" alt="Trading Results">
        <div class="stats">
            <p><strong>Green Shade:</strong> Bull Market | <strong>Red Shade:</strong> Bear Market | <strong>Grey:</strong> Sideways</p>
        </div>
    </div>
</body>
</html>'''
    with open('index.html', 'w') as f:
        f.write(html_content)
    
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Server started at http://localhost:{port}")
        httpd.serve_forever()

def run_simulation():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    
    logger.info("1. Generating Complex Market Data (Bull/Bear/Chop)...")
    df = generate_complex_data(n_days=2500, volatility=0.015)
    df = add_indicators(df) # Add RSI, MACD, SMA
    
    train_size = int(len(df) * 0.75)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    
    logger.info("2. Setting up Vectorized Environment with Normalization...")
    # VecNormalize is CRITICAL for PPO convergence on financial data
    # It automatically scales rewards and observations to mean 0, std 1
    env_maker = lambda: AdvancedTradingEnv(train_df)
    train_env = DummyVecEnv([env_maker])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    logger.info("3. Training PPO Agent (MlpPolicy)...")
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003, ent_coef=0.01)
    
    # Train
    model.learn(total_timesteps=80000)
    logger.info("Training complete.")
    
    # Save statistics for normalizing the test environment
    train_env.save("vec_normalize.pkl")

    # 4. Backtesting
    logger.info("4. Backtesting on Unseen Data...")
    
    # Load the test env but use the stats (mean/std) from training
    test_env = DummyVecEnv([lambda: AdvancedTradingEnv(test_df)])
    test_env = VecNormalize.load("vec_normalize.pkl", test_env)
    test_env.training = False # Do not update stats during test
    test_env.norm_reward = False # We want actual raw rewards for metric calculation
    
    obs = test_env.reset()
    done = [False]
    
    net_worth_history = []
    actions_history = []
    
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = test_env.step(action)
        
        # Access the underlying env to get actual net worth
        net_worth_history.append(test_env.envs[0].net_worth)
        actions_history.append(action[0][0])

    # 5. Visualization
    start_idx = 60 # Skip indicator warmup
    closes = test_df['Close'].values[start_idx : start_idx + len(net_worth_history)]
    
    # Benchmark
    initial_balance = 10000
    buy_hold_qty = initial_balance / closes[0]
    buy_hold_equity = closes * buy_hold_qty
    
    plt.figure(figsize=(12, 8))
    
    # Top: Performance
    plt.subplot(3, 1, 1)
    plt.plot(net_worth_history, label='RL Agent (Alpha-Hunter)', color='blue', linewidth=1.5)
    plt.plot(buy_hold_equity, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
    plt.title('Agent vs Market (Regime Switching)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Middle: Actions
    plt.subplot(3, 1, 2)
    plt.fill_between(range(len(actions_history)), actions_history, color='orange', alpha=0.3)
    plt.plot(actions_history, color='orange', linewidth=0.5)
    plt.axhline(0, color='black', linestyle='-')
    plt.title('Agent Exposure (-1 Short to +1 Long)')
    plt.ylabel('Position')
    
    # Bottom: Underlying Price
    plt.subplot(3, 1, 3)
    plt.plot(closes, color='black')
    plt.title('Market Price')
    
    plt.tight_layout()
    plt.savefig('rl_trading_results_v2.png')
    logger.info("Results saved.")

    # Start Server
    server_thread = threading.Thread(target=start_web_server, args=(8080,), daemon=True)
    server_thread.start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_simulation()
