import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import http.server
import socketserver
import os
import threading
import time
import logging

# ==========================================
# 1. Data Generation (Sharper Regimes)
# ==========================================
def generate_complex_data(n_days=3000, start_price=100, volatility=0.015):
    np.random.seed(45) # New seed
    
    # 0: Bull, 1: Bear, 2: Sideways
    regime_len = n_days // 5
    # Pattern: Bull -> Sideways -> Bull -> Bear -> Bull
    regimes = [0]*regime_len + [2]*regime_len + [0]*regime_len + [1]*regime_len + [0]*regime_len
    
    if len(regimes) < n_days:
        regimes.extend([0] * (n_days - len(regimes)))
        
    price = start_price
    data = []
    
    for i in range(n_days):
        regime = regimes[i]
        noise = np.random.normal(0, volatility)
        
        if regime == 0:   # Bull
            trend = 0.001 # Slow steady climb
        elif regime == 1: # Bear
            trend = -0.002 # Sharp drops (fear is faster than greed)
        else:             # Sideways
            trend = 0.0
            
        ret = trend + noise
        price = price * (1 + ret)
        
        # Add some spikes
        if np.random.rand() < 0.01:
            price *= (1 + np.random.normal(0, 0.05))

        # OHLC
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_p = price * (1 + np.random.normal(0, 0.002))
        
        data.append([open_p, high, low, price])
        
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
    return df

# ==========================================
# 2. Indicators
# ==========================================
def add_indicators(df):
    df = df.copy()
    # SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_Dist'] = (df['Close'] / df['SMA_20']) - 1.0
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))) / 100.0 # Normalize 0-1
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal
    
    # Volatility (Bollinger Band width approx)
    df['Vol'] = df['Close'].rolling(20).std() / df['SMA_20']
    
    df.fillna(0, inplace=True)
    return df

# ==========================================
# 3. Professional Windowed Environment
# ==========================================
class ProTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, window_size=30):
        super(ProTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        # Action: [-1 to 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation: Flattened window of features
        # Features: [LogReturn, RSI, MACD_Hist, SMA_Dist, Vol, Position] = 6 features
        self.n_features = 6
        self.obs_shape = (self.window_size * self.n_features,)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        self.max_steps = len(df) - 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size + 10
        self.balance = self.initial_balance
        self.position = 0.0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        
        self.history_net_worth = [self.initial_balance]
        
        return self._get_window_observation(), {}

    def _get_window_observation(self):
        # We grab the last 'window_size' rows of features
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        
        # Extract features
        closes = self.df['Close'].values[start-1:end]
        log_rets = np.diff(np.log(closes)) # Length = window_size
        
        rsi = self.df['RSI'].values[start:end]
        macd = self.df['MACD_Hist'].values[start:end]
        sma = self.df['SMA_Dist'].values[start:end]
        vol = self.df['Vol'].values[start:end]
        
        # Position is scalar, but we need it as a vector to stack or just scalar broadcast?
        # Better: Create a vector of current position repeated (agent needs to know it held this position during this window context)
        pos_arr = np.full(self.window_size, self.position)
        
        # Stack and Flatten
        # Shape: (Window_Size, Features)
        obs_matrix = np.column_stack([log_rets, rsi, macd, sma, vol, pos_arr])
        
        return obs_matrix.flatten().astype(np.float32)

    def step(self, action):
        # 1. Market Move
        current_price = self.df.iloc[self.current_step]['Close']
        prev_price = self.df.iloc[self.current_step - 1]['Close']
        
        # 2. Execute Action
        target_ratio = np.clip(action[0], -1, 1)
        
        # Portfolio Rebalancing Logic
        current_val = self.balance + (self.position * current_price)
        
        # Transaction Costs (Higher to prevent churning)
        # 0.2% fee
        current_exposure = self.position * current_price
        target_exposure = current_val * target_ratio
        trade_amt = abs(target_exposure - current_exposure)
        cost = trade_amt * 0.002 
        
        current_val -= cost
        self.balance = current_val - target_exposure
        self.position = target_exposure / current_price
        
        # 3. Update Stats
        self.net_worth = self.balance + (self.position * current_price)
        self.history_net_worth.append(self.net_worth)
        
        # 4. Reward Calculation
        # Calculate returns based on actual position
        step_return = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
        market_return = (current_price - prev_price) / prev_price
        
        # A) Alpha Reward
        alpha = step_return - market_return
        
        # B) Sortino-ish Penalty (Penalize downside heavily)
        risk_penalty = 0
        if step_return < 0:
            risk_penalty = abs(step_return) * 2.0 # Double pain for losing money
            
        reward = (alpha * 100) - (risk_penalty * 50)
        
        self.prev_net_worth = self.net_worth
        self.current_step += 1
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        if self.net_worth < self.initial_balance * 0.2: # 80% loss
            terminated = True
            reward = -100

        return self._get_window_observation(), reward, terminated, truncated, {}

# ==========================================
# 4. Training & Web
# ==========================================
def start_web_server(port=8080):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    content = """
    <html><body>
    <h1>Professional RL Agent Results</h1>
    <img src='rl_pro_results.png' style='max-width:100%'>
    </body></html>
    """
    with open('index.html', 'w') as f: f.write(content)
    with socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Server: http://localhost:{port}")
        httpd.serve_forever()

def run_simulation():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    
    logger.info("Generating Data...")
    df = generate_complex_data()
    df = add_indicators(df)
    
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    
    logger.info("Setup Environment...")
    env_maker = lambda: ProTradingEnv(train_df, window_size=30)
    train_env = DummyVecEnv([env_maker])
    # Clip rewards to stabilize training
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_reward=10.)
    
    logger.info("Training PPO (Deep Network)...")
    # Larger Network: [256, 256] 
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    
    model = PPO("MlpPolicy", train_env, verbose=0, 
                learning_rate=0.0001,  # Lower LR for stability
                batch_size=128,        # Larger batches
                policy_kwargs=policy_kwargs,
                ent_coef=0.005)        # Exploration
    
    # Progress Logger
    class SimpleLog(BaseCallback):
        def _on_step(self):
            if self.n_calls % 5000 == 0:
                logger.info(f"Step {self.n_calls}")
            return True

    model.learn(total_timesteps=20000, callback=SimpleLog())
    train_env.save("pro_vec_norm.pkl")
    
    logger.info("Backtesting...")
    test_env = DummyVecEnv([lambda: ProTradingEnv(test_df, window_size=30)])
    test_env = VecNormalize.load("pro_vec_norm.pkl", test_env)
    test_env.training = False
    test_env.norm_reward = False
    
    obs = test_env.reset()
    done = [False]
    nw_history = []
    actions = []
    
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = test_env.step(action)
        nw_history.append(test_env.envs[0].net_worth)
        actions.append(action[0][0])
        
    # Plotting
    closes = test_df['Close'].values[31 : 31 + len(nw_history)]
    bh_qty = 10000 / closes[0]
    bh_val = closes * bh_qty
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3,1,1)
    plt.plot(nw_history, label='Pro Agent', color='blue')
    plt.plot(bh_val, label='Buy & Hold', color='gray', linestyle='--')
    plt.title("Net Worth Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3,1,2)
    plt.plot(actions, color='orange', linewidth=0.8)
    plt.axhline(0, color='black')
    plt.title("Agent Positions (-1 to 1)")
    plt.ylabel("Exposure")
    
    plt.subplot(3,1,3)
    plt.plot(closes, color='black')
    plt.title("Market Price")
    
    plt.tight_layout()
    plt.savefig('rl_pro_results.png')
    logger.info("Done. Starting Server.")
    
    server = threading.Thread(target=start_web_server, args=(8080,), daemon=True)
    server.start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: pass

if __name__ == "__main__":
    run_simulation()
