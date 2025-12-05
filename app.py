import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import http.server
import socketserver
import os
import threading
import time
import logging

# ==========================================
# 1. Synthetic Data Generation
# ==========================================
def generate_price_data(n_days=1000, start_price=100, volatility=0.02):
    """
    Generates synthetic daily OHLC data with a sine wave trend + random noise.
    """
    np.random.seed(42)
    
    # Generate a trend (sine wave + linear trend)
    x = np.linspace(0, 50, n_days)
    trend = 10 * np.sin(x) + x * 0.5
    
    # Random walk component
    returns = np.random.normal(0, volatility, n_days)
    price_curve = start_price * (1 + np.cumsum(returns)) + trend
    
    data = []
    for i in range(n_days):
        close_price = price_curve[i]
        # Simulate High/Low/Open based on Close
        daily_vol = close_price * volatility
        high_price = close_price + abs(np.random.normal(0, daily_vol/2))
        low_price = close_price - abs(np.random.normal(0, daily_vol/2))
        open_price = close_price + np.random.normal(0, daily_vol/4)
        
        # Ensure logical consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        data.append([open_price, high_price, low_price, close_price])
        
    df = pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Close'])
    return df

# ==========================================
# 2. Custom Trading Environment
# ==========================================
class TradingEnv(gym.Env):
    """
    A custom trading environment for RL.
    
    Actions:
        Box(-1, 1): 
        -1 = Full Short
         0 = Flat
         1 = Full Long
         
    Stop Loss:
        Triggers if High/Low exceeds p% from entry price.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stop_loss_pct=0.02, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.stop_loss_pct = stop_loss_pct
        self.initial_balance = initial_balance
        
        # Action space: Continuous value between -1 and 1
        # We interpret this as the target position ratio (100% long to 100% short)
        self.action_space = spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        
        # Observation space: 
        # [Recent Returns, Volatility, Current Position, Unrealized PnL %]
        # Shape is (6,) for a simple window of 3 returns + 3 state vars
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = len(df) - 1
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 10 # Start at 10 to allow for lag features
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0 # Current units held (can be float, but usually integer in real life. We use float for simplicity)
        self.entry_price = 0
        self.max_net_worth = self.initial_balance
        self.history = {'net_worth': []}
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Get data window
        frame = self.df.iloc[self.current_step - 5 : self.current_step + 1]
        
        # 1. Log Returns of the last 3 days
        close_prices = frame['Close'].values
        returns = np.diff(np.log(close_prices))[-3:]
        
        # 2. Volatility (std dev of last 5 prices normalized)
        vol = np.std(close_prices) / np.mean(close_prices)
        
        # 3. Current State
        pos_norm = 0
        if self.entry_price > 0:
            # Normalize Unrealized PnL roughly
            unrealized_pnl = (close_prices[-1] - self.entry_price) / self.entry_price
            if self.position < 0: unrealized_pnl *= -1
        else:
            unrealized_pnl = 0
            
        # Normalize current position status (-1 to 1)
        # We estimate max position size based on balance for normalization
        current_pos_ratio = 0
        if self.net_worth > 0:
            current_pos_ratio = (self.position * close_prices[-1]) / self.net_worth

        obs = np.concatenate((returns, [vol, current_pos_ratio, unrealized_pnl]))
        
        # Handle cases where data might be missing or nan (though synthetic data is clean)
        return np.nan_to_num(obs, nan=0.0).astype(np.float32)

    def step(self, action):
        # Current Market Data
        current_price = self.df.iloc[self.current_step]['Close']
        high_price = self.df.iloc[self.current_step]['High']
        low_price = self.df.iloc[self.current_step]['Low']
        
        # 1. Check Stop Loss on EXISTING position before executing new action
        sl_triggered = False
        
        if self.position != 0:
            if self.position > 0: # Long
                sl_price = self.entry_price * (1 - self.stop_loss_pct)
                if low_price <= sl_price:
                    # SL Hit
                    self.balance += self.position * sl_price
                    self.position = 0
                    self.entry_price = 0
                    sl_triggered = True
            elif self.position < 0: # Short
                sl_price = self.entry_price * (1 + self.stop_loss_pct)
                if high_price >= sl_price:
                    # SL Hit
                    self.balance += self.position * sl_price
                    self.position = 0
                    self.entry_price = 0
                    sl_triggered = True

        # 2. Execute Action (if SL didn't trigger this step, or if we want to re-enter)
        # The action represents the TARGET percentage of portfolio to invest
        # action is array like [-0.5], extract float
        target_ratio = float(action[0]) 
        
        # Clip action
        target_ratio = np.clip(target_ratio, -1, 1)
        
        if not sl_triggered:
            # Calculate target value
            current_val = self.balance + (self.position * current_price)
            target_exposure = current_val * target_ratio
            
            # Units needed to reach target
            units_needed = target_exposure / current_price
            
            # Simple transaction cost can be added here (e.g., 0.1%)
            # We skip explicit transaction costs for this simple example, 
            # but usually, you subtract cost from balance.
            
            # Update Position
            if abs(units_needed - self.position) > 0.01: # Only trade if significant change
                
                # If flipping from Long to Short or vice versa, reset entry price
                if np.sign(units_needed) != np.sign(self.position) and abs(units_needed) > 0:
                    self.entry_price = current_price
                elif self.position == 0 and abs(units_needed) > 0:
                    self.entry_price = current_price
                    
                # Realize PnL on the portion sold/bought (Simulated via balance update)
                # Mathematical simplification: We just set new position and update cash
                # Cash = Total Value - (New Position * Price)
                self.position = units_needed
                self.balance = current_val - (self.position * current_price)

        # 3. Calculate New Net Worth
        self.net_worth = self.balance + (self.position * current_price)
        self.history['net_worth'].append(self.net_worth)
        
        # 4. Calculate Reward
        # Reward is percentage change in Net Worth (to be scale invariant)
        # Can also add penalty for volatility or holding too long
        prev_net_worth = self.history['net_worth'][-2] if len(self.history['net_worth']) > 1 else self.initial_balance
        reward = (self.net_worth - prev_net_worth) / prev_net_worth * 100
        
        if sl_triggered:
            reward -= 0.5 # Penalty for hitting stop loss

        # 5. Advance Step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Check for bankruptcy
        if self.net_worth < self.initial_balance * 0.1:
            terminated = True
            reward = -10 # Heavy penalty for blowing up account

        return self._next_observation(), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Pos: {self.position:.2f}')

# ==========================================
# 3. Training and Evaluation
# ==========================================



def start_web_server(port=8080):
    """Start a simple HTTP server to serve the results PNG."""
    # Change to current directory to serve files from here
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Create a simple HTML page that displays the image
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>RL Trading Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>RL Trading Agent Results</h1>
        <p>Performance comparison between RL Agent and Buy & Hold strategy.</p>
        <img src="rl_trading_results.png" alt="Trading Results">
        <p>Server running on port {port}. Image generated from simulation.</p>
    </div>
</body>
</html>'''
    
    # Write the HTML file
    with open('index.html', 'w') as f:
        f.write(html_content.format(port=port))
    
    # Start the HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Web server started on port {port}")
        print(f"Open http://localhost:{port} in your browser to view results")
        httpd.serve_forever()

def run_simulation():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # 1. Generate Data
    logger.info("Generating synthetic price data...")
    df = generate_price_data(n_days=1500, start_price=100, volatility=0.015)
    logger.info(f"Generated {len(df)} days of price data")
    
    # Split into train and test
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    
    # 2. Setup Environments
    # We use DummyVecEnv for SB3 compatibility
    train_env = DummyVecEnv([lambda: TradingEnv(train_df, stop_loss_pct=0.03)])
    test_env = TradingEnv(test_df, stop_loss_pct=0.03)

    # 3. Initialize RL Model (PPO)
    # MLPPolicy is suitable for vector inputs (non-image)
    logger.info("Training PPO Agent...")
    model = PPO("MlpPolicy", train_env, verbose=0, learning_rate=0.0003)
    
    # 4. Train with progress logging
    total_timesteps = 50000
    logger.info(f"Starting training for {total_timesteps} timesteps")
    
    # Custom callback for progress logging
    from stable_baselines3.common.callbacks import BaseCallback
    
    class ProgressCallback(BaseCallback):
        def __init__(self, check_freq=1, verbose=0):
            super(ProgressCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.start_time = None
            self.last_log_time = None
            self.log_interval = 100  # Log every 100 steps to avoid overwhelming output
            
        def _on_training_start(self) -> None:
            self.start_time = time.time()
            self.last_log_time = self.start_time
            logger.info(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Total timesteps: {self.num_timesteps}")
            
        def _on_step(self) -> bool:
            current_time = time.time()
            
            # Log every log_interval steps or if it's the first/last step
            if self.n_calls % self.log_interval == 0 or self.n_calls == 1 or self.n_calls == self.num_timesteps:
                elapsed_time = current_time - self.start_time
                progress_percent = (self.n_calls / self.num_timesteps) * 100
                
                # Calculate estimated time remaining
                if self.n_calls > 0:
                    time_per_step = elapsed_time / self.n_calls
                    remaining_steps = self.num_timesteps - self.n_calls
                    estimated_remaining = time_per_step * remaining_steps
                    
                    # Format time remaining
                    if estimated_remaining < 60:
                        time_str = f"{estimated_remaining:.0f}s"
                    elif estimated_remaining < 3600:
                        time_str = f"{estimated_remaining/60:.1f}m"
                    else:
                        time_str = f"{estimated_remaining/3600:.1f}h"
                else:
                    time_str = "calculating..."
                
                # Calculate steps per second
                steps_per_sec = self.n_calls / elapsed_time if elapsed_time > 0 else 0
                
                # Create progress bar visualization
                bar_length = 30
                filled_length = int(bar_length * self.n_calls // self.num_timesteps)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                
                logger.info(f"Step {self.n_calls}/{self.num_timesteps} [{bar}] {progress_percent:.1f}% | ETA: {time_str} | Speed: {steps_per_sec:.1f} steps/sec")
                
                # Log milestone every 10%
                if self.n_calls % (self.num_timesteps // 10) == 0 and self.n_calls > 0:
                    logger.info(f"Training milestone: {progress_percent:.0f}% complete ({self.n_calls}/{self.num_timesteps} steps)")
                
                self.last_log_time = current_time
            
            return True
        
        def _on_training_end(self) -> None:
            total_time = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / total_time if total_time > 0 else 0
            logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info(f"Average speed: {steps_per_sec:.2f} steps/second")
            logger.info(f"Total steps trained: {self.num_timesteps}")
    
    callback = ProgressCallback(check_freq=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    logger.info("Training complete.")

    # 5. Backtest on Unseen Data
    logger.info("Backtesting on Test Data...")
    obs, _ = test_env.reset()
    done = False
    
    net_worth_history = []
    actions_history = []
    
    total_steps = len(test_df) - 10  # Environment starts at index 10
    step_count = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        
        net_worth_history.append(test_env.net_worth)
        actions_history.append(action[0])
        
        step_count += 1
        if step_count % 100 == 0:
            logger.info(f"Backtesting progress: {step_count}/{total_steps} steps ({step_count/total_steps*100:.1f}%)")
    
    logger.info(f"Backtesting completed. Total steps: {step_count}")

    # 6. Compare with Buy and Hold
    initial_price = test_df.iloc[10]['Close'] # Environment starts at index 10
    final_price_dataset = test_df['Close'].values[10:10+len(net_worth_history)]
    
    # Normalize Buy & Hold to start at same amount as Agent
    buy_hold_performance = (final_price_dataset / initial_price) * 10000
    
    # 7. Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot Net Worth
    plt.subplot(2, 1, 1)
    plt.plot(net_worth_history, label='RL Agent (PPO)', color='blue')
    plt.plot(buy_hold_performance, label='Buy & Hold', color='gray', linestyle='--')
    plt.title('Agent Performance vs Buy & Hold')
    plt.ylabel('Net Worth ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Actions
    plt.subplot(2, 1, 2)
    plt.bar(range(len(actions_history)), actions_history, color='orange', alpha=0.5, width=1.0)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Agent Actions (Position Sizing)')
    plt.ylabel('Action (-1 Short to 1 Long)')
    plt.xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig('rl_trading_results.png')
    logger.info("Results saved to 'rl_trading_results.png'")
    # plt.show() # Uncomment if running locally with GUI
    
    # Start web server in a separate thread
    logger.info("Starting web server on port 8080...")
    server_thread = threading.Thread(target=start_web_server, args=(8080,), daemon=True)
    server_thread.start()
    
    # Keep the main thread alive to serve requests
    logger.info("Web server is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down web server...")

if __name__ == "__main__":
    run_simulation()
