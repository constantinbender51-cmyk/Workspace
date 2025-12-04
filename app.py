import numpy as np
import pandas as pd
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template_string, request

class TradingEnvironment:
    """
    A custom trading environment for reinforcement learning.
    The environment generates synthetic sine-shaped price data.
    """
    def __init__(self, data_points=1000, initial_balance=10000):
        self.data_points = data_points
        self.initial_balance = initial_balance
        self.reset()
        
        # Define action space: 0=flat, 1=long, 2=short, 3=half_long, 4=half_short
        self.action_space = [0, 1, 2, 3, 4]
        self.action_names = ['flat', 'long', 'short', 'half_long', 'half_short']
        
        # Position multipliers for each action
        self.position_multipliers = {
            0: 0.0,    # flat: no position
            1: 1.0,    # long: full long position
            2: -1.0,   # short: full short position
            3: 0.5,    # half_long: half long position
            4: -0.5    # half_short: half short position
        }
        
    def generate_price_data(self):
        """Generate synthetic sine-shaped price data."""
        t = np.linspace(0, 10 * np.pi, self.data_points)
        # Generate price series
        prices = 100 + 20 * np.sin(t) + np.random.normal(0, 2, self.data_points)
        
        return prices
    
    def reset(self):
        """Reset the environment to initial state."""
        self.prices = self.generate_price_data()
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Current position multiplier (-1 to 1)
        self.returns = []
        self.total_return = 0.0
        self.done = False
        
        # Initial state: lookback window of previous prices
        state = self._get_state()
        return state
    
    def _get_state(self):
        """Get current state representation using lookback window of 1 previous price."""
        # Use lookback window of 1 previous price (t-1)
        lookback = 1
        
        # Get indices for lookback window
        start_idx = max(0, self.current_step - lookback)
        end_idx = self.current_step  # Exclude current price at time t
        
        # Extract lookback window of prices
        if end_idx > start_idx:
            lookback_prices = self.prices[start_idx:end_idx]
            # Pad with zeros if we don't have enough history
            if len(lookback_prices) < lookback:
                lookback_prices = np.concatenate([
                    np.zeros(lookback - len(lookback_prices)),
                    lookback_prices
                ])
        else:
            # At the beginning, use zeros for missing history
            lookback_prices = np.zeros(lookback)
        
        # Normalize lookback prices to [0, 1] range based on min/max of entire price series
        price_min = np.min(self.prices)
        price_max = np.max(self.prices)
        normalized_lookback = (lookback_prices - price_min) / (price_max - price_min)
        
        # Normalize balance
        normalized_balance = self.balance / (self.initial_balance * 2)
        
        # Position is already in [-1, 1] range
        normalized_position = (self.position + 1) / 2  # Map to [0, 1]
        
        # Combine all features into a single state tuple
        state_features = tuple(normalized_lookback) + (normalized_position, normalized_balance)
        
        return state_features
    
    def step(self, action):
        """Take a step in the environment with exclusive actions."""
        if self.done:
            raise ValueError("Episode has already terminated")
        
        # Get current price (price at time t)
        current_price = self.prices[self.current_step]
        
        # Update position based on action (exclusive: position set directly from action)
        # Actions are exclusive per time step, so position is not cumulative
        self.position = self.position_multipliers[action]
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.prices) - 1:
            self.done = True
            next_price = current_price  # No price change on last step
        else:
            next_price = self.prices[self.current_step]
        
        # Calculate return based on position (using price change from t to t+1)
        # Position taken at current price (t), evaluated at next price (t+1)
        price_change = next_price - current_price
        position_return = self.position * price_change
        
        # Update balance (simplified - no transaction costs)
        self.balance += position_return
        
        # Track returns
        self.returns.append(position_return)
        self.total_return = np.sum(self.returns)
        
        # Calculate reward (could be return, Sharpe ratio, etc.)
        reward = position_return
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'price': current_price,
            'position': self.position,
            'balance': self.balance,
            'step_return': position_return,
            'total_return': self.total_return
        }
        
        return next_state, reward, self.done, info
    
    def render(self):
        """Print current environment status."""
        print(f"Step: {self.current_step}, Price: {self.prices[self.current_step]:.2f}, "
              f"Position: {self.position}, Balance: {self.balance:.2f}, "
              f"Total Return: {self.total_return:.2f}")


class QLearningAgent:
    """
    Q-learning agent for trading optimization.
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # Discretize state space for Q-table
        # State now has 4 dimensions: 2 lookback prices + position + balance
        self.state_bins = 5  # Reduce bins due to larger state space
        self.q_table = defaultdict(lambda: np.zeros(len(env.action_space)))
        
    def discretize_state(self, state):
        """Convert continuous state to discrete bins."""
        discretized = []
        for value in state:
            # Clip to [0, 1] and bin
            clipped = np.clip(value, 0, 1)
            bin_index = int(clipped * (self.state_bins - 1))
            discretized.append(bin_index)
        return tuple(discretized)
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        discretized_state = self.discretize_state(state)
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Explore: random action
            action = random.choice(self.env.action_space)
        else:
            # Exploit: best action from Q-table
            q_values = self.q_table[discretized_state]
            action = np.argmax(q_values)
        
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning algorithm."""
        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)
        
        # Current Q-value
        current_q = self.q_table[discretized_state][action]
        
        # Next best Q-value
        if done:
            next_q = 0
        else:
            next_q = np.max(self.q_table[discretized_next_state])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_q - current_q)
        self.q_table[discretized_state][action] = new_q
        
        # Decay exploration rate
        if done:
            self.exploration_rate = max(self.min_exploration, 
                                        self.exploration_rate * self.exploration_decay)
    
    def train(self, episodes=100):
        """Train the agent for multiple episodes."""
        episode_returns = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Choose and take action
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Learn from experience
                self.learn(state, action, reward, next_state, done)
                
                # Update for next iteration
                state = next_state
                total_reward += reward
            
            episode_returns.append(total_reward)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Total Return: {total_reward:.2f}, "
                      f"Exploration Rate: {self.exploration_rate:.3f}")
        
        return episode_returns
    
    def test(self):
        """Test the trained agent."""
        state = self.env.reset()
        done = False
        actions_taken = []
        prices = []
        positions = []
        balances = []
        
        while not done:
            # Choose action (no exploration during test)
            discretized_state = self.discretize_state(state)
            q_values = self.q_table[discretized_state]
            action = np.argmax(q_values)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Record data for visualization
            actions_taken.append(action)
            prices.append(info['price'])
            positions.append(info['position'])
            balances.append(info['balance'])
            
            # Update state
            state = next_state
        
        return actions_taken, prices, positions, balances, info['total_return']


def plot_results(prices, positions, balances, actions_taken, total_return):
    """Plot trading results and return base64 encoded image."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Price with background colors based on positions
    axes[0].plot(prices, label='Open Price', color='blue', alpha=0.7, linewidth=2)
    axes[0].set_ylabel('Open Price')
    axes[0].set_title(f'Trading Results - Total Return: {total_return:.2f}')
    axes[0].legend(loc='upper left')
    
    # Define background colors for different position types
    position_colors = {
        0.0: 'gray',    # flat
        1.0: 'green',   # long
        -1.0: 'red',    # short
        0.5: 'lightgreen',  # half_long
        -0.5: 'lightcoral'  # half_short
    }
    
    # Add background colors for each position
    for i in range(len(positions)):
        pos = positions[i]
        color = position_colors.get(pos, 'white')
        # Add colored rectangle for each time step
        axes[0].axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.3)
    
    # Create custom legend for position colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.3, label='Flat (0.0)'),
        Patch(facecolor='green', alpha=0.3, label='Long (1.0)'),
        Patch(facecolor='red', alpha=0.3, label='Short (-1.0)'),
        Patch(facecolor='lightgreen', alpha=0.3, label='Half Long (0.5)'),
        Patch(facecolor='lightcoral', alpha=0.3, label='Half Short (-0.5)')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize='small')
    
    # Plot 2: Balance over time
    axes[1].plot(balances, label='Balance', color='green')
    axes[1].set_ylabel('Balance')
    axes[1].set_xlabel('Step')
    axes[1].legend()
    
    # Plot 3: Actions taken with clear markers
    action_names = ['flat', 'long', 'short', 'half_long', 'half_short']
    action_colors = ['gray', 'green', 'red', 'lightgreen', 'lightcoral']
    
    # Plot each action type with different markers
    for action_idx in range(5):
        # Find all steps where this action was taken
        action_steps = [i for i, a in enumerate(actions_taken) if a == action_idx]
        if action_steps:
            # Plot points for this action
            axes[2].scatter(action_steps, [action_idx] * len(action_steps), 
                          color=action_colors[action_idx], s=50, 
                          label=action_names[action_idx], alpha=0.8)
    
    axes[2].set_ylabel('Action')
    axes[2].set_xlabel('Step')
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(action_names)
    axes[2].set_title('Actions Taken Over Time')
    axes[2].legend(loc='upper right', fontsize='small')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Encode to base64 for HTML display
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def run_training():
    """Run the RL training and return results."""
    print("=== Reinforcement Learning Trading System ===")
    print("Generating synthetic sine-shaped data with 2000 points...")
    
    # Create environment
    env = TradingEnvironment(data_points=2000, initial_balance=10000)
    
    # Create and train agent
    print("\nTraining Q-learning agent...")
    agent = QLearningAgent(env)
    episode_returns = agent.train(episodes=50)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    actions_taken, prices, positions, balances, total_return = agent.test()
    
    # Action distribution
    action_names = ['flat', 'long', 'short', 'half_long', 'half_short']
    action_counts = {name: 0 for name in action_names}
    for action in actions_taken:
        action_counts[action_names[action]] += 1
    
    # Generate visualization
    img_base64 = plot_results(prices, positions, balances, actions_taken, total_return)
    
    # Prepare summary data
    summary = {
        'total_return': total_return,
        'final_balance': balances[-1],
        'exploration_rate': agent.exploration_rate,
        'q_table_size': len(agent.q_table),
        'action_distribution': action_counts,
        'total_steps': len(actions_taken),
        'image': img_base64
    }
    
    return summary


# Flask web application
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>RL Trading System Visualization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
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
        .summary {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .summary-item {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .summary-item h3 {
            margin-top: 0;
            color: #555;
        }
        .visualization {
            text-align: center;
            margin-top: 20px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .action-distribution {
            margin-top: 20px;
        }
        .action-bar {
            display: flex;
            margin-bottom: 5px;
            align-items: center;
        }
        .action-name {
            width: 100px;
            font-weight: bold;
        }
        .action-bar-inner {
            height: 20px;
            background-color: #4CAF50;
            border-radius: 3px;
        }
        .action-count {
            margin-left: 10px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 0;
        }
        .btn:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Reinforcement Learning Trading System</h1>
        <p style="text-align: center;">Optimizing returns on synthetic sine-shaped data (2000 training examples)</p>
        
        <div style="text-align: center;">
            <form method="POST">
                <button type="submit" class="btn">Run New Training Session</button>
            </form>
        </div>
        
        {% if summary %}
        <div class="summary">
            <h2>Training Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>Total Return</h3>
                    <p style="font-size: 24px; font-weight: bold; color: {% if summary.total_return > 0 %}#4CAF50{% else %}#f44336{% endif %};">
                        {{ "%.2f"|format(summary.total_return) }}
                    </p>
                </div>
                <div class="summary-item">
                    <h3>Final Balance</h3>
                    <p style="font-size: 24px; font-weight: bold;">{{ "%.2f"|format(summary.final_balance) }}</p>
                </div>
                <div class="summary-item">
                    <h3>Exploration Rate</h3>
                    <p style="font-size: 24px; font-weight: bold;">{{ "%.4f"|format(summary.exploration_rate) }}</p>
                </div>
                <div class="summary-item">
                    <h3>Q-table Size</h3>
                    <p style="font-size: 24px; font-weight: bold;">{{ summary.q_table_size }} states</p>
                </div>
            </div>
            
            <div class="action-distribution">
                <h3>Action Distribution</h3>
                {% for action_name, count in summary.action_distribution.items() %}
                {% set percentage = (count / summary.total_steps * 100) %}
                <div class="action-bar">
                    <div class="action-name">{{ action_name }}</div>
                    <div class="action-bar-inner" style="width: {{ percentage }}%;"></div>
                    <div class="action-count">{{ count }} ({{ "%.1f"|format(percentage) }}%)</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="visualization">
            <h2>Trading Visualization</h2>
            <img src="data:image/png;base64,{{ summary.image }}" alt="Trading Results Visualization">
        </div>
        {% else %}
        <div style="text-align: center; padding: 40px;">
            <h2>Click the button above to run a training session</h2>
            <p>The system will train a Q-learning agent on synthetic sine-shaped data and display the results.</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = None
    
    if request.method == 'POST':
        # Run training when form is submitted
        summary = run_training()
    
    return render_template_string(HTML_TEMPLATE, summary=summary)


if __name__ == "__main__":
    print("Starting web server on port 8080...")
    print("Open your browser and navigate to http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=False)