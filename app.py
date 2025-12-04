import numpy as np
import pandas as pd
import random
from collections import defaultdict
import matplotlib.pyplot as plt

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
        
    def generate_sine_data(self):
        """Generate synthetic sine-shaped price data."""
        t = np.linspace(0, 10 * np.pi, self.data_points)
        price = 100 + 20 * np.sin(t) + np.random.normal(0, 2, self.data_points)
        return price
    
    def reset(self):
        """Reset the environment to initial state."""
        self.prices = self.generate_sine_data()
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0  # Current position multiplier (-1 to 1)
        self.returns = []
        self.total_return = 0.0
        self.done = False
        
        # Initial state: normalized price, position, and balance
        state = self._get_state()
        return state
    
    def _get_state(self):
        """Get current state representation."""
        # Normalize price to [0, 1] range based on min/max of entire series
        price_min = np.min(self.prices)
        price_max = np.max(self.prices)
        normalized_price = (self.prices[self.current_step] - price_min) / (price_max - price_min)
        
        # Normalize balance
        normalized_balance = self.balance / (self.initial_balance * 2)
        
        # Position is already in [-1, 1] range
        normalized_position = (self.position + 1) / 2  # Map to [0, 1]
        
        return (normalized_price, normalized_position, normalized_balance)
    
    def step(self, action):
        """Take a step in the environment."""
        if self.done:
            raise ValueError("Episode has already terminated")
        
        # Get current and next price
        current_price = self.prices[self.current_step]
        
        # Update position based on action
        self.position = self.position_multipliers[action]
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.prices) - 1:
            self.done = True
            next_price = current_price  # No price change on last step
        else:
            next_price = self.prices[self.current_step]
        
        # Calculate return based on position
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
        self.state_bins = 10
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
    """Plot trading results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Price and positions
    axes[0].plot(prices, label='Price', color='blue', alpha=0.7)
    axes[0].set_ylabel('Price')
    axes[0].set_title(f'Trading Results - Total Return: {total_return:.2f}')
    axes[0].legend(loc='upper left')
    
    # Create secondary y-axis for positions
    ax2 = axes[0].twinx()
    ax2.plot(positions, label='Position', color='red', alpha=0.7, linewidth=2)
    ax2.set_ylabel('Position')
    ax2.legend(loc='upper right')
    
    # Plot 2: Balance over time
    axes[1].plot(balances, label='Balance', color='green')
    axes[1].set_ylabel('Balance')
    axes[1].set_xlabel('Step')
    axes[1].legend()
    
    # Plot 3: Actions taken
    action_names = ['flat', 'long', 'short', 'half_long', 'half_short']
    axes[2].bar(range(len(actions_taken)), actions_taken, alpha=0.7)
    axes[2].set_ylabel('Action')
    axes[2].set_xlabel('Step')
    axes[2].set_yticks(range(5))
    axes[2].set_yticklabels(action_names)
    axes[2].set_title('Actions Taken')
    
    plt.tight_layout()
    plt.savefig('trading_results.png')
    plt.show()


def main():
    """Main function to run the RL trading system."""
    print("=== Reinforcement Learning Trading System ===")
    print("Generating synthetic sine-shaped data with 1000 points...")
    
    # Create environment
    env = TradingEnvironment(data_points=1000, initial_balance=10000)
    
    # Create and train agent
    print("\nTraining Q-learning agent...")
    agent = QLearningAgent(env)
    episode_returns = agent.train(episodes=50)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    actions_taken, prices, positions, balances, total_return = agent.test()
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Final exploration rate: {agent.exploration_rate:.4f}")
    print(f"Number of states in Q-table: {len(agent.q_table)}")
    print(f"Final test total return: {total_return:.2f}")
    print(f"Final balance: {balances[-1]:.2f}")
    
    # Action distribution
    action_names = ['flat', 'long', 'short', 'half_long', 'half_short']
    action_counts = {name: 0 for name in action_names}
    for action in actions_taken:
        action_counts[action_names[action]] += 1
    
    print("\nAction distribution:")
    for action_name, count in action_counts.items():
        percentage = (count / len(actions_taken)) * 100
        print(f"  {action_name}: {count} times ({percentage:.1f}%)")
    
    # Plot results
    print("\nGenerating visualization...")
    plot_results(prices, positions, balances, actions_taken, total_return)
    
    # Save Q-table for future use
    print("\nSaving Q-table to file...")
    q_table_dict = {str(k): v.tolist() for k, v in agent.q_table.items()}
    import json
    with open('q_table.json', 'w') as f:
        json.dump(q_table_dict, f)
    
    print("\n=== Program Complete ===")
    print("Results saved to 'trading_results.png' and 'q_table.json'")


if __name__ == "__main__":
    main()