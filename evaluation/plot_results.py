import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import deque
import json
import os

def load_rewards(filename):
    """Load reward history from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)

def plot_learning_curve(q_rewards, dqn_rewards, ppo_rewards, save_path="plots/learning_curve.png"):
    """Plot the learning curves of different RL agents."""
    plt.figure(figsize=(10,5))
    
    if q_rewards:
        plt.plot(np.mean(q_rewards, axis=1), label="Q-Learning", linestyle='--')
    if dqn_rewards:
        plt.plot(np.mean(dqn_rewards, axis=1), label="DQN", linestyle='-.')
    if ppo_rewards:
        plt.plot(np.mean(ppo_rewards, axis=1), label="PPO", linestyle='-')
    
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title("Agent Learning Curve Comparison")
    plt.grid()
    plt.savefig(save_path)
    plt.show()

class PacmanMetrics:
    def __init__(self, window_size=100):
        self.rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.ghost_captures = []
        self.power_pellets_collected = []
        self.window_size = window_size
        
        # Moving averages
        self.avg_rewards = deque(maxlen=window_size)
        self.avg_lengths = deque(maxlen=window_size)
        self.avg_wins = deque(maxlen=window_size)
    
    def add_episode(self, reward, length, won, ghost_captures=0, power_pellets=0):
        """Record metrics for an episode"""
        self.rewards.append(reward)
        self.episode_lengths.append(length)
        self.win_rates.append(1 if won else 0)
        self.ghost_captures.append(ghost_captures)
        self.power_pellets_collected.append(power_pellets)
        
        # Update moving averages
        self.avg_rewards.append(reward)
        self.avg_lengths.append(length)
        self.avg_wins.append(1 if won else 0)
    
    def plot_metrics(self, save_path=None):
        """Plot all metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pac-Man Training Metrics', fontsize=16)
        
        # Plot rewards
        axes[0,0].plot(self.rewards, alpha=0.3, label='Raw')
        axes[0,0].plot(np.convolve(self.rewards, 
                                  np.ones(self.window_size)/self.window_size, 
                                  mode='valid'),
                      label=f'{self.window_size}-ep Average')
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].legend()
        
        # Plot episode lengths
        axes[0,1].plot(self.episode_lengths, alpha=0.3, label='Raw')
        axes[0,1].plot(np.convolve(self.episode_lengths,
                                  np.ones(self.window_size)/self.window_size,
                                  mode='valid'),
                      label=f'{self.window_size}-ep Average')
        axes[0,1].set_title('Episode Lengths')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Steps')
        axes[0,1].legend()
        
        # Plot win rate
        axes[1,0].plot(np.convolve(self.win_rates,
                                  np.ones(self.window_size)/self.window_size,
                                  mode='valid'),
                      label='Win Rate')
        axes[1,0].set_title(f'Win Rate ({self.window_size}-episode window)')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Win Rate')
        axes[1,0].set_ylim(0, 1)
        
        # Plot ghost captures and power pellets
        axes[1,1].plot(self.ghost_captures, label='Ghost Captures', alpha=0.5)
        axes[1,1].plot(self.power_pellets_collected, label='Power Pellets', alpha=0.5)
        axes[1,1].set_title('Ghost Captures & Power Pellets')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_metrics(self, filepath):
        """Save metrics to JSON file"""
        metrics = {
            'rewards': self.rewards,
            'episode_lengths': self.episode_lengths,
            'win_rates': self.win_rates,
            'ghost_captures': self.ghost_captures,
            'power_pellets': self.power_pellets_collected
        }
        with open(filepath, 'w') as f:
            json.dump(metrics, f)

def main():
    """Load reward histories and plot results."""
    q_rewards = load_rewards("logs/q_learning_rewards.pkl") if "logs/q_learning_rewards.pkl" else []
    dqn_rewards = load_rewards("logs/dqn_rewards.pkl") if "logs/dqn_rewards.pkl" else []
    ppo_rewards = load_rewards("logs/ppo_rewards.pkl") if "logs/ppo_rewards.pkl" else []
    
    plot_learning_curve(q_rewards, dqn_rewards, ppo_rewards)

if __name__ == "__main__":
    main()
