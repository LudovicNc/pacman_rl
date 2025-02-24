import numpy as np
import matplotlib.pyplot as plt

class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.win_history = []
        self.episode_lengths = []
        
    def add_episode(self, total_reward, won, length):
        self.episode_rewards.append(total_reward)
        self.win_history.append(1 if won else 0)
        self.episode_lengths.append(length)
        
    def get_stats(self):
        avg_reward = np.mean(self.episode_rewards[-100:])
        win_rate = np.mean(self.win_history[-100:])
        avg_length = np.mean(self.episode_lengths[-100:])
        return {
            'average_reward': avg_reward,
            'win_rate': win_rate,
            'average_length': avg_length
        }
        
    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(132)
        plt.plot(self.win_history)
        plt.title('Win Rate')
        plt.xlabel('Episode')
        plt.ylabel('Won (1) / Lost (0)')
        
        plt.subplot(133)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.show() 