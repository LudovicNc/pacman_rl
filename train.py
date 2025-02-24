from environment.pacman_env import PacmanEnv
from agents.q_learning import QLearningAgent
from evaluation.plot_results import PacmanMetrics
import numpy as np
import time
import os

def train(episodes=5000, render_delay=0.1):
    env = PacmanEnv(render_mode=None)
    state_size = (15, 15, 3)
    action_size = 4
    
    agent = QLearningAgent(state_size, action_size)
    metrics = PacmanMetrics(window_size=100)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        ghost_captures = 0
        power_pellets = 0
        
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Track metrics
            if reward == 100:  # Ghost capture reward
                ghost_captures += 1
            elif reward == 20:  # Power pellet reward
                power_pellets += 1
            
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                won = len(env.food_positions) == 0
                metrics.add_episode(total_reward, steps, won, 
                                  ghost_captures, power_pellets)
                
                # Print progress
                print(f"Episode: {episode + 1}")
                print(f"Total Reward: {total_reward}")
                print(f"Steps: {steps}")
                print(f"Ghost Captures: {ghost_captures}")
                print(f"Power Pellets: {power_pellets}")
                
                # Plot every 100 episodes
                if (episode + 1) % 100 == 0:
                    metrics.plot_metrics(save_path=f"plots/episode_{episode+1}.png")
                    metrics.save_metrics(f"metrics/episode_{episode+1}.json")
                break
    
    return metrics

if __name__ == "__main__":
    # Create directories for saving results
    os.makedirs("plots", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    # Adjust these parameters:
    metrics = train(
        episodes=5000,
        render_delay=0.001  # 10x faster
    )
    
    # Save final results
    metrics.plot_metrics(save_path="plots/final_results.png")
    metrics.save_metrics("metrics/final_results.json") 