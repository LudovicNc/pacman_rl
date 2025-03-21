from environment.pacman_env import PacmanEnv
from agents.q_learning import QLearningAgent
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def train(episodes=500, render_mode=None, render_delay=0.1):
    """
    Train the Pac-Man agent using Q-learning
    
    Args:
        episodes: Number of episodes to train
        render_mode: None or "human" for visualization
        render_delay: Delay between frames when rendering
    """
    # Create directories for saving results
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialize environment and agent
    env = PacmanEnv(render_mode=render_mode)
    agent = QLearningAgent(
        state_size=(15, 15),
        action_size=4,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Track metrics
    all_rewards = []
    all_steps = []
    win_count = 0
    win_history = []
    ghost_captures = []
    power_pellets = []
    
    print(f"Starting training for {episodes} episodes...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_ghost_captures = 0
        episode_power_pellets = 0
        
        while not done:
            # Choose action
            action = agent.choose_action(state)
            
            # Take step
            next_state, reward, done, _, _ = env.step(action)
            
            # Track special rewards
            if reward == 100:  # Ghost capture reward
                episode_ghost_captures += 1
            elif reward == 20:  # Power pellet reward
                episode_power_pellets += 1
            
            # Learn from this step
            agent.learn(state, action, reward, next_state, done)
            
            # Update current state
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render if requested
            if render_mode == "human":
                env.render()
                time.sleep(render_delay)
        
        # Track metrics for this episode
        all_rewards.append(total_reward)
        all_steps.append(steps)
        ghost_captures.append(episode_ghost_captures)
        power_pellets.append(episode_power_pellets)
        
        # Check if won (all food collected)
        won = len(env.food_positions) == 0
        if won:
            win_count += 1
        win_history.append(1 if won else 0)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            avg_steps = np.mean(all_steps[-10:])
            recent_win_rate = np.mean(win_history[-10:]) if win_history else 0
            
            print(f"Episode: {episode + 1}/{episodes}")
            print(f"Average Reward (last 10): {avg_reward:.1f}")
            print(f"Average Steps (last 10): {avg_steps:.1f}")
            print(f"Recent Win Rate: {recent_win_rate:.1%}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Ghost Captures: {episode_ghost_captures}")
            print("-" * 30)
        
        # Save model periodically
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            agent.save_q_table(f"models/q_table_episode_{episode+1}.pkl")
            plot_training_results(all_rewards, all_steps, win_history, ghost_captures, power_pellets, episode+1)
    
    # Save final model
    agent.save_q_table("models/q_table_final.pkl")
    
    # Plot final results
    plot_training_results(all_rewards, all_steps, win_history, ghost_captures, power_pellets, episodes)
    
    print(f"\nTraining completed. Final win rate: {win_count/episodes:.1%}")
    print(f"Final Q-table size: {len(agent.q_table)} states")
    
    env.close()
    return agent

def plot_training_results(rewards, steps, wins, ghost_captures, power_pellets, episodes):
    """Plot training results and save to file"""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards, alpha=0.5, label='Rewards')
    plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), label='20-ep Moving Avg')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot steps
    plt.subplot(2, 2, 2)
    plt.plot(steps, alpha=0.5, label='Steps')
    plt.plot(np.convolve(steps, np.ones(20)/20, mode='valid'), label='20-ep Moving Avg')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    # Plot win rate
    plt.subplot(2, 2, 3)
    window_size = min(100, len(wins))
    if window_size > 0:
        win_rate = np.convolve(wins, np.ones(window_size)/window_size, mode='valid')
        plt.plot(win_rate, label=f'{window_size}-ep Win Rate')
        plt.title(f'Win Rate ({window_size}-episode window)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
    
    # Plot ghost captures and power pellets
    plt.subplot(2, 2, 4)
    plt.plot(ghost_captures, label='Ghost Captures', alpha=0.5)
    plt.plot(power_pellets, label='Power Pellets', alpha=0.5)
    plt.title('Ghost Captures & Power Pellets')
    plt.xlabel('Episode')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"plots/training_progress_{episodes}.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Pac-Man agent")
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--delay', type=float, default=0.001, help='Delay between frames when rendering')
    args = parser.parse_args()
    
    trained_agent = train(
        episodes=args.episodes,
        render_mode="human" if args.render else None,
        render_delay=args.delay
    )