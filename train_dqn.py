from environment.pacman_env import PacmanEnv
from agents.dqn import DQNAgent
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import argparse

def train_dqn(episodes=500, render_mode=None, render_delay=0.1):
    """
    Train the Pac-Man agent using DQN
    
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
    agent = DQNAgent(
        state_size=11,         # Size of our feature vector
        action_size=4,         # Up, Down, Left, Right
        learning_rate=0.0005,  # Lower learning rate for stability
        discount_factor=0.99,  # Higher discount factor for long-term planning
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,     # Store 10,000 experiences
        batch_size=64,         # Train on 64 samples at a time
        target_update_freq=100 # Update target network every 100 steps
    )
    
    # Track metrics
    all_rewards = []
    all_steps = []
    win_count = 0
    win_history = []
    ghost_captures = []
    power_pellets = []
    losses = []
    
    print(f"Starting DQN training for {episodes} episodes...")
    print(f"Using device: {agent.device}")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_ghost_captures = 0
        episode_power_pellets = 0
        episode_losses = []
        
        while not done:
            # Choose action
            action = agent.choose_action(state)
            
            # Take step
            next_state, reward, done, _, _ = env.step(action)
            
            # Track special rewards
            if reward == 100:  # Ghost capture reward
                episode_ghost_captures += 1
            elif reward == 30:  # Power pellet reward (note it's 30 in improved env)
                episode_power_pellets += 1
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experiences
            if len(agent.memory) >= agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)
            
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
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0)
        
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
            print(f"Avg Loss: {np.mean(episode_losses) if episode_losses else 0:.6f}")
            print("-" * 30)
        
        # Save model periodically
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            agent.save_model(f"models/dqn_model_episode_{episode+1}.pth")
            plot_training_results(all_rewards, all_steps, win_history, ghost_captures, power_pellets, losses, episode+1)
    
    # Save final model
    agent.save_model("models/dqn_model_final.pth")
    
    # Plot final results
    plot_training_results(all_rewards, all_steps, win_history, ghost_captures, power_pellets, losses, episodes)
    
    print(f"\nTraining completed. Final win rate: {win_count/episodes:.1%}")
    
    env.close()
    return agent

def plot_training_results(rewards, steps, wins, ghost_captures, power_pellets, losses, episodes):
    """Plot training results and save to file"""
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(rewards, alpha=0.5, label='Rewards')
    plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), label='20-ep Moving Avg')
    plt.title('DQN Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # Plot steps
    plt.subplot(3, 2, 2)
    plt.plot(steps, alpha=0.5, label='Steps')
    plt.plot(np.convolve(steps, np.ones(20)/20, mode='valid'), label='20-ep Moving Avg')
    plt.title('DQN Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    # Plot win rate
    plt.subplot(3, 2, 3)
    window_size = min(100, len(wins))
    if window_size > 0:
        win_rate = np.convolve(wins, np.ones(window_size)/window_size, mode='valid')
        plt.plot(win_rate, label=f'{window_size}-ep Win Rate')
        plt.title(f'DQN Win Rate ({window_size}-episode window)')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
    
    # Plot ghost captures and power pellets
    plt.subplot(3, 2, 4)
    plt.plot(ghost_captures, label='Ghost Captures', alpha=0.5)
    plt.plot(power_pellets, label='Power Pellets', alpha=0.5)
    plt.title('DQN Ghost Captures & Power Pellets')
    plt.xlabel('Episode')
    plt.legend()
    
    # Plot losses
    plt.subplot(3, 2, 5)
    plt.plot(losses, alpha=0.5, label='Loss')
    plt.plot(np.convolve(losses, np.ones(20)/20, mode='valid'), label='20-ep Moving Avg')
    plt.title('DQN Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot cumulative win rate
    plt.subplot(3, 2, 6)
    cumulative_wins = np.cumsum(wins)
    episodes_array = np.arange(1, len(wins) + 1)
    plt.plot(cumulative_wins / episodes_array, label='Cumulative Win Rate')
    plt.title('DQN Cumulative Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"plots/dqn_training_progress_{episodes}.png")
    plt.close()

    # Create comparison plot with Q-learning results if available
    try:
        # Load Q-learning results if they exist
        q_rewards = np.load("plots/q_rewards.npy")
        q_steps = np.load("plots/q_steps.npy")
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # Compare rewards
        plt.subplot(1, 2, 1)
        plt.plot(np.convolve(q_rewards, np.ones(20)/20, mode='valid'), label='Q-Learning')
        plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), label='DQN')
        plt.title('Reward Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Reward (20-ep Moving Avg)')
        plt.legend()
        
        # Compare steps
        plt.subplot(1, 2, 2)
        plt.plot(np.convolve(q_steps, np.ones(20)/20, mode='valid'), label='Q-Learning')
        plt.plot(np.convolve(steps, np.ones(20)/20, mode='valid'), label='DQN')
        plt.title('Episode Length Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Steps (20-ep Moving Avg)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"plots/algorithm_comparison_{episodes}.png")
        plt.close()
    except:
        # Save Q-learning results for future comparison
        np.save("plots/dqn_rewards.npy", np.array(rewards))
        np.save("plots/dqn_steps.npy", np.array(steps))
        print("Saved DQN results for future comparisons")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pac-Man agent using DQN")
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--delay', type=float, default=0.001, help='Delay between frames when rendering')
    args = parser.parse_args()
    
    # Save Q-learning results for comparison if available
    if os.path.exists("plots/training_progress_500.png"):
        try:
            # Load Q-learning metrics from train.py
            import json
            with open("metrics/episode_500.json", "r") as f:
                q_data = json.load(f)
                np.save("plots/q_rewards.npy", np.array(q_data["rewards"]))
                np.save("plots/q_steps.npy", np.array(q_data["episode_lengths"]))
                print("Loaded Q-learning results for comparison")
        except:
            print("Could not load Q-learning results, will save DQN results only")
    
    trained_agent = train_dqn(
        episodes=args.episodes,
        render_mode="human" if args.render else None,
        render_delay=args.delay
    )