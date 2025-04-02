from environment.pacman_env import PacmanEnv
from agents.dqn import DQNAgent
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import argparse
import matplotlib.font_manager as fm

def train_dqn(episodes=750, render_mode=None, render_delay=0.1):
    """
    Train the Pac-Man agent using DQN.
    
    Tracks:
      - Total reward per episode
      - Episode length (steps)
      - Win flag (1 if all food is collected, else 0)
      - Ghost captures (env.ghost_capture_count)
      - Power pellet pickups (computed as initial count - remaining power pellets)
      - Food left at episode end (len(env.food_positions))
      - Power pellets left at episode end (len(env.power_pellets))
      - Food progress: proportion of food eaten = 1.0 - (food_left / initial_food_count)
    
    Every 10 episodes, the average values are printed.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialize environment and agent
    env = PacmanEnv(render_mode=render_mode)
    agent = DQNAgent(
        state_size=11,
        action_size=4,
        learning_rate=0.0005,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=128,
        target_update_freq=25
    )
    
    # Metrics tracking lists
    all_rewards = []
    all_steps = []
    wins = []  # 1 if all food collected, else 0
    ghost_captures = []  # Taken directly from env.ghost_capture_count
    power_pellets_collected = []  # Computed as env.initial_power_pellet_count - len(env.power_pellets)
    losses = []
    food_left = []   # Food pellets remaining at episode end
    power_left = []  # Power pellets remaining at episode end
    food_progress = []  # Proportion of food eaten: 1 - (food_left / initial_food_count)
    
    print(f"Starting DQN training for {episodes} episodes...")
    print(f"Using device: {agent.device}")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_losses = []
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if len(agent.memory) >= agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render_mode == "human":
                env.render()
                time.sleep(render_delay)
        
        # Decay epsilon at the end of the episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Get ghost captures and power pellet pickups from the environment
        episode_ghost_captures = env.ghost_capture_count
        episode_power_pellets = env.initial_power_pellet_count - len(env.power_pellets)
        
        # Compute the proportion of food eaten in this episode.
        progress = 1.0 - len(env.food_positions) / env.initial_food_count
        
        # Log metrics for this episode
        all_rewards.append(total_reward)
        all_steps.append(steps)
        wins.append(1 if len(env.food_positions) == 0 else 0)
        ghost_captures.append(episode_ghost_captures)
        power_pellets_collected.append(episode_power_pellets)
        food_left.append(len(env.food_positions))
        power_left.append(len(env.power_pellets))
        food_progress.append(progress)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_rewards[-10:])
            avg_steps = np.mean(all_steps[-10:])
            avg_win = np.mean(wins[-10:]) * 100
            avg_ghost = np.mean(ghost_captures[-10:])
            avg_power = np.mean(power_pellets_collected[-10:])
            avg_food_prog = np.mean(food_progress[-10:]) * 100
            print(f"Episode: {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.1f}")
            print(f"  Avg Steps (last 10): {avg_steps:.1f}")
            print(f"  Avg Win Rate (last 10): {avg_win:.1f}%")
            print(f"  Avg Ghost Captures (last 10): {avg_ghost:.1f}")
            print(f"  Avg Power Pellets Eaten (last 10): {avg_power:.1f}")
            print(f"  Avg Food Eaten % (last 10): {avg_food_prog:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print("-" * 30)
        
        if (episode + 1) % 100 == 0 or episode == episodes - 1:
            agent.save_model(f"models/dqn_model_episode_{episode+1}.pth")
            plot_training_results(all_rewards, all_steps, wins, ghost_captures, 
                                  power_pellets_collected, losses, food_left, power_left, food_progress, episode+1)
    
    agent.save_model("models/dqn_model_final.pth")
    plot_training_results(all_rewards, all_steps, wins, ghost_captures, 
                          power_pellets_collected, losses, food_left, power_left, food_progress, episodes)
    
    print(f"\nTraining completed. Final win rate: {np.mean(wins):.1%}")
    env.close()
    return agent


def get_font(font_name, fallback):
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    return font_name if font_name in available_fonts else fallback


def plot_training_results(rewards, steps, wins, ghost_captures, power_pellets_collected, 
                          losses, food_left, power_left, food_progress, episodes):
    """Plot training results with a retro 80's vibe and save to file"""
    import matplotlib.pyplot as plt
    import numpy as np

    title_font_name = get_font("Pixelion", "DejaVu Sans")
    label_font_name = get_font("Roboto Compact", "DejaVu Sans")

    title_font = {'family': title_font_name, 'color': 'white', 'size': 14}
    label_font = {'family': label_font_name, 'color': 'white', 'size': 12}

    # Create a figure with black background
    plt.figure(figsize=(15, 20), facecolor='black')
    
    # Helper function to style each axis
    def style_axis():
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        plt.grid(True, color='#341bca')
        leg = plt.legend(prop={'family': 'DejaVu Sans'})
        leg.get_frame().set_facecolor('black')
        leg.get_frame().set_edgecolor('#341bca')
        for text in leg.get_texts():
            text.set_color('white')
    
    # Subplot 1: Episode Rewards
    plt.subplot(5, 2, 1)
    plt.plot(rewards, alpha=0.5, label='Rewards', color='white')
    plt.plot(np.convolve(rewards, np.ones(20)/20, mode='valid'), label='20-ep Moving Avg', color='red')
    plt.title('DQN Episode Rewards', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    plt.ylabel('Total Reward', fontdict=label_font)
    style_axis()
    
    # Subplot 2: Episode Lengths
    plt.subplot(5, 2, 2)
    plt.plot(steps, alpha=0.5, label='Steps', color='white')
    plt.plot(np.convolve(steps, np.ones(20)/20, mode='valid'), label='20-ep Moving Avg', color='red')
    plt.title('DQN Episode Lengths', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    plt.ylabel('Steps', fontdict=label_font)
    style_axis()
    
    # Subplot 3: Win Rate (20-episode moving average)
    plt.subplot(5, 2, 3)
    window_size = min(100, len(wins))
    if window_size > 0:
        win_rate = np.convolve(wins, np.ones(window_size)/window_size, mode='valid')
        plt.plot(win_rate, label=f'{window_size}-ep Win Rate', color='white')
    plt.title(f'DQN Win Rate ({window_size}-episode window)', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    plt.ylabel('Win Rate', fontdict=label_font)
    plt.ylim(0, 1)
    style_axis()
    
    # Subplot 4: Ghost Captures & Power Pellet Pickups
    plt.subplot(5, 2, 4)
    plt.plot(ghost_captures, label='Ghost Captures', alpha=0.5, color='white')
    plt.plot(power_pellets_collected, label='Power Pellet Pickups', alpha=0.5, color='red')
    plt.title('Ghost Captures & Power Pellet Pickups', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    style_axis()
    
    # Subplot 5: Training Loss
    plt.subplot(5, 2, 5)
    plt.plot(losses, alpha=0.5, label='Loss', color='white')
    plt.plot(np.convolve(losses, np.ones(20)/20, mode='valid'), label='20-ep Moving Avg', color='red')
    plt.title('Training Loss', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    plt.ylabel('Loss', fontdict=label_font)
    style_axis()
    
    # Subplot 6: Cumulative Win Rate
    plt.subplot(5, 2, 6)
    cumulative_wins = np.cumsum(wins)
    episodes_array = np.arange(1, len(wins) + 1)
    plt.plot(cumulative_wins / episodes_array, label='Cumulative Win Rate', color='white')
    plt.title('Cumulative Win Rate', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    plt.ylabel('Win Rate', fontdict=label_font)
    plt.ylim(0, 1)
    style_axis()
    
    # Subplot 7: Food Pellets Left per Episode
    plt.subplot(5, 2, 7)
    plt.plot(food_left, alpha=0.5, label='Food Left', color='white')
    plt.plot(np.convolve(food_left, np.ones(10)/10, mode='valid'), label='10-ep Moving Avg', color='red')
    plt.title('Food Pellets Left per Episode', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    plt.ylabel('Pellets Left', fontdict=label_font)
    style_axis()
    
    # Subplot 8: Power Pellets Left per Episode
    plt.subplot(5, 2, 8)
    plt.plot(power_left, alpha=0.5, label='Power Pellets Left', color='white')
    plt.plot(np.convolve(power_left, np.ones(10)/10, mode='valid'), label='10-ep Moving Avg', color='red')
    plt.title('Power Pellets Left per Episode', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    plt.ylabel('Pellets Left', fontdict=label_font)
    style_axis()
    
    # Subplot 9: Food Eaten Proportion per Episode
    plt.subplot(5, 2, 9)
    plt.plot(food_progress, alpha=0.5, label='Food Eaten Proportion', color='white')
    plt.plot(np.convolve(food_progress, np.ones(10)/10, mode='valid'), label='10-ep Moving Avg', color='red')
    plt.title('Food Eaten Proportion per Episode', fontdict=title_font)
    plt.xlabel('Episode', fontdict=label_font)
    plt.ylabel('Proportion Eaten', fontdict=label_font)
    style_axis()
    
    # Subplot 10: (Empty)
    plt.subplot(5, 2, 10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"plots/dqn_training_progress_{episodes}.png", facecolor='black')
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pac-Man agent using DQN")
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--delay', type=float, default=0.001, help='Delay between frames when rendering')
    args = parser.parse_args()
    
    # Save Q-learning results for comparison if available
    if os.path.exists("plots/training_progress_500.png"):
        try:
            import json
            with open("metrics/episode_500.json", "r") as f:
                q_data = json.load(f)
                np.save("plots/q_rewards.npy", np.array(q_data["rewards"]))
                np.save("plots/q_steps.npy", np.array(q_data["episode_lengths"]))
                print("Loaded Q-learning results for comparison")
        except:
            print("Could not load Q-learning results, will save DQN results only")
    
    trained_agent = train_dqn(
        render_mode="human" if args.render else None,
        render_delay=args.delay
    )
