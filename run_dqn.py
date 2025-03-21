import time
import argparse
from environment.pacman_env import PacmanEnv
from agents.dqn import DQNAgent
import numpy as np

def run_agent(model_path="models/dqn_model_final.pth", episodes=5, render_delay=0.1):
    """
    Run a trained DQN Pac-Man agent
    
    Args:
        model_path: Path to the saved model
        episodes: Number of episodes to run
        render_delay: Delay between frames when rendering
    """
    # Initialize environment (always render when running)
    env = PacmanEnv(render_mode="human")
    
    # Initialize agent
    agent = DQNAgent(state_size=11, action_size=4)
    
    # Load saved model
    if not agent.load_model(model_path):
        print(f"Failed to load model from {model_path}. Exiting.")
        return
    
    # Set epsilon to minimum for evaluation (minimal exploration)
    agent.epsilon = agent.epsilon_min
    
    # Track statistics
    rewards = []
    steps_list = []
    win_count = 0
    ghost_capture_count = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_ghost_captures = 0
        done = False
        
        print(f"\nEpisode {episode+1}/{episodes}")
        
        while not done:
            # Choose action
            action = agent.choose_action(state)
            
            # Take step
            next_state, reward, done, _, _ = env.step(action)
            
            # Track ghost captures
            if reward == 100:  # Ghost capture reward
                episode_ghost_captures += 1
                ghost_capture_count += 1
            
            total_reward += reward
            steps += 1
            
            # Render
            env.render()
            time.sleep(render_delay)
            
            # Print info (less frequently to avoid spam)
            if steps % 10 == 0 or done:
                food_remaining = len(env.food_positions)
                food_collected = env.initial_food_count - food_remaining
                progress = 100 * (food_collected / env.initial_food_count)
                
                print(f"Step {steps}, Total Reward: {total_reward:.1f}, Food: {food_collected}/{env.initial_food_count} ({progress:.1f}%)")
            
            state = next_state
        
        # Track statistics
        rewards.append(total_reward)
        steps_list.append(steps)
        
        # Check if won
        win = len(env.food_positions) == 0
        if win:
            win_count += 1
        
        # Print episode summary
        print(f"Episode {episode+1} finished after {steps} steps with reward {total_reward:.1f}")
        print(f"Result: {'Won' if win else 'Lost'}")
        print(f"Ghost captures: {episode_ghost_captures}")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    print(f"Win Rate: {win_count/episodes:.1%}")
    print(f"Total Ghost Captures: {ghost_capture_count}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained DQN Pac-Man agent")
    parser.add_argument("--model", type=str, default="models/dqn_model_final.pth", 
                      help="Path to saved model file")
    parser.add_argument("--episodes", type=int, default=5, 
                      help="Number of episodes to run")
    parser.add_argument("--delay", type=float, default=0.1, 
                      help="Delay between frames")
    
    args = parser.parse_args()
    
    run_agent(
        model_path=args.model,
        episodes=args.episodes,
        render_delay=args.delay
    )