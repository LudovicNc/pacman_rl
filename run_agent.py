import time
import argparse
from environment.pacman_env import PacmanEnv
from agents.q_learning import QLearningAgent
import numpy as np

def run_agent(model_path="models/q_table_final.pkl", episodes=5, render_delay=0.1):
    """
    Run a trained Pac-Man agent
    
    Args:
        model_path: Path to the saved Q-table
        episodes: Number of episodes to run
        render_delay: Delay between frames when rendering
    """
    # Initialize environment
    env = PacmanEnv(render_mode="human")
    state_size = (15, 15)  # Match the environment grid size
    action_size = 4
    
    # Initialize agent
    agent = QLearningAgent(state_size=state_size, action_size=action_size)
    
    # Load saved Q-table
    if not agent.load_q_table(model_path):
        print(f"Failed to load model from {model_path}. Starting with a new agent.")
    
    # Set epsilon to minimum for evaluation (minimal exploration)
    agent.epsilon = agent.epsilon_min
    
    # Track statistics
    rewards = []
    steps_list = []
    win_count = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode+1}/{episodes}")
        
        while not done:
            # Choose action
            action = agent.choose_action(state)
            
            # Take step
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            # Render
            env.render()
            time.sleep(render_delay)
            
            # Print info (less frequently to avoid spam)
            if steps % 10 == 0 or done:
                print(f"Step {steps}, Total Reward: {total_reward:.1f}")
            
            state = next_state
        
        # Track statistics
        rewards.append(total_reward)
        steps_list.append(steps)
        win = len(env.food_positions) == 0
        if win:
            win_count += 1
        
        # Print episode summary
        print(f"Episode {episode+1} finished after {steps} steps with reward {total_reward:.1f}")
        print(f"Result: {'Won' if win else 'Lost'}")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    print(f"Win Rate: {win_count/episodes:.1%}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained Pac-Man agent")
    parser.add_argument("--model", type=str, default="models/q_table_final.pkl", 
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