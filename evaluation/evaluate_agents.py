import gymnasium as gym
import argparse
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent

def evaluate_agent(agent, env, episodes=100):
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.ravel_multi_index(state[:2], (10, 10))
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.ravel_multi_index(next_state[:2], (10, 10))
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    return rewards

def plot_results(q_rewards, dqn_rewards, ppo_rewards):
    plt.figure(figsize=(10,5))
    plt.plot(q_rewards, label="Q-Learning")
    plt.plot(dqn_rewards, label="DQN")
    plt.plot(ppo_rewards, label="PPO")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.title("Agent Performance Comparison")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["q_learning", "dqn", "ppo", "all"], required=True, help="Select the agent to evaluate")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    args = parser.parse_args()
    
    env = gym.make("PacmanEnv")
    
    q_rewards, dqn_rewards, ppo_rewards = [], [], []
    
    if args.agent in ["q_learning", "all"]:
        q_agent = QLearningAgent(state_size=100, action_size=4)
        q_agent.load_q_table()
        q_rewards = evaluate_agent(q_agent, env, args.episodes)
    
    if args.agent in ["dqn", "all"]:
        dqn_agent = DQNAgent(state_size=100, action_size=4)
        dqn_agent.policy_net.load_state_dict(torch.load("models/dqn_model.pth"))
        dqn_rewards = evaluate_agent(dqn_agent, env, args.episodes)
    
    if args.agent in ["ppo", "all"]:
        ppo_agent = PPOAgent(state_size=100, action_size=4)
        ppo_agent.policy_net.load_state_dict(torch.load("models/ppo_model.pth"))
        ppo_rewards = evaluate_agent(ppo_agent, env, args.episodes)
    
    if args.agent == "all":
        plot_results(q_rewards, dqn_rewards, ppo_rewards)
    
    env.close()

if __name__ == "__main__":
    main()
