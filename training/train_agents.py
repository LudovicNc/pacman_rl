import gymnasium as gym
import argparse
import numpy as np
import torch
import pickle
from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent

def train_q_learning(env, episodes=1000):
    agent = QLearningAgent(state_size=100, action_size=4)
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.ravel_multi_index(state[:2], (10, 10))
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.ravel_multi_index(next_state[:2], (10, 10))
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        agent.decay_exploration()
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    agent.save_q_table()

def train_dqn(env, episodes=1000):
    agent = DQNAgent(state_size=100, action_size=4)
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.ravel_multi_index(state[:2], (10, 10))
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.ravel_multi_index(next_state[:2], (10, 10))
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        if episode % 10 == 0:
            agent.update_target_network()
        print(f"Episode {episode+1}: Total Reward: {total_reward}")

def train_ppo(env, episodes=1000):
    agent = PPOAgent(state_size=100, action_size=4)
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.ravel_multi_index(state[:2], (10, 10))
        done = False
        total_reward = 0
        states, actions, rewards, log_probs = [], [], [], []
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.ravel_multi_index(next_state[:2], (10, 10))
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
            total_reward += reward
        values = agent.value_net(torch.FloatTensor(states)).squeeze()
        advantages, returns = agent.compute_advantage(rewards, values)
        agent.update(states, actions, log_probs, returns, advantages)
        print(f"Episode {episode+1}: Total Reward: {total_reward}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["q_learning", "dqn", "ppo"], required=True, help="Select the agent to train")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    args = parser.parse_args()
    
    env = gym.make("PacmanEnv")
    
    if args.agent == "q_learning":
        train_q_learning(env, args.episodes)
    elif args.agent == "dqn":
        train_dqn(env, args.episodes)
    elif args.agent == "ppo":
        train_ppo(env, args.episodes)
    
    env.close()

if __name__ == "__main__":
    main()
