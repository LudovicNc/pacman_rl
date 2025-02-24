import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPOAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0003, gamma=0.99, epsilon=0.2, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        
        self.policy_net = PolicyNetwork(state_size, action_size)
        self.value_net = ValueNetwork(state_size)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=learning_rate)
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def compute_advantage(self, rewards, values):
        advantages = []
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        advantages = returns - values.detach()
        return advantages, returns
    
    def update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.cat(log_probs_old)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        for _ in range(10):
            probs = self.policy_net(states)
            dist = Categorical(probs)
            log_probs_new = dist.log_prob(actions)
            ratio = torch.exp(log_probs_new - log_probs_old)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(self.value_net(states).squeeze(), returns)
            
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()
            
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()
    
if __name__ == "__main__":
    env = gym.make("PacmanEnv")
    state_size = 100  # Assuming a flattened 10x10 grid
    action_size = 4
    agent = PPOAgent(state_size, action_size)
    
    episodes = 1000
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
