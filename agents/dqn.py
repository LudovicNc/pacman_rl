import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque

# Define the neural network for DQN
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=11, action_size=4, learning_rate=0.001, 
                 discount_factor=0.95, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, memory_size=10000, batch_size=64,
                 target_update_freq=100):
        self.state_size = state_size  # State vector size after preprocessing
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory = deque(maxlen=memory_size)
        self.step_counter = 0
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Network and Target Network
        self.q_network = DQNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For state preprocessing
        self.prev_walls = set()

    def preprocess_state(self, state):
        """Convert the state (grid) to a feature vector the DQN can use"""
        pacman_pos = None
        ghost_pos = None
        food_positions = []
        power_pellets = []
        
        # Find positions of game elements
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if np.array_equal(state[i, j], [255, 255, 0]):  # Pac-Man (yellow)
                    pacman_pos = (i, j)
                elif np.array_equal(state[i, j], [255, 0, 0]):  # Ghost (red)
                    ghost_pos = (i, j)
                    ghost_vulnerable = False
                elif np.array_equal(state[i, j], [0, 0, 255]):  # Vulnerable ghost (blue) or wall
                    # Distinguish between ghost and wall
                    if (i, j) != ghost_pos and (i, j) not in self.prev_walls:
                        ghost_pos = (i, j)
                        ghost_vulnerable = True
                elif np.array_equal(state[i, j], [255, 255, 255]):  # Food or power pellet (white)
                    # Check if it's a power pellet (in corners)
                    if (i, j) in [(1, 1), (1, state.shape[1]-2), 
                                 (state.shape[0]-2, 1), (state.shape[0]-2, state.shape[1]-2)]:
                        power_pellets.append((i, j))
                    else:
                        food_positions.append((i, j))
        
        # Remember walls for next time
        self.prev_walls = set()
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if np.array_equal(state[i, j], [0, 0, 255]) and (i, j) != ghost_pos:
                    self.prev_walls.add((i, j))
        
        # Handle case where we can't find elements
        if not pacman_pos or not ghost_pos:
            return np.zeros(self.state_size)
        
        grid_size = state.shape[0]  # Assuming square grid
        
        # Calculate relative positions and distances
        # Ghost direction and distance
        dx_ghost = ghost_pos[0] - pacman_pos[0]
        dy_ghost = ghost_pos[1] - pacman_pos[1]
        
        # Account for wraparound
        if abs(dx_ghost) > grid_size // 2:
            dx_ghost = -np.sign(dx_ghost) * (grid_size - abs(dx_ghost))
        if abs(dy_ghost) > grid_size // 2:
            dy_ghost = -np.sign(dy_ghost) * (grid_size - abs(dy_ghost))
            
        ghost_distance = abs(dx_ghost) + abs(dy_ghost)  # Manhattan distance
        
        # Normalized direction to ghost
        ghost_dir_x = dx_ghost / (grid_size // 2) if dx_ghost != 0 else 0
        ghost_dir_y = dy_ghost / (grid_size // 2) if dy_ghost != 0 else 0
        
        # Check if ghost is vulnerable
        ghost_vulnerable = 1.0 if 'ghost_vulnerable' in locals() and ghost_vulnerable else 0.0
        
        # Find nearest food
        nearest_food_distance = float('inf')
        nearest_food_dir_x = 0
        nearest_food_dir_y = 0
        
        if food_positions:
            for food_pos in food_positions:
                dx = food_pos[0] - pacman_pos[0]
                dy = food_pos[1] - pacman_pos[1]
                
                # Account for wraparound
                if abs(dx) > grid_size // 2:
                    dx = -np.sign(dx) * (grid_size - abs(dx))
                if abs(dy) > grid_size // 2:
                    dy = -np.sign(dy) * (grid_size - abs(dy))
                    
                distance = abs(dx) + abs(dy)
                
                if distance < nearest_food_distance:
                    nearest_food_distance = distance
                    nearest_food_dir_x = dx / (grid_size // 2) if dx != 0 else 0
                    nearest_food_dir_y = dy / (grid_size // 2) if dy != 0 else 0
        else:
            # No food left - use default values
            nearest_food_distance = 0
        
        # Normalize food distance
        nearest_food_distance = nearest_food_distance / (grid_size * 2)
        
        # Check if power pellets are available
        has_power_pellet = 1.0 if power_pellets else 0.0
        
        # Find nearest power pellet if any
        nearest_pellet_dir_x = 0
        nearest_pellet_dir_y = 0
        nearest_pellet_distance = 0
        
        if power_pellets:
            nearest_pellet_distance = float('inf')
            for pellet_pos in power_pellets:
                dx = pellet_pos[0] - pacman_pos[0]
                dy = pellet_pos[1] - pacman_pos[1]
                
                # Account for wraparound
                if abs(dx) > grid_size // 2:
                    dx = -np.sign(dx) * (grid_size - abs(dx))
                if abs(dy) > grid_size // 2:
                    dy = -np.sign(dy) * (grid_size - abs(dy))
                    
                distance = abs(dx) + abs(dy)
                
                if distance < nearest_pellet_distance:
                    nearest_pellet_distance = distance
                    nearest_pellet_dir_x = dx / (grid_size // 2) if dx != 0 else 0
                    nearest_pellet_dir_y = dy / (grid_size // 2) if dy != 0 else 0
            
            # Normalize power pellet distance
            nearest_pellet_distance = nearest_pellet_distance / (grid_size * 2)
        
        # Check if in tunnel
        in_tunnel = 1.0 if (pacman_pos[0] <= 1 or pacman_pos[0] >= grid_size-2) else 0.0
        
        # Build feature vector (normalized to [-1, 1] or [0, 1] range)
        features = np.array([
            ghost_dir_x,                  # Direction to ghost (x)
            ghost_dir_y,                  # Direction to ghost (y)
            ghost_distance / (grid_size), # Normalized distance to ghost
            nearest_food_dir_x,           # Direction to nearest food (x)
            nearest_food_dir_y,           # Direction to nearest food (y)
            nearest_food_distance,        # Normalized distance to nearest food
            ghost_vulnerable,             # Whether ghost is vulnerable (0 or 1)
            has_power_pellet,             # Whether power pellets are available (0 or 1)
            in_tunnel,                    # Whether Pac-Man is in tunnel area (0 or 1)
            nearest_pellet_dir_x,         # Direction to nearest power pellet (x)
            nearest_pellet_dir_y          # Direction to nearest power pellet (y)
        ])
        
        return features
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        state_features = self.preprocess_state(state)
        next_state_features = self.preprocess_state(next_state)
        
        self.memory.append((state_features, action, reward, next_state_features, done))
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        # Initialize walls tracking if not already done
        if not hasattr(self, 'prev_walls'):
            self.prev_walls = set()
            
        state_features = self.preprocess_state(state)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Exploitation - choose best action from Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
    
    def learn(self):
        """Update model weights using batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions).squeeze(1)
        
        # Calculate target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.discount_factor * next_q
        
        # Compute loss and optimize
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save_model(self, path="models/dqn_model.pth"):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="models/dqn_model.pth"):
        """Load trained model"""
        try:
            checkpoint = torch.load(path)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {path}")
            return True
        except FileNotFoundError:
            print(f"No saved model found at {path}")
            self.prev_walls = set()  # Initialize walls tracking
            return False
