import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, state_size=(15, 15), action_size=4, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        
    def get_state_key(self, state):
        """Convert state to a more detailed representation with core game elements"""
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
                    # Check if it's a ghost or wall - ghost is usually moving
                    if (i, j) != ghost_pos and (i, j) not in self.prev_walls:
                        ghost_pos = (i, j)
                        ghost_vulnerable = True
                elif np.array_equal(state[i, j], [255, 255, 255]):  # Food or power pellet (white)
                    # Check if it's a power pellet (in corners)
                    if (i, j) in [(1, 1), (1, self.state_size[1]-2), 
                                 (self.state_size[0]-2, 1), (self.state_size[0]-2, self.state_size[1]-2)]:
                        power_pellets.append((i, j))
                    else:
                        food_positions.append((i, j))
        
        # Remember walls for next time
        self.prev_walls = set()
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if np.array_equal(state[i, j], [0, 0, 255]) and (i, j) != ghost_pos:
                    self.prev_walls.add((i, j))
        
        # Handle case where we can't find elements (should rarely happen)
        if not pacman_pos or not ghost_pos:
            return (0, 0, 0, 0, 0, 0, 0)
        
        # Calculate relative position from pacman to ghost
        # X direction (left/right)
        if ghost_pos[0] < pacman_pos[0]:
            dx_direct = pacman_pos[0] - ghost_pos[0]
            dx_wrap = ghost_pos[0] + self.state_size[0] - pacman_pos[0]
            ghost_dir_x = -1 if dx_direct <= dx_wrap else 1  # -1 = ghost is left, 1 = ghost is right
        elif ghost_pos[0] > pacman_pos[0]:
            dx_direct = ghost_pos[0] - pacman_pos[0]
            dx_wrap = pacman_pos[0] + self.state_size[0] - ghost_pos[0]
            ghost_dir_x = 1 if dx_direct <= dx_wrap else -1
        else:
            ghost_dir_x = 0  # same column
            
        # Y direction (up/down)
        if ghost_pos[1] < pacman_pos[1]:
            dy_direct = pacman_pos[1] - ghost_pos[1]
            dy_wrap = ghost_pos[1] + self.state_size[1] - pacman_pos[1]
            ghost_dir_y = -1 if dy_direct <= dy_wrap else 1  # -1 = ghost is above, 1 = ghost is below
        elif ghost_pos[1] > pacman_pos[1]:
            dy_direct = ghost_pos[1] - pacman_pos[1]
            dy_wrap = pacman_pos[1] + self.state_size[1] - ghost_pos[1]
            ghost_dir_y = 1 if dy_direct <= dy_wrap else -1
        else:
            ghost_dir_y = 0  # same row
        
        # Calculate Manhattan distance to ghost (handling wraparound)
        dx = abs(ghost_pos[0] - pacman_pos[0])
        dy = abs(ghost_pos[1] - pacman_pos[1])
        dx = min(dx, self.state_size[0] - dx)  # Account for wraparound
        dy = min(dy, self.state_size[1] - dy)
        ghost_distance = dx + dy
        
        # Simplify distance into more bins
        if ghost_distance < 2:
            ghost_distance_bin = 0  # Very close
        elif ghost_distance < 4:
            ghost_distance_bin = 1  # Close
        elif ghost_distance < 7:
            ghost_distance_bin = 2  # Medium
        else:
            ghost_distance_bin = 3  # Far away
        
        # Find the nearest food
        nearest_food_distance = float('inf')
        nearest_food_dir_x = 0
        nearest_food_dir_y = 0
        
        if food_positions:
            for food_pos in food_positions:
                dx = abs(food_pos[0] - pacman_pos[0])
                dy = abs(food_pos[1] - pacman_pos[1])
                dx = min(dx, self.state_size[0] - dx)  # Account for wraparound
                dy = min(dy, self.state_size[1] - dy)
                distance = dx + dy
                
                if distance < nearest_food_distance:
                    nearest_food_distance = distance
                    # X direction
                    if food_pos[0] < pacman_pos[0]:
                        dx_direct = pacman_pos[0] - food_pos[0]
                        dx_wrap = food_pos[0] + self.state_size[0] - pacman_pos[0]
                        nearest_food_dir_x = -1 if dx_direct <= dx_wrap else 1
                    elif food_pos[0] > pacman_pos[0]:
                        dx_direct = food_pos[0] - pacman_pos[0]
                        dx_wrap = pacman_pos[0] + self.state_size[0] - food_pos[0]
                        nearest_food_dir_x = 1 if dx_direct <= dx_wrap else -1
                    else:
                        nearest_food_dir_x = 0
                    
                    # Y direction
                    if food_pos[1] < pacman_pos[1]:
                        dy_direct = pacman_pos[1] - food_pos[1]
                        dy_wrap = food_pos[1] + self.state_size[1] - pacman_pos[1]
                        nearest_food_dir_y = -1 if dy_direct <= dy_wrap else 1
                    elif food_pos[1] > pacman_pos[1]:
                        dy_direct = food_pos[1] - pacman_pos[1]
                        dy_wrap = pacman_pos[1] + self.state_size[1] - food_pos[1]
                        nearest_food_dir_y = 1 if dy_direct <= dy_wrap else -1
                    else:
                        nearest_food_dir_y = 0
        
        # Discretize food distance into bins
        if nearest_food_distance < 2:
            food_distance_bin = 0  # Very close
        elif nearest_food_distance < 4:
            food_distance_bin = 1  # Close
        elif nearest_food_distance < 7:
            food_distance_bin = 2  # Medium
        else:
            food_distance_bin = 3  # Far away
            
        # Check if any power pellets are available
        has_power_pellet = 1 if power_pellets else 0
        
        # Check if in tunnel area (simplified)
        in_tunnel = 1 if (pacman_pos[0] <= 1 or pacman_pos[0] >= self.state_size[0]-2) else 0
        
        # Direction to nearest power pellet (if available)
        power_pellet_dir_x = 0
        power_pellet_dir_y = 0
        
        if power_pellets:
            # Find nearest power pellet
            nearest_pellet_distance = float('inf')
            nearest_pellet_pos = None
            
            for pellet_pos in power_pellets:
                dx = abs(pellet_pos[0] - pacman_pos[0])
                dy = abs(pellet_pos[1] - pacman_pos[1])
                dx = min(dx, self.state_size[0] - dx)
                dy = min(dy, self.state_size[1] - dy)
                distance = dx + dy
                
                if distance < nearest_pellet_distance:
                    nearest_pellet_distance = distance
                    nearest_pellet_pos = pellet_pos
            
            # Calculate direction to nearest power pellet
            if nearest_pellet_pos:
                # X direction
                if nearest_pellet_pos[0] < pacman_pos[0]:
                    dx_direct = pacman_pos[0] - nearest_pellet_pos[0]
                    dx_wrap = nearest_pellet_pos[0] + self.state_size[0] - pacman_pos[0]
                    power_pellet_dir_x = -1 if dx_direct <= dx_wrap else 1
                elif nearest_pellet_pos[0] > pacman_pos[0]:
                    dx_direct = nearest_pellet_pos[0] - pacman_pos[0]
                    dx_wrap = pacman_pos[0] + self.state_size[0] - nearest_pellet_pos[0]
                    power_pellet_dir_x = 1 if dx_direct <= dx_wrap else -1
                
                # Y direction
                if nearest_pellet_pos[1] < pacman_pos[1]:
                    dy_direct = pacman_pos[1] - nearest_pellet_pos[1]
                    dy_wrap = nearest_pellet_pos[1] + self.state_size[1] - pacman_pos[1]
                    power_pellet_dir_y = -1 if dy_direct <= dy_wrap else 1
                elif nearest_pellet_pos[1] > pacman_pos[1]:
                    dy_direct = nearest_pellet_pos[1] - pacman_pos[1]
                    dy_wrap = pacman_pos[1] + self.state_size[1] - nearest_pellet_pos[1]
                    power_pellet_dir_y = 1 if dy_direct <= dy_wrap else -1
        
        # Create an expanded state key
        # Now the state includes ghost-related information, food-related information,
        # and environmental awareness (tunnel)
        state_key = (
            ghost_dir_x, ghost_dir_y, ghost_distance_bin,  # Ghost info
            nearest_food_dir_x, nearest_food_dir_y, food_distance_bin,  # Food info
            1 if ghost_vulnerable else 0,  # Ghost vulnerability
            has_power_pellet,  # Power pellet availability 
            in_tunnel,  # Environment awareness
            # Only include power pellet directions if they're available to avoid unnecessary state expansion
            power_pellet_dir_x if has_power_pellet else 0,
            power_pellet_dir_y if has_power_pellet else 0
        )
        
        return state_key
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        # Initialize walls tracking if not already done
        if not hasattr(self, 'prev_walls'):
            self.prev_walls = set()
            
        state_key = self.get_state_key(state)
        
        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        
        # Initialize state in Q-table if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Exploitation
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-values using Q-learning update rule."""
        # Initialize walls tracking if not already done
        if not hasattr(self, 'prev_walls'):
            self.prev_walls = set()
            
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        
        # If terminal state, no future reward
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state_key])
        
        # Q-learning formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state_key][action] = new_q
        
    def save_q_table(self, filename="models/q_table.pkl"):
        """Save the Q-table to a file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved with {len(self.q_table)} states")
    
    def load_q_table(self, filename="models/q_table.pkl"):
        """Load the Q-table from a file"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded with {len(self.q_table)} states")
            return True
        except FileNotFoundError:
            print(f"No saved Q-table found at {filename}")
            self.prev_walls = set()  # Initialize walls tracking
            return False
