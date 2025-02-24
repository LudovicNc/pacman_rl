import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        
    def get_state_key(self, state):
        """Convert state array to a hashable tuple."""
        # Extract Pac-Man, Ghost, and Food positions from state
        pacman_pos = None
        ghost_pos = None
        food_positions = []
        
        # Scan the state array
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                pixel = state[i, j]
                if np.array_equal(pixel, [255, 255, 0]):  # Pac-Man
                    pacman_pos = (i, j)
                elif np.array_equal(pixel, [255, 0, 0]):  # Ghost
                    ghost_pos = (i, j)
                elif np.array_equal(pixel, [255, 255, 255]):  # Food
                    food_positions.append((i, j))
        
        return (pacman_pos, ghost_pos, tuple(sorted(food_positions)))
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        state_key = self.get_state_key(state)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state):
        """Update Q-values using Q-learning update rule."""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        self.epsilon *= self.epsilon_decay
