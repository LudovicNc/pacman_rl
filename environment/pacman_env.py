import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class PacmanEnv(gym.Env):
    """Custom Pac-Man Environment for Reinforcement Learning"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Basic environment setup
        self.grid_size = (15, 15)
        self.walls = self._create_walls()
        self.state = None
        self.done = False
        self.power_pellets = set()  # Will be initialized in reset
        self.ghost_is_vulnerable = False
        self.power_pellet_timer = 0
        self.POWER_PELLET_DURATION = 50  # Duration in steps
        self.steps_counter = 0
        self.MAX_STEPS = 750  # Max steps per episode
        self.last_food_steps = 0  # Track steps since last food
        self.MAX_STEPS_WITHOUT_FOOD = 100  # Prevent endless wandering
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low=0, high=255, shape=(15, 15, 3), dtype=np.uint8)
        
        # Setup for visualization
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.window_size = 600
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Pac-Man RL")
    
    def reset(self, seed=None, options=None):
      """Reset the environment for a new episode"""
      super().reset(seed=seed)
      
      # Initialize game state
      self._initialize_game()
      
      # Generate initial state
      self.state = self._generate_state_representation()
      self.done = False
      self.steps_counter = 0
      self.last_food_steps = 0
      self.power_active_steps = 0 
      self.ghost_capture_count = 0
      return self.state, {}
    
    def step(self, action):
  
      reward = 0
      self.steps_counter += 1
      self.last_food_steps += 1

      # Update power pellet timer
      if self.ghost_is_vulnerable:
          self.power_pellet_timer -= 1
          if self.power_pellet_timer <= 0:
              self.ghost_is_vulnerable = False

      # Track power pellet active time
      if self.ghost_is_vulnerable:
          self.power_active_steps += 1

      # Get current position
      pacman_x, pacman_y = self.pacman_position

      # Calculate new position based on action
      new_x, new_y = pacman_x, pacman_y
      if action == 0:  # Up
          new_y = (pacman_y - 1) % self.grid_size[1]
      elif action == 1:  # Down
          new_y = (pacman_y + 1) % self.grid_size[1]
      elif action == 2:  # Left
          new_x = (pacman_x - 1) % self.grid_size[0]
      elif action == 3:  # Right
          new_x = (pacman_x + 1) % self.grid_size[0]

      # Check if new position is valid (not a wall)
      if (new_x, new_y) not in self.walls:
          self.pacman_position = (new_x, new_y)
      else:
          reward -= 1  # Penalty for hitting wall

      # Check for power pellet collection
      if self.pacman_position in self.power_pellets:
          self.power_pellets.remove(self.pacman_position)
          self.ghost_is_vulnerable = True
          self.power_pellet_timer = self.POWER_PELLET_DURATION
          reward += 50  # Increased bonus for getting power pellet

      # Record ghost's old position before it moves (for ghost-chasing reward)
      ghost_old_position = self.ghost_position

      # Move ghost
      self._move_ghost()

      # Check for ghost collision
      if self.pacman_position == self.ghost_position:
        if self.ghost_is_vulnerable:
            # Ghost is captured by Pac-Man
            self.ghost_capture_count += 1  # Increment ghost capture counter
            self.ghost_position = (5, 5)  # Reset ghost to center
            reward += 200  # Large bonus for eating ghost
            self.ghost_is_vulnerable = False  # Ghost respawns
        else:
            # Pac-Man is eaten
            reward -= 500
            self.done = True

      # Check for food collection
      if self.pacman_position in self.food_positions:
          self.food_positions.remove(self.pacman_position)
          reward += 10  # Reward for eating food
          self.last_food_steps = 0  # Reset the counter

          # Bonus for progress (percentage of food eaten)
          progress = 1.0 - len(self.food_positions) / self.initial_food_count
          reward += (np.exp(progress * 3) - 1) * 5

      # Add a small reward for moving in the direction of food
      if len(self.food_positions) > 0:
          nearest_food = min(self.food_positions, 
                            key=lambda food: self._manhattan_distance(self.pacman_position, food))
          old_distance = self._manhattan_distance((pacman_x, pacman_y), nearest_food)
          new_distance = self._manhattan_distance(self.pacman_position, nearest_food)
          if new_distance < old_distance:
              reward += 3

      # Reward for moving closer to power pellets
      if len(self.power_pellets) > 0:
          nearest_pellet = min(self.power_pellets, 
                              key=lambda pellet: self._manhattan_distance((pacman_x, pacman_y), pellet))
          old_distance_pellet = self._manhattan_distance((pacman_x, pacman_y), nearest_pellet)
          new_distance_pellet = self._manhattan_distance(self.pacman_position, nearest_pellet)
          if new_distance_pellet < old_distance_pellet:
              reward += 1

      # Reward for moving closer to ghosts if power pellet timer > 5
      if self.power_pellet_timer > 5:
          old_distance_ghost = self._manhattan_distance((pacman_x, pacman_y), ghost_old_position)
          new_distance_ghost = self._manhattan_distance(self.pacman_position, self.ghost_position)
          if new_distance_ghost < old_distance_ghost:
              reward += 1

      # Small penalty for each step to encourage efficiency
      reward -= 0.1

      # Generate new state
      self.state = self._generate_state_representation()

      # Check if game is won (all food collected)
      if not self.food_positions:
          reward += 1000  # Big bonus for winning
          self.done = True

      # Check for episode timeout or stuck behavior
      if self.steps_counter >= self.MAX_STEPS or self.last_food_steps >= self.MAX_STEPS_WITHOUT_FOOD:
          self.done = True

      return self.state, reward, self.done, False, {}

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance with wraparound"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        # Consider wraparound for shortest path
        dx = min(dx, self.grid_size[0] - dx)
        dy = min(dy, self.grid_size[1] - dy)
        return dx + dy

    def _move_ghost(self):
        """Improved ghost AI"""
        ghost_x, ghost_y = self.ghost_position
        pacman_x, pacman_y = self.pacman_position
        
        # Ghost house area
        in_ghost_house = (4 <= ghost_x <= 6 and 5 <= ghost_y <= 6)
        
        # Handle ghost in house
        if in_ghost_house:
            # Move up to exit
            if ghost_y > 4 and (ghost_x, ghost_y-1) not in self.walls:
                self.ghost_position = (ghost_x, ghost_y-1)
                return
            elif ghost_x != 5:
                # Center horizontally
                self.ghost_position = (5, ghost_y)
                return
        
        # Increase randomness, especially when vulnerable
        random_threshold = 0.4 if self.ghost_is_vulnerable else 0.2
        if np.random.random() < random_threshold:
            possible_moves = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (ghost_x + dx) % self.grid_size[0]
                new_y = (ghost_y + dy) % self.grid_size[1]
                if (new_x, new_y) not in self.walls:
                    possible_moves.append((new_x, new_y))
            
            if possible_moves:
                self.ghost_position = possible_moves[np.random.randint(len(possible_moves))]
                return
        
        # Chase or flee based on vulnerability
        if self.ghost_is_vulnerable:
            # Try to move away from Pac-Man
            move_away = True
        else:
            # Try to move toward Pac-Man
            move_away = False
        
        # Determine directions to/from Pac-Man
        dx = pacman_x - ghost_x
        dy = pacman_y - ghost_y
        
        # Handle wraparound for shortest path
        if abs(dx) > self.grid_size[0] // 2:
            dx = -dx
        if abs(dy) > self.grid_size[1] // 2:
            dy = -dy
        
        # Reverse direction if moving away
        if move_away:
            dx = -dx
            dy = -dy
        
        # Try to move in the preferred direction
        if abs(dx) > abs(dy):
            # Try horizontal movement first
            dir_x = 1 if dx > 0 else -1
            new_x = (ghost_x + dir_x) % self.grid_size[0]
            if (new_x, ghost_y) not in self.walls:
                self.ghost_position = (new_x, ghost_y)
                return
            
            # Then try vertical
            dir_y = 1 if dy > 0 else -1
            new_y = (ghost_y + dir_y) % self.grid_size[1]
            if (ghost_x, new_y) not in self.walls:
                self.ghost_position = (ghost_x, new_y)
                return
        else:
            # Try vertical movement first
            dir_y = 1 if dy > 0 else -1
            new_y = (ghost_y + dir_y) % self.grid_size[1]
            if (ghost_x, new_y) not in self.walls:
                self.ghost_position = (ghost_x, new_y)
                return
            
            # Then try horizontal
            dir_x = 1 if dx > 0 else -1
            new_x = (ghost_x + dir_x) % self.grid_size[0]
            if (new_x, ghost_y) not in self.walls:
                self.ghost_position = (new_x, ghost_y)
                return
        
        # If all else fails, try any valid move
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x = (ghost_x + dx) % self.grid_size[0]
            new_y = (ghost_y + dy) % self.grid_size[1]
            if (new_x, new_y) not in self.walls:
                self.ghost_position = (new_x, new_y)
                return

    def _generate_state_representation(self):
        """Generate the state as a grid with RGB channels"""
        state = np.zeros((15, 15, 3), dtype=np.uint8)
        
        # Add Pac-Man (yellow)
        pacman_x, pacman_y = self.pacman_position
        state[pacman_x, pacman_y] = [255, 255, 0]
        
        # Add Ghost (red normally, blue when vulnerable)
        ghost_x, ghost_y = self.ghost_position
        if self.ghost_is_vulnerable:
            state[ghost_x, ghost_y] = [0, 0, 255]  # Blue
        else:
            state[ghost_x, ghost_y] = [255, 0, 0]  # Red
        
        # Add Food (white)
        for x, y in self.food_positions:
            state[x, y] = [255, 255, 255]
        
        # Add Power Pellets (white)
        for x, y in self.power_pellets:
            state[x, y] = [255, 255, 255]
        
        # Add Walls (blue)
        for x, y in self.walls:
            state[x, y] = [0, 0, 255]
        
        return state
    
    def render(self):
        """Render the current state of the environment"""
        if self.render_mode == "human":
            # Handle window close event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            self.screen.fill((0, 0, 0))  # Black background
            
            cell_size = self.window_size // self.grid_size[0]
            
            # Render walls (blue)
            for wall_x, wall_y in self.walls:
                pygame.draw.rect(self.screen, (0, 0, 255),
                               (wall_x * cell_size, wall_y * cell_size,
                                cell_size, cell_size))
            
            # Render Food (white dots)
            for food_x, food_y in self.food_positions:
                center_x = (food_x + 0.5) * cell_size
                center_y = (food_y + 0.5) * cell_size
                pygame.draw.circle(self.screen, (255, 255, 255),
                                 (int(center_x), int(center_y)), cell_size // 6)
            
            # Render Power Pellets (larger white dots)
            for pellet_x, pellet_y in self.power_pellets:
                center_x = (pellet_x + 0.5) * cell_size
                center_y = (pellet_y + 0.5) * cell_size
                pygame.draw.circle(self.screen, (255, 255, 255),
                                 (int(center_x), int(center_y)), cell_size // 3)
            
            # Render Ghost (blue when vulnerable, red otherwise)
            ghost_x, ghost_y = self.ghost_position
            ghost_center_x = (ghost_x + 0.5) * cell_size
            ghost_center_y = (ghost_y + 0.5) * cell_size
            ghost_color = (0, 0, 255) if self.ghost_is_vulnerable else (255, 0, 0)
            pygame.draw.circle(self.screen, ghost_color,
                             (int(ghost_center_x), int(ghost_center_y)), cell_size // 2)
            
            # Render Pac-Man (yellow)
            pacman_x, pacman_y = self.pacman_position
            center_x = (pacman_x + 0.5) * cell_size
            center_y = (pacman_y + 0.5) * cell_size
            pygame.draw.circle(self.screen, (255, 255, 0),
                             (int(center_x), int(center_y)), cell_size // 2)
            
            # Show step counter and food remaining
            font = pygame.font.SysFont(None, 24)
            step_text = font.render(f"Steps: {self.steps_counter}/{self.MAX_STEPS}", True, (255, 255, 255))
            food_text = font.render(f"Food: {len(self.food_positions)}/{self.initial_food_count}", True, (255, 255, 255))
            status_text = font.render(f"Vulnerable: {'Yes' if self.ghost_is_vulnerable else 'No'}", True, (255, 255, 255))
            progress = 100 * (1.0 - len(self.food_positions) / self.initial_food_count)
            progress_text = font.render(f"Progress: {progress:.1f}%", True, (255, 255, 255))
            
            self.screen.blit(step_text, (10, 10))
            self.screen.blit(food_text, (10, 30))
            self.screen.blit(status_text, (10, 50))
            self.screen.blit(progress_text, (10, 70))
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """Clean up resources"""
        if self.render_mode == "human":
            pygame.quit()
    
    def _create_walls(self):
        """Create a simple maze layout"""
        walls = set()
        
        # Outer walls (with tunnel openings)
        for i in range(self.grid_size[0]):
            # Skip tunnel entrances
            if i != 0 and i != self.grid_size[0]-1:
                walls.add((i, 0))  # Top wall
                walls.add((i, self.grid_size[1]-1))  # Bottom wall
        
        for j in range(self.grid_size[1]):
            # Skip middle for tunnels
            if j != self.grid_size[1]//2:
                walls.add((0, j))  # Left wall
                walls.add((self.grid_size[0]-1, j))  # Right wall
        
        # Add some inner walls for a simple maze
        # Vertical walls
        for j in range(3, 7):
            walls.add((3, j))  # Left vertical wall
            walls.add((11, j))  # Right vertical wall
        
        for j in range(8, 12):
            walls.add((3, j))  # Left vertical wall
            walls.add((11, j))  # Right vertical wall
        
        # Horizontal walls
        for i in range(4, 11):
            walls.add((i, 3))  # Top horizontal wall
            walls.add((i, 11))  # Bottom horizontal wall
        
        # Ghost house (simplified)
        ghost_house = [
            (6, 6), (6, 7), (6, 8),  # Left wall
            (7, 8),                   # Bottom wall
            (8, 6), (8, 7), (8, 8)    # Right wall
        ]
        
        for pos in ghost_house:
            walls.add(pos)
        
        return walls
    
    def _initialize_game(self):
      """Initialize game state"""
      # Set initial positions
      self.pacman_position = (self.grid_size[0] // 2, self.grid_size[1] - 2)
      self.ghost_position = (7, 7)  # Ghost starts in the ghost house

      # Reset state
      self.ghost_is_vulnerable = False
      self.power_pellet_timer = 0
      
      # Initialize food (everywhere except walls, ghost house, and Pac-Man position)
      self.food_positions = set()
      for x in range(1, self.grid_size[0] - 1):
          for y in range(1, self.grid_size[1] - 1):
              if (x, y) not in self.walls and (x, y) != self.pacman_position and (x, y) != self.ghost_position:
                  # Skip ghost house interior
                  if not (6 < x < 8 and 6 < y < 8):
                      self.food_positions.add((x, y))
      
      # Set power pellets in corners
      self.power_pellets = {
          (1, 1),  # Top-left
          (1, self.grid_size[1] - 2),  # Bottom-left
          (self.grid_size[0] - 2, 1),  # Top-right
          (self.grid_size[0] - 2, self.grid_size[1] - 2)  # Bottom-right
      }
      # Save the initial power pellet count for tracking
      self.initial_power_pellet_count = len(self.power_pellets)
      self.initial_food_count = len(self.food_positions)