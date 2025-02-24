import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class PacmanEnv(gym.Env):
    """
    Custom Pac-Man Environment for Reinforcement Learning using Gymnasium & Pygame
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode="human"):
        super().__init__()
        
        self.grid_size = (15, 15)  # Increased grid size for better maze layout
        self.walls = self._create_walls()
        self.state = None
        self.done = False
        self.power_pellets = set()  # New attribute for power pellets
        self.ghost_is_vulnerable = False
        self.power_pellet_timer = 0
        self.POWER_PELLET_DURATION = 50  # Duration in steps
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(15, 15, 3), dtype=np.uint8)
        
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.window_size = 600
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Pac-Man RL")  # Add window title
    
    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        
        # Use the new initialization method
        self._initialize_game()
        
        # Generate initial state
        self.state = self._generate_state_representation()
        self.done = False
        
        return self.state, {}

    
    def step(self, action):
        reward = 0
        
        # Update power pellet timer
        if self.ghost_is_vulnerable:
            self.power_pellet_timer -= 1
            if self.power_pellet_timer <= 0:
                self.ghost_is_vulnerable = False
        
        # Get current position
        pacman_x, pacman_y = self.pacman_position
        
        # Store new position temporarily
        new_x, new_y = pacman_x, pacman_y
        
        # Define movement logic with wraparound
        if action == 0 and pacman_y > 0:  # Up
            new_y -= 1
        elif action == 1 and pacman_y < self.grid_size[1] - 1:  # Down
            new_y += 1
        elif action == 2:  # Left
            new_x = (pacman_x - 1) % self.grid_size[0]
        elif action == 3:  # Right
            new_x = (pacman_x + 1) % self.grid_size[0]

        # Check if new position hits a wall
        if (new_x, new_y) not in self.walls:
            self.pacman_position = (new_x, new_y)
        else:
            reward -= 5  # Penalty for hitting wall

        # Check for power pellet collection
        if self.pacman_position in self.power_pellets:
            self.power_pellets.remove(self.pacman_position)
            self.ghost_is_vulnerable = True
            self.power_pellet_timer = self.POWER_PELLET_DURATION
            reward += 20  # Bonus for getting power pellet
        
        # Update ghost position
        self._move_ghost()
        
        # Check for ghost collision
        if self.pacman_position == self.ghost_position:
            if self.ghost_is_vulnerable:
                # Ghost is eaten
                self.ghost_position = (5, 5)  # Reset ghost to center
                reward += 100  # Large bonus for eating ghost
                self.ghost_is_vulnerable = False  # Ghost respawns as normal
            else:
                # Pac-Man is eaten
                self.done = True
                reward -= 50

        # Add distance-based reward when ghost is vulnerable
        if self.ghost_is_vulnerable:
            ghost_x, ghost_y = self.ghost_position
            pacman_x, pacman_y = self.pacman_position
            distance = abs(ghost_x - pacman_x) + abs(ghost_y - pacman_y)
            # Small reward for being close to vulnerable ghost
            reward += max(0, (10 - distance)) 

        # Recalculate state representation
        self.state = self._generate_state_representation()

        # Assign rewards
        reward -= 1  # Small penalty for each step
        if self.pacman_position in self.food_positions:
            reward += 10
            self.food_positions.remove(self.pacman_position)

        # Check if game is over
        if self._check_game_end():
            self.done = True
            if not self.food_positions:  # Won
                reward += 50
            elif self.pacman_position == self.ghost_position:  # Lost
                reward -= 50

        return self.state, reward, self.done, False, {}

    def _move_ghost(self):
        """Simplified ghost AI with reliable house exit"""
        ghost_x, ghost_y = self.ghost_position
        pacman_x, pacman_y = self.pacman_position
        
        # Define ghost house area
        in_ghost_house = (4 <= ghost_x <= 6 and 5 <= ghost_y <= 6)  # Adjusted area
        
        if in_ghost_house and not self.ghost_is_vulnerable:
            # Simple upward movement until out of house
            if ghost_y > 4:  # Move up until reaching the exit area
                new_y = ghost_y - 1
                if (ghost_x, new_y) not in self.walls:
                    self.ghost_position = (ghost_x, new_y)
            elif ghost_x != 5:  # Center horizontally if not already
                new_x = 5
                self.ghost_position = (new_x, ghost_y)
        else:
            # Normal ghost movement once outside
            dx = ((pacman_x - ghost_x + self.grid_size[0]//2) % self.grid_size[0]) - self.grid_size[0]//2
            dy = pacman_y - ghost_y
            
            # Reverse direction if vulnerable
            if self.ghost_is_vulnerable:
                dx = -dx
                dy = -dy
            
            # Random movement with increased randomness when vulnerable
            random_threshold = 0.4 if self.ghost_is_vulnerable else 0.2
            if np.random.random() < random_threshold:
                possible_moves = []
                for move_dx, move_dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    new_x = (ghost_x + move_dx) % self.grid_size[0]
                    new_y = ghost_y + move_dy
                    # Allow movement anywhere except walls and ghost house bottom
                    if (0 <= new_y < self.grid_size[1] and 
                        (new_x, new_y) not in self.walls):
                        possible_moves.append((new_x, new_y))
                if possible_moves:
                    self.ghost_position = possible_moves[np.random.randint(len(possible_moves))]
            else:
                # Move towards/away from Pac-Man
                if abs(dx) > abs(dy):
                    new_x = (ghost_x + (1 if dx > 0 else -1)) % self.grid_size[0]
                    if (new_x, ghost_y) not in self.walls:
                        self.ghost_position = (new_x, ghost_y)
                else:
                    new_y = ghost_y + (1 if dy > 0 else -1)
                    if (0 <= new_y < self.grid_size[1] and 
                        (ghost_x, new_y) not in self.walls):
                        self.ghost_position = (ghost_x, new_y)

    def _generate_initial_state(self):
        """Initialize the environment state with Pac-Man, food, and ghosts."""
        state = np.zeros((10, 10, 3), dtype=np.uint8)  # Create an empty grid

        # Place Pac-Man in the center
        self.pacman_position = (5, 5)
        state[5, 5] = [255, 255, 0]  # Yellow color for Pac-Man

        # Place random food in the grid
        for _ in range(10):  # Example: 10 food items
            x, y = np.random.randint(0, 10, size=2)
            state[x, y] = [255, 255, 255]  # White color for food

        # Place a ghost (example, modify as needed)
        self.ghost_position = (2, 2)
        state[2, 2] = [255, 0, 0]  # Red color for a ghost

        return state
    
    def _generate_state_representation(self):
        """Generate the state representation as a 10x10 grid."""
        state = np.zeros((15, 15, 3), dtype=np.uint8)  # Empty grid

        # Add Pac-Man
        pacman_x, pacman_y = self.pacman_position
        state[pacman_x, pacman_y] = [255, 255, 0]  # Yellow color for Pac-Man

        # Add Ghost
        ghost_x, ghost_y = self.ghost_position
        state[ghost_x, ghost_y] = [255, 0, 0]  # Red color for Ghost

        # Add Food
        for x, y in self.food_positions:
            state[x, y] = [255, 255, 255]  # White color for Food

        # Add Power Pellets
        for x, y in self.power_pellets:
            state[x, y] = [255, 255, 255]  # White color for Power Pellets

        return state
    
    def render(self):
        if self.render_mode == "human":
            # Handle window close event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            self.screen.fill((0, 0, 0))
            
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
                                 (center_x, center_y), cell_size // 6)
            
            # Render Power Pellets (larger white dots)
            for pellet_x, pellet_y in self.power_pellets:
                center_x = (pellet_x + 0.5) * cell_size
                center_y = (pellet_y + 0.5) * cell_size
                pygame.draw.circle(self.screen, (255, 255, 255),
                                 (center_x, center_y), cell_size // 3)
            
            # Render Ghost (blue when vulnerable, red otherwise)
            ghost_x, ghost_y = self.ghost_position
            ghost_center_x = (ghost_x + 0.5) * cell_size
            ghost_center_y = (ghost_y + 0.5) * cell_size
            ghost_color = (0, 0, 255) if self.ghost_is_vulnerable else (255, 0, 0)
            pygame.draw.circle(self.screen, ghost_color,
                             (ghost_center_x, ghost_center_y), cell_size // 2)
            
            # Render Pac-Man (yellow)
            pacman_x, pacman_y = self.pacman_position
            center_x = (pacman_x + 0.5) * cell_size
            center_y = (pacman_y + 0.5) * cell_size
            pygame.draw.circle(self.screen, (255, 255, 0),
                             (center_x, center_y), cell_size // 2)
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
    def _check_game_end(self):
        """Check if the game has ended."""
        # Condition 1: All food is eaten (Pac-Man wins)
        if not self.food_positions:
            print("Pac-Man ate all the food! Game Over: Victory")
            return True

        # Condition 2: Pac-Man collides with a ghost (Pac-Man loses)
        if self.pacman_position == self.ghost_position:
            print("Pac-Man was caught by a ghost! Game Over: Defeat")
            return True

        return False  # Game continues

    
    def close(self):
        """Clean up resources and close pygame window properly"""
        if self.render_mode == "human":
            try:
                pygame.display.quit()
                pygame.quit()
                import sys
                sys.exit()
            except Exception as e:
                print(f"Error while closing: {e}")

    def _create_walls(self):
        """Create a classic Pac-Man style maze layout with ghost house"""
        walls = set()
        
        # Outer walls (excluding the tunnel positions)
        tunnel_y = self.grid_size[1] // 2
        
        # Add outer walls with tunnel gaps
        for i in range(self.grid_size[0]):
            if i not in [0, self.grid_size[0]-1]:  # Skip tunnel entrances
                walls.add((i, 0))  # Top wall
                walls.add((i, self.grid_size[1]-1))  # Bottom wall
        for j in range(self.grid_size[1]):
            if j != tunnel_y:  # Skip tunnel row
                walls.add((0, j))  # Left wall
                walls.add((self.grid_size[0]-1, j))  # Right wall
        
        # Inner vertical walls - modified to not block tunnel access
        vertical_walls = [
            (2, 2, tunnel_y-1),  # Stop before tunnel
            (4, 2, tunnel_y-1),
            (7, 2, tunnel_y-1),
            (2, tunnel_y+1, 8),  # Start after tunnel
            (4, tunnel_y+1, 8),
            (7, tunnel_y+1, 8),
        ]
        
        for x, start_y, end_y in vertical_walls:
            for y in range(start_y, end_y):
                walls.add((x, y))
                mirror_x = self.grid_size[0] - 1 - x
                walls.add((mirror_x, y))
        
        # Inner horizontal walls
        horizontal_walls = [
            #(2, 4, 2, 7),  # (start_x, end_x, y, length)
            (2, 4, 8, 7),
            (4, 7, 5, 3),
        ]
        
        for start_x, end_x, y, length in horizontal_walls:
            for x in range(start_x, start_x + length):
                walls.add((x, y))
                mirror_y = self.grid_size[1] - 1 - y
                walls.add((x, mirror_y))
        
        # Ghost house (center box) - with clear exit path
        ghost_house = [
            (4, 5), (4, 6),  # Left wall (shorter)
            (5, 6),          # Bottom wall only
            (6, 5), (6, 6)   # Right wall (shorter)
        ]
        for x, y in ghost_house:
            walls.add((x, y))
        
        return walls

    def _initialize_game(self):
        """Initialize game state with ghost slightly higher in house"""
        self.pacman_position = (self.grid_size[0] // 2, self.grid_size[1] - 3)
        self.ghost_position = (5, 5)  # Start ghost in middle of house
        self.ghost_is_vulnerable = False
        
        # Initialize food positions (everywhere except walls and ghost house)
        self.food_positions = set()
        for x in range(1, self.grid_size[0] - 1):
            for y in range(1, self.grid_size[1] - 1):
                if (x, y) not in self.walls and (x, y) != self.pacman_position:
                    # Skip ghost house area
                    if not (4 <= x <= 6 and 4 <= y <= 6):
                        self.food_positions.add((x, y))
        
        # Add power pellets in corners
        self.power_pellets = {
            (1, 1), 
            (1, self.grid_size[1] - 2),
            (self.grid_size[0] - 2, 1),
            (self.grid_size[0] - 2, self.grid_size[1] - 2)
        }
        self.food_positions -= self.power_pellets
