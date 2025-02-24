import time
from environment.pacman_env import PacmanEnv
import pygame
import sys

def main():
    # Create environment instance
    env = PacmanEnv(render_mode="human")
    
    try:
        # Reset environment
        state, _ = env.reset()
        print("Environment loaded successfully")

        running = True
        step = 0
        
        while running and step < 100:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            if not running:
                break

            action = env.action_space.sample()  # Choose a random action
            state, reward, done, _, _ = env.step(action)

            env.render()  # Ensure rendering happens every step
            time.sleep(0.1)  # Add a small delay to visualize movement

            if done:  # Stop if the game ends
                print("Game over after", step + 1, "steps")
                break
                
            step += 1

    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        env.close()
        print("Game closed properly")
        sys.exit()

if __name__ == "__main__":
    main()
