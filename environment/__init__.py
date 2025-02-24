# Makes the environment a package for easy imports
from .pacman_env import PacmanEnv
from gymnasium.envs.registration import register

register(
    id="PacmanEnv-v0",
    entry_point="environment.pacman_env:PacmanEnv",
)
