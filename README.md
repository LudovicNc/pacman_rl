# Pac-Man Reinforcement Learning Project

## Overview
This project implements Q-learning to train an agent to play a simplified Pac-Man environment. The environment is built using **Gymnasium** and **Pygame**, featuring ghost movements, power pellets, and wraparound tunnels.

## Folder Structure
```
pacman_rl_project/
│── environment/          # Custom Pac-Man Gym environment
│   ├── pacman_env.py     # Environment implementation
│
│── agents/               # RL Agent
│   ├── q_learning.py     # Q-learning agent implementation
│
│── evaluation/           # Evaluation & visualization
│   ├── plot_results.py   # Visualize learning curves
│
│── plots/                # Saved training plots
│
│── metrics/              # Training metrics
│
│── requirements.txt      # Dependencies
│── README.md             # Documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pacman_rl_project.git
   cd pacman_rl_project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project
### **1. Testing the Environment**
To test the Pac-Man environment:
```bash
python test_env.py
```

### **2. Training the Agent**
To train the Q-learning agent:
```bash
python train.py
```

### **3. Viewing Results**
Training metrics and plots are automatically saved to:
- `plots/` directory: Learning curves and performance metrics
- `metrics/` directory: Detailed JSON files with training data

## Features
- Custom Pac-Man environment with:
  - Ghost that chases Pac-Man
  - Power pellets that make ghosts vulnerable
  - Wraparound tunnels
  - Classic maze layout
- Q-learning agent that learns to:
  - Collect food pellets
  - Avoid ghosts
  - Use power pellets strategically
  - Chase vulnerable ghosts

## Dependencies
- gymnasium
- pygame
- numpy
- matplotlib

## License
This project is open-source and free to use.

