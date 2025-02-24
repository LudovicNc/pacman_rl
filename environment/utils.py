import pickle
import numpy as np
import torch

def save_model(model, filename):
    """Save a trained model (Q-table or neural network)."""
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load a saved model."""
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_torch_model(model, filename):
    """Save a PyTorch model."""
    torch.save(model.state_dict(), filename)

def load_torch_model(model, filename):
    """Load a PyTorch model."""
    model.load_state_dict(torch.load(filename))
    model.eval()

def discretize_state(state, grid_size=(10, 10)):
    """Convert a continuous state into a discrete index for Q-learning."""
    return np.ravel_multi_index(state[:2], grid_size)
