# model.py  –  DQN, replay buffer, and Flower helpers
# ---------------------------------------------------
from collections import deque
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------
# 1. Deep Q‑Network (simple 2‑hidden‑layer MLP)
# ---------------------------------------------------
class DQN(nn.Module):
    """Fully‑connected network returning Q‑values for each action/class."""

    def __init__(self, input_dim: int, num_actions: int,
                 hidden_dims: List[int] = (128, 128)):
        super().__init__()
        layers = []
        dims = (input_dim, *hidden_dims, num_actions)
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            if out_d != num_actions:          # no activation on last layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------
# 2. Replay Buffer
# ---------------------------------------------------
class ReplayBuffer:
    """Cyclic buffer storing (state, action, reward, next_state) tuples."""

    def __init__(self, max_size: int = 10_000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions,          dtype=torch.long),
            torch.tensor(rewards,          dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------
# 3. Flower parameter utilities
# ---------------------------------------------------
def get_parameters(model: nn.Module):
    """Extract parameters as a list of NumPy arrays (for Flower)."""
    return [p.cpu().detach().numpy() for p in model.state_dict().values()]


def set_parameters(model: nn.Module, params) -> None:
    """Load parameters (received from Flower) into the model in‑place."""
    state_dict = model.state_dict()
    for (key, _), param in zip(state_dict.items(), params):
        state_dict[key] = torch.tensor(param)
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------
# 4. Simple reward function
# ---------------------------------------------------
def get_reward(pred_action: int, true_label: int) -> int:
    """Return +1 if the predicted class equals the ground‑truth label else –1."""
    return 1 if pred_action == true_label else -1


# ---------------------------------------------------
# (Optional) quick smoke‑test
# ---------------------------------------------------
if __name__ == "__main__":
    inp, n_classes = 41, 10
    net = DQN(inp, n_classes)
    dummy = torch.randn(3, inp)
    print("Q‑values:", net(dummy))           # [3 × n_classes] logits
    print("Parameters:", sum(p.numel() for p in net.parameters()))
    rb = ReplayBuffer(max_size=5)
    for i in range(7):                       # push > max_size to test eviction
        rb.push(np.zeros(inp), 0, 1, np.ones(inp))
    print("Replay length:", len(rb))
    s, a, r, ns = rb.sample(2)
    print("Sample shapes:", s.shape, a.shape, r.shape, ns.shape)
