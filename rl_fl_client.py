# client.py  –  Federated DQN IDS client
# ------------------------------------------------------------
# Usage:  python client.py <cid>   (cid = 0, 1, … NUM_CLIENTS‑1)

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from data      import load_partition                      # existing loader
from rl_model  import DQN, ReplayBuffer, get_parameters, set_parameters

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ------------------------------------------------------------------
# Minimal inline "environment" for tabular data
# ------------------------------------------------------------------
class TabularBatchEnv:
    def __init__(self, X: np.ndarray, y: np.ndarray, *, shuffle: bool = True):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.idx = np.arange(len(X))
        self.ptr = 0

    def reset_episode(self):
        if self.shuffle:
            np.random.shuffle(self.idx)
        self.ptr = 0
        return self.X[self.idx[self.ptr]], self.y[self.idx[self.ptr]]

    def step(self):
        self.ptr += 1
        done = self.ptr >= len(self.X)
        next_state = self.X[self.idx[min(self.ptr, len(self.X)-1)]]
        next_label = self.y[self.idx[min(self.ptr, len(self.X)-1)]]
        return next_state, next_label, done


def get_reward(pred_action: int, true_label: int) -> int:
    """+1 if correct class, else –1."""
    return 1 if pred_action == true_label else -1


# ------------------------------------------------------------
# Hyper‑params / constants
# ------------------------------------------------------------
NUM_CLIENTS            = 4     # keep in sync with server.py
EPISODES_PER_ROUND     = 1     # one sweep over the local shard per FL round
BATCH_SIZE_REPLAY      = 64
REPLAY_CAPACITY        = 10_000
MIN_REPLAY_BEFORE_TRAIN= 64
GAMMA                  = 0.99
EPSILON                = 0.10   # fixed ε‑greedy; schedule later if you like
TARGET_SYNC_INTERVAL   = 1      # hard copy after each episode
LR                     = 1e-3
DEVICE                 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Data loaders & env
# ------------------------------------------------------------
cid = int(sys.argv[1])                          # e.g. python client.py 0
train_loader, val_loader = load_partition(cid, NUM_CLIENTS)

INPUT_DIM   = train_loader.dataset.tensors[0].shape[1]
NUM_CLASSES = len(torch.unique(train_loader.dataset.tensors[1]))
print(f"[Client {cid}] features={INPUT_DIM}  classes={NUM_CLASSES}")

# Build RL components
qnet        = DQN(INPUT_DIM, NUM_CLASSES).to(DEVICE)
target_qnet = DQN(INPUT_DIM, NUM_CLASSES).to(DEVICE)
target_qnet.load_state_dict(qnet.state_dict())
optimizer   = optim.Adam(qnet.parameters(), lr=LR)
replay      = ReplayBuffer(max_size=REPLAY_CAPACITY)

# We treat the tensor dataset as a deterministic “episode”
X_local = train_loader.dataset.tensors[0].cpu().numpy()
y_local = train_loader.dataset.tensors[1].cpu().numpy()
env     = TabularBatchEnv(X_local, y_local, shuffle=True)

# ------------------------------------------------------------
# Local RL update
# ------------------------------------------------------------
def rl_local_update() -> None:
    """Run EPISODES_PER_ROUND episodes of ε‑greedy DQN on the local shard."""
    qnet.train()
    for ep in range(EPISODES_PER_ROUND):
        state, label = env.reset_episode()
        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE)

            # ε‑greedy action selection
            if np.random.rand() < EPSILON:
                action = np.random.randint(NUM_CLASSES)
            else:
                with torch.no_grad():
                    action = qnet(state_t).argmax().item()

            reward = get_reward(action, int(label))          # +1 / –1

            next_state, next_label, done = env.step()
            replay.push(state, action, reward, next_state)

            # Train once replay has enough samples
            if len(replay.buffer) >= MIN_REPLAY_BEFORE_TRAIN:
                s_b, a_b, r_b, ns_b = replay.sample(BATCH_SIZE_REPLAY)
                s_b  = s_b.to(DEVICE)
                a_b  = a_b.to(DEVICE)
                r_b  = r_b.to(DEVICE)
                ns_b = ns_b.to(DEVICE)

                q_values   = qnet(s_b)
                q_selected = q_values.gather(1, a_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_qnet(ns_b).max(1)[0]
                    target = r_b + GAMMA * next_q

                loss = nn.MSELoss()(q_selected, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state, label = next_state, next_label

        # Sync target net
        if (ep + 1) % TARGET_SYNC_INTERVAL == 0:
            target_qnet.load_state_dict(qnet.state_dict())

# ------------------------------------------------------------
# Metric helpers (reuse supervised pattern)
# ------------------------------------------------------------
def collect_preds(loader):
    qnet.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            preds = torch.argmax(qnet(xb), dim=1)
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)

# ------------------------------------------------------------
# Flower client
# ------------------------------------------------------------
class RLFedClient(fl.client.NumPyClient):

    # ---------- required 3 methods -----------------------------------------
    def get_parameters(self, config):
        return get_parameters(qnet)

    def fit(self, params, config):
        # 1) receive global weights → load into both nets
        set_parameters(qnet, params)
        target_qnet.load_state_dict(qnet.state_dict())

        # 2) local RL training
        rl_local_update()

        # 3) local validation metrics
        y_true, y_pred = collect_preds(val_loader)
        metrics = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
        # Per‑class
        for cls in range(NUM_CLASSES):
            y_true_bin = (y_true == cls).astype(int)
            y_pred_bin = (y_pred == cls).astype(int)
            metrics[f"class_{cls}_prec"] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            metrics[f"class_{cls}_rec"]  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            metrics[f"class_{cls}_f1"]   = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        print(f"[Client {cid}] round metrics: "
              f"acc={metrics['accuracy']:.3f}  prec={metrics['precision']:.3f}  "
              f"rec={metrics['recall']:.3f}  f1={metrics['f1_score']:.3f}")

        return get_parameters(qnet), len(train_loader.dataset), metrics

    def evaluate(self, params, config):
        set_parameters(qnet, params)
        y_true, y_pred = collect_preds(val_loader)
        loss = 1.0 - accuracy_score(y_true, y_pred)          # simple surrogate
        metrics = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_score":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        }
        return loss, len(val_loader.dataset), metrics

# ------------------------------------------------------------
if __name__ == "__main__":
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=RLFedClient().to_client(),   # NumPyClient → native Client
    )
