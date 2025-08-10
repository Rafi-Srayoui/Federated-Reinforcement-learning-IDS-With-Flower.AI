# Federated Reinforcement Learning for Intrusion Detection (FL-RL-IDS)

This project implements a **Federated Learning (FL)**-based **Reinforcement Learning (RL)** Intrusion Detection System (IDS) using the [Flower](https://flower.dev) framework and Double Deep Q-Networks (DDQN). It operates across multiple simulated clients, each training on non-IID subsets of the UNSW-NB15 dataset, and aggregates learned policies on a central server.

---

## 🚀 Overview

Traditional supervised IDSs rely on large, centrally collected labeled datasets. This project explores an alternative paradigm:

- **Federated Setup**: Clients never share raw data—only model weights.
- **Reinforcement Learning**: Each client treats network traffic as an environment and learns a classification policy using Deep Q-Learning.
- **Non-IID Data**: Client datasets are label-skewed using a Dirichlet distribution (α = 1000).

---

## 📁 Project Structure

FL_RL/

├── rl_fl_client.py # Federated RL client using DQN + replay buffer

├── rl_fl_server.py # Central FL server with Flower + evaluation logic

├── rl_model.py # DQN architecture, replay buffer, FL parameter utilities

├── data.py # UNSW-NB15 data preprocessing, partitioning, loading

├── run_fl_ids.bat # Windows launcher script to start clients/server

├── dataset/ # Folder containing UNSW-NB15 CSV files (10% subset)

└── Results.docx # Evaluation results from a 50-round training session

---

## 🧠 Model Architecture

Each client runs a **DQN** model:

- Input: 41 numeric features (preprocessed)
- MLP: `128 → 128 → Num_Classes`
- Output: Q-values for each attack category (10 classes)

Training uses ε-greedy exploration and experience replay.

---

## 🧪 Dataset: UNSW-NB15

- Preprocessed to retain only numeric features
- 80/20 train-test global split
- Client partitions use Dirichlet label-skewing (`α = 1000`)
- Optional per-client SMOTE oversampling available (`USE_SMOTE = False`)

---

## ⚙️ How It Works

1. **Clients**:
   - Train local DDQN agents with episodic updates.
   - Share updated Q-network weights with the server.
   - Evaluate locally on held-out validation data.

2. **Server**:
   - Aggregates client weights using FedAvg.
   - Evaluates aggregated model on a global test set.
   - Tracks macro/micro metrics and per-class performance.

---

## 📝 Results Summary (from `Results.docx`)

- **Rounds**: 50
- **Duration**: ~2h 11min
- **Global Accuracy**: 84.96%
- **Global F1 Score (macro)**: 0.481
- **Convergence**: Loss flattens after ~35 rounds

**Per-Class Highlights**:
- Perfect recall for class 6 (DoS)
- Strong F1 for classes 4–7
- Poor detection for rare classes (0, 1, 9)

**Suggestions for Future Runs**:
- Class-balanced loss or oversampling
- Data augmentation for minority classes
- Early stopping around round 45
- Adaptive client sampling (bias toward minority-heavy shards)

---

## ▶️ Running the Project

### 1. Requirements

Install dependencies (Python ≥ 3.8):

```bash
pip install flwr torch scikit-learn imbalanced-learn matplotlib pandas

Launch Clients & Server
Use the batch script (run_fl_ids.bat) or manually run:

# Terminal 1: Server
python rl_fl_server.py

# Terminal 2+: Clients
python rl_fl_client.py 0
python rl_fl_client.py 1
python rl_fl_client.py 2
python rl_fl_client.py 3

📊 Visualizations
The server produces plots for:

Global Accuracy, Precision, Recall, F1 over time

Per-Class F1, Precision, Recall

These help analyze convergence and class imbalance effects.
