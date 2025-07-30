"""
data.py  –  Data loading utilities for the FL‑IDS & RL‑IDS projects
==================================================================

Key features
------------
* **Single load / global split:**  UNSW‑NB15 CSV corpus is read **once**,
  then split 80 / 20 into a *train pool* and a *global hold‑out* test set.
* **Memory‑friendly preprocessing:** keep only numeric columns (≈ 40 features);
  NaNs are median‑imputed and standard‑scaled.
* **Dirichlet label‑skew partitioner:** a single α hyper‑parameter controls
  how non‑IID each client shard is.  α→∞ ⇒ IID ; α→0.1 ⇒ highly skewed.
* **Optional SMOTE:** class‑balanced oversampling *per client shard* can be
  toggled with `USE_SMOTE`.
* Provides **DataLoader helpers** used by both Flower clients and the server.

Import‑side‑effect free: nothing expensive happens at import time except the
initial dataset load.  Run `python data.py` to see dataset statistics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 0.  Globals / user‑configurable constants
# ---------------------------------------------------------------------------

SEED: int = 42
DATA_DIR: Path = Path("dataset/10%")     # folder containing UNSW‑NB15 *.csv
BATCH_SIZE: int = 256
ALPHA_DIRICHLET: float = 1000             # lower = more skew
USE_SMOTE: bool = False                  # toggle oversampling

# rng used **everywhere** to keep determinism
RNG: np.random.Generator = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1.  CSV loading helper
# ---------------------------------------------------------------------------


def _load_csv_folder(folder: Path) -> pd.DataFrame:
    """Read every *.csv under *folder* (non‑recursively) into one DataFrame."""
    paths = sorted(Path(folder).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {folder!s}")
    dfs = [pd.read_csv(p, low_memory=False) for p in paths]
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# 2.  Pre‑processing – numeric‑only pipeline
# ---------------------------------------------------------------------------


def _preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Return X (float32), y (int64), class_names after basic cleaning.

    Steps
    -----
    1. Replace ±∞ → NaN.
    2. Fill missing/empty labels with "normal".
    3. Keep only numeric columns (≈ 40 features).
    4. Median‑impute NaNs and standard‑scale.
    5. Label‑encode `attack_cat`.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    df["attack_cat"] = df["attack_cat"].fillna("normal").replace("", "normal")

    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Impute → scale
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    scaler = StandardScaler()
    X = scaler.fit_transform(df[num_cols]).astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(df["attack_cat"].astype(str)).astype(np.int64)
    return X, y, le.classes_.tolist()


# ---------------------------------------------------------------------------
# 3.  Load full dataset exactly once, then split 80/20
# ---------------------------------------------------------------------------


def _load_and_split(
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df_raw = _load_csv_folder(DATA_DIR)
    print(f"[data.py] Raw rows before preprocess: {len(df_raw):,}")

    X_full, y_full, class_names = _preprocess(df_raw)
    print(
        f"[data.py] Rows after preprocess: {len(X_full):,} "
        f"(features={X_full.shape[1]})"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=test_size,
        stratify=y_full,
        random_state=SEED,
    )
    return X_train, y_train, X_test, y_test, class_names


# materialise at import time
(
    X_TRAIN_POOL,
    Y_TRAIN_POOL,
    X_TEST_GLOBAL,
    Y_TEST_GLOBAL,
    CLASS_NAMES,
) = _load_and_split()
print(
    f"[data.py] Global split ➜ train={len(X_TRAIN_POOL):,}  "
    f"test={len(X_TEST_GLOBAL):,}"
)

# ---------------------------------------------------------------------------
# 4.  Dirichlet label‑skew partitioner
# ---------------------------------------------------------------------------


def _dirichlet_label_split(
    y: np.ndarray, num_clients: int, alpha: float, rng: np.random.Generator
) -> List[np.ndarray]:
    """Return list of index arrays, one per client, with Dirichlet label skew."""
    cls_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(y):
        cls_indices[int(label)].append(idx)

    client_shards: list[list[int]] = [[] for _ in range(num_clients)]
    for idxs in cls_indices.values():
        rng.shuffle(idxs)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        split_pts = (np.cumsum(proportions) * len(idxs)).astype(int)
        shards = np.split(idxs, split_pts[:-1])
        for cid, shard in enumerate(shards):
            client_shards[cid].extend(shard)

    return [np.array(s, dtype=np.int32) for s in client_shards]


# ---------------------------------------------------------------------------
# 5.  Public Loader API used by server & clients
# ---------------------------------------------------------------------------


def load_global_test(batch_size: int = BATCH_SIZE) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X_TEST_GLOBAL), torch.from_numpy(Y_TEST_GLOBAL)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_partition(
    cid: int,
    num_clients: int,
    batch_size: int = BATCH_SIZE,
    alpha: float = ALPHA_DIRICHLET,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for client *cid* under Dirichlet skew."""
    # Build shards once per call – cheap, uses only label vector
    client_indices = _dirichlet_label_split(Y_TRAIN_POOL, num_clients, alpha, RNG)
    part = client_indices[cid]
    X_part = X_TRAIN_POOL[part]
    y_part = Y_TRAIN_POOL[part]

    # ---------- optional per‑class SMOTE oversampling ------------------------
    if USE_SMOTE:
        X_part, y_part = _selective_smote_per_class(X_part, y_part, seed=SEED)

    # 80/20 validation inside the shard
    stratify = y_part if len(np.unique(y_part)) > 1 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_part,
        y_part,
        test_size=0.2,
        stratify=stratify,
        random_state=SEED,
    )

    print(
        f"[Client {cid}] α={alpha}  » train {len(X_tr):,}  "
        f"val {len(X_val):,}  classes {len(np.unique(y_tr))}"
    )

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


# ---------------------------------------------------------------------------
# 6.  SMOTE helper (usable by RL & supervised clients)
# ---------------------------------------------------------------------------


def _selective_smote_per_class(
    X: np.ndarray,
    y: np.ndarray,
    target_fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample *each minority class independently* **only if**
    it has at least (k + 1) real samples.  Classes with too few
    rows are left unchanged.
    """
    counts = Counter(y)
    majority = max(counts.values())
    candidates = {cls: n for cls, n in counts.items() if 1 < n < majority}

    if not candidates:
        return X, y  # nothing to oversample

    target = {
        cls: max(int(target_fraction * majority), n + 1) for cls, n in candidates.items()
    }

    min_n = min(candidates.values())
    k = max(1, min(5, min_n - 1))  # SMOTE requires k < min_n

    smote = BorderlineSMOTE(
        sampling_strategy=target, k_neighbors=k, random_state=seed
    )
    return smote.fit_resample(X, y)


# ---------------------------------------------------------------------------
# 7.  Diagnostics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_CLIENTS = 4
    print("\n=== Diagnostics ===")
    shards = _dirichlet_label_split(Y_TRAIN_POOL, NUM_CLIENTS, ALPHA_DIRICHLET, RNG)
    for cid, shard in enumerate(shards):
        unique, counts = np.unique(Y_TRAIN_POOL[shard], return_counts=True)
        print(
            f" Client {cid}: {len(shard):,} rows  –  "
            f"class counts = dict({dict(zip(unique, counts))})"
        )
