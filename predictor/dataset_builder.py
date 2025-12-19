"""Dataset construction for the acceptance-rate predictor.

Loads per-token JSONL run logs produced by the speculative engine,
engineers ML-ready features, and returns a clean DataFrame suitable
for supervised classification.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Features used by the predictor models.  Order matters for column alignment.
NUMERIC_FEATURES = [
    "p_draft",
    "draft_log_prob",
    "draft_entropy",
    "draft_top5_mass",
    "confidence_margin",
    "kl_divergence",
    "cross_entropy",
    "position",
    "rolling_acceptance",
    "token_length",
]

BOOLEAN_FEATURES = [
    "is_punctuation",
    "is_whitespace",
]

DOMAIN_COLUMNS: List[str] = []  # filled dynamically after one-hot encoding

TARGET_COL = "accepted"


def _load_jsonl_files(results_dir: Path) -> List[Dict]:
    """Load all run_*.jsonl files from a directory."""
    records: List[Dict] = []
    jsonl_files = sorted(results_dir.glob("run_*.jsonl"))
    if not jsonl_files:
        warnings.warn(f"No JSONL files found in {results_dir}")
        return records

    for path in jsonl_files:
        # Infer domain from filename:  run_YYYYMMDD_HHMMSS_{domain}.jsonl
        parts = path.stem.split("_")
        domain = parts[-1] if len(parts) >= 3 else "unknown"

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry.setdefault("domain", domain)
                    entry.setdefault("prompt_id", path.stem)
                    records.append(entry)
                except json.JSONDecodeError:
                    continue

    return records


def build_dataset(
    results_dir: Path,
    min_samples: int = 100,
) -> pd.DataFrame:
    """Build the ML dataset from per-token JSONL logs.

    Args:
        results_dir: Directory containing ``run_*.jsonl`` files.
        min_samples: Minimum records to proceed.

    Returns:
        Clean DataFrame with features + target column (``accepted``).
    """
    global DOMAIN_COLUMNS

    records = _load_jsonl_files(results_dir)
    if len(records) < min_samples:
        raise ValueError(
            f"Only {len(records)} records found; need at least {min_samples}. "
            f"Run Phase 2 first to generate data."
        )

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df)} token records from {results_dir}")

    # ── Ensure required columns exist ──
    # Some older runs may lack enriched features; fill with sensible defaults.
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            if col == "draft_log_prob" and "p_draft" in df.columns:
                df["draft_log_prob"] = np.log(df["p_draft"].clip(lower=1e-10))
            elif col == "rolling_acceptance":
                df["rolling_acceptance"] = 0.5
            else:
                df[col] = 0.0

    for col in BOOLEAN_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # ── Target ──
    df[TARGET_COL] = df["accepted"].astype(int)

    # ── One-hot encode domain ──
    if "domain" in df.columns:
        dummies = pd.get_dummies(df["domain"], prefix="domain", dtype=float)
        DOMAIN_COLUMNS = sorted(dummies.columns.tolist())
        df = pd.concat([df, dummies], axis=1)
    else:
        DOMAIN_COLUMNS = []

    # ── Select feature columns ──
    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + DOMAIN_COLUMNS
    available = [c for c in feature_cols if c in df.columns]

    # Drop rows with NaN in feature columns
    before = len(df)
    df = df.dropna(subset=available + [TARGET_COL])
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with NaN values")

    # Cast types
    for col in available:
        df[col] = df[col].astype(float)

    print(f"  Dataset ready: {len(df)} samples, {len(available)} features")
    print(f"  Class balance: {df[TARGET_COL].mean():.1%} accepted")
    print(f"  Features: {available}")

    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    leaky: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Train/test split preventing data leakage.
    
    If leaky=False, uses GroupShuffleSplit to ensure tokens from the
    same generated trajectory (prompt_id) are strictly separated.
    If leaky=True, uses standard random split (demonstrates leakage).

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + DOMAIN_COLUMNS
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)

    if leaky:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y,
        )
    else:
        from sklearn.model_selection import GroupShuffleSplit
        groups = df["prompt_id"].values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    method_str = "Random (Leaky)" if leaky else "Grouped (Rigorous)"
    print(
        f"  Split ({method_str}): {len(X_train)} train / {len(X_test)} test  "
        f"(train accept rate: {y_train.mean():.1%})"
    )
    return X_train, X_test, y_train, y_test


def get_feature_names(df: pd.DataFrame) -> List[str]:
    """Return the ordered list of feature column names used by models."""
    feature_cols = NUMERIC_FEATURES + BOOLEAN_FEATURES + DOMAIN_COLUMNS
    return [c for c in feature_cols if c in df.columns]
