"""Rigorous error analysis for the Acceptance Rate Predictor.

Stratifies prediction errors across multiple dimensions to identify
failure modes, such as domain-specific biases, high-entropy uncertainty,
and sequence length degradation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import PLOTS_DIR


def analyze_errors(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> None:
    """Perform multi-dimensional error analysis."""
    print("\n" + "=" * 50)
    print(" ERROR ANALYSIS & STRATIFICATION")
    print("=" * 50)

    # Reconstruct DataFrame for easy slicing
    df = pd.DataFrame(X_test, columns=feature_names)
    df["y_true"] = y_test
    df["y_pred"] = y_pred
    df["correct"] = df["y_true"] == df["y_pred"]
    df["error_type"] = "TN"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 1), "error_type"] = "TP"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "error_type"] = "FP"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "error_type"] = "FN"

    _analyze_by_domain(df)
    _analyze_by_entropy(df)
    _analyze_by_position(df)


def _analyze_by_domain(df: pd.DataFrame) -> None:
    """Analyze errors stratified by prompt domain."""
    print("\n--- Errors by Domain ---")
    domain_cols = [c for c in df.columns if c.startswith("domain_")]
    if not domain_cols:
        print("No domain columns found.")
        return

    print(f"{'Domain':<15} | {'Accuracy':<10} | {'FN Rate':<10} | {'FP Rate':<10}")
    print("-" * 55)

    for col in domain_cols:
        domain = col.replace("domain_", "")
        sub = df[df[col] == 1.0]
        if len(sub) == 0:
            continue
            
        acc = sub["correct"].mean()
        # False Negative Rate = FN / Actual Positives
        positives = len(sub[sub["y_true"] == 1])
        fn_rate = len(sub[sub["error_type"] == "FN"]) / max(positives, 1)
        
        # False Positive Rate = FP / Actual Negatives
        negatives = len(sub[sub["y_true"] == 0])
        fp_rate = len(sub[sub["error_type"] == "FP"]) / max(negatives, 1)
        
        print(f"{domain:<15} | {acc:<10.1%} | {fn_rate:<10.1%} | {fp_rate:<10.1%}")


def _analyze_by_entropy(df: pd.DataFrame) -> None:
    """Analyze accuracy relative to the draft model's uncertainty."""
    if "draft_entropy" not in df.columns:
        return
        
    print("\n--- Errors by Draft Entropy (Uncertainty) ---")
    bins = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
    labels = ["0.0-0.5", "0.5-1.0", "1.0-2.0", "2.0-5.0", "5.0+"]
    df["entropy_bin"] = pd.cut(df["draft_entropy"], bins=bins, labels=labels)
    
    grouped = df.groupby("entropy_bin", observed=False)
    acc = grouped["correct"].mean()
    counts = grouped.size()
    
    print(f"{'Entropy Range':<15} | {'Accuracy':<10} | {'Support':<10}")
    print("-" * 45)
    for label in labels:
        if counts[label] > 0:
            print(f"{label:<15} | {acc[label]:<10.1%} | {counts[label]:<10}")


def _analyze_by_position(df: pd.DataFrame) -> None:
    """Analyze accuracy as sequence generation gets longer."""
    if "position" not in df.columns:
        return
        
    print("\n--- Errors by Sequence Position ---")
    bins = [0, 32, 64, 128, 256, 1024]
    labels = ["0-32", "33-64", "65-128", "129-256", "257+"]
    df["pos_bin"] = pd.cut(df["position"], bins=bins, labels=labels)
    
    grouped = df.groupby("pos_bin", observed=False)
    acc = grouped["correct"].mean()
    counts = grouped.size()
    
    print(f"{'Position':<15} | {'Accuracy':<10} | {'Support':<10}")
    print("-" * 45)
    for label in labels:
        if counts[label] > 0:
            print(f"{label:<15} | {acc[label]:<10.1%} | {counts[label]:<10}")
