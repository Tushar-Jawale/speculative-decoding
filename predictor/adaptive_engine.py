"""Simulation for adaptive speculative decoding.

Demonstrates the potential compute savings if we use the predictor
to early-reject draft tokens before target-model verification.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def simulate_adaptive_decoding(
    y_prob: np.ndarray, y_true: np.ndarray, threshold: float = 0.5
) -> None:
    """Simulate adaptive decoding based on predictor probabilities.
    
    If y_prob < threshold, we reject early (save target model compute).
    If y_prob >= threshold, we verify with target model.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    total = len(y_true)
    verified = tp + fp
    skipped = tn + fn
    
    print(f"\n--- Adaptive Decoding Simulation (Threshold: {threshold}) ---")
    print(f"Total tokens evaluated: {total}")
    
    print(f"Target model verifications skipped: {skipped} ({skipped/total:.1%})")
    print(f"  - True negatives (Compute Saved!): {tn} ({tn/total:.1%})")
    print(f"  - False negatives (Missed Accepts): {fn} ({fn/total:.1%})")
    
    print(f"Target model verifications run: {verified} ({verified/total:.1%})")
    print(f"  - True positives (Accepted): {tp} ({tp/total:.1%})")
    print(f"  - False positives (Rejected): {fp} ({fp/total:.1%})")
