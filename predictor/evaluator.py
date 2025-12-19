"""Evaluation utilities for predictor models."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

from config import PLOTS_DIR


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    """Evaluate a trained model and return metrics + curves."""
    # Handle PyTorch MLP vs sklearn-compatible models
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
    else:
        import torch
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            logits = model(X_t)
            y_prob = torch.sigmoid(logits).numpy()
            y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(classification_report(y_test, y_pred))

    return {
        "name": name,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
    }


def plot_all(results: list, title_prefix: str = "") -> None:
    """Plot ROC and PR curves for all evaluated models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for res in results:
        name = res["name"]
        
        # ROC
        ax1.plot(res["fpr"], res["tpr"], label=f'{name} (AUC={res["roc_auc"]:.3f})')
        
        # PR
        ax2.plot(res["recall"], res["precision"], label=f'{name} (AUC={res["pr_auc"]:.3f})')

    # ROC aesthetics
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"{title_prefix}ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # PR aesthetics
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"{title_prefix}Precision-Recall Curve")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOTS_DIR / "predictor_curves.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nSaved evaluation curves to {out_path}")
    plt.close()
