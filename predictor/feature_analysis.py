"""Feature analysis tools for the acceptance-rate predictor.

Calculates and visualizes Native Feature Importance,
Permutation Importance, and SHAP values (if installed).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

from config import PLOTS_DIR


def analyze_features(
    model, X_test: np.ndarray, y_test: np.ndarray, feature_names: list[str]
) -> None:
    """Run full feature analysis on a trained model."""
    print("\n--- Feature Analysis ---")
    
    # 1. Native Feature Importance (for tree-based models like XGBoost)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        _plot_importances(importances, feature_names, "Native Feature Importance")
    
    # 2. Permutation Importance
    if hasattr(model, "predict"):
        print("Calculating Permutation Importance...")
        result = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        _plot_importances(result.importances_mean, feature_names, "Permutation Importance")
        
    # 3. SHAP Values
    try:
        import shap
        print("Calculating SHAP values...")
        if type(model).__name__ == "XGBClassifier":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            
            out_path = PLOTS_DIR / "shap_summary.png"
            plt.savefig(out_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Saved SHAP plot to {out_path}")
    except ImportError:
        print("SHAP not installed. Skipping SHAP analysis.")


def _plot_importances(importances: np.ndarray, feature_names: list[str], title: str) -> None:
    """Helper to plot and save feature importances."""
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    
    filename = title.lower().replace(" ", "_") + ".png"
    out_path = PLOTS_DIR / filename
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {title} plot to {out_path}")
