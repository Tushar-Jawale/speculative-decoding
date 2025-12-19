"""Acceptance rate prediction for speculative decoding (Phase 3).

Trains supervised models to predict whether a draft token will be
accepted before target-model verification.
"""

from .dataset_builder import build_dataset, split_dataset
from .models import build_logistic_regression, build_xgboost, AcceptanceMLP
from .evaluator import evaluate_model, plot_all
from .feature_analysis import analyze_features

__all__ = [
    "build_dataset",
    "split_dataset",
    "build_logistic_regression",
    "build_xgboost",
    "AcceptanceMLP",
    "evaluate_model",
    "plot_all",
    "analyze_features",
]
