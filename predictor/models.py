"""Model definitions for the acceptance-rate predictor.

Implements Logistic Regression, XGBoost, and a small MLP
for predicting draft token acceptance.
"""

from typing import Any

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import SEED


def build_logistic_regression(**kwargs: Any) -> LogisticRegression:
    """Build a Logistic Regression classifier with default hyperparams."""
    params = {
        "max_iter": 1000,
        "random_state": SEED,
        "class_weight": "balanced",
    }
    params.update(kwargs)
    return LogisticRegression(**params)


def build_xgboost(**kwargs: Any) -> XGBClassifier:
    """Build an XGBoost classifier with default hyperparams."""
    params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "random_state": SEED,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }
    params.update(kwargs)
    return XGBClassifier(**params)


class AcceptanceMLP(nn.Module):
    """A small Multi-Layer Perceptron for acceptance prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits."""
        return self.net(x).squeeze(-1)
