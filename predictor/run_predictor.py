"""Phase 3: Rigorous Acceptance Rate Prediction Pipeline.

This pipeline executes a defensible ML experimentation workflow:
1. Data Leakage Analysis (Leaky Random Split vs Rigorous Grouped Split)
2. Multi-seed Benchmarking (Statistical reliability)
3. Feature Ablation Studies (Hypothesis testing)
4. Interpretability & Error Analysis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import RESULTS_DIR, MLP_EPOCHS, MLP_LR, SEED
from predictor.dataset_builder import build_dataset, get_feature_names, split_dataset
from predictor.models import build_logistic_regression, build_xgboost, AcceptanceMLP
from predictor.evaluator import evaluate_model, plot_all
from predictor.feature_analysis import analyze_features
from predictor.adaptive_engine import simulate_adaptive_decoding
from predictor.error_analysis import analyze_errors


SEEDS = [42, 100, 777]


def train_mlp(X_train, y_train, X_test, y_test, seed=42, verbose=False):
    """Train the PyTorch MLP classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = AcceptanceMLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=MLP_LR)
    
    if verbose:
        print(f"    Training PyTorch MLP (Seed {seed})...")
        
    for epoch in range(MLP_EPOCHS):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    model.cpu()
    return model


def run_experiment(model_type, df, feature_names, seeds, leaky=False, subset_features=None, verbose=False):
    """Run an experiment across multiple seeds and return aggregate metrics."""
    accs, aucs = [], []
    best_model = None
    best_test_data = None
    
    for s in seeds:
        X_train, X_test, y_train, y_test = split_dataset(df, seed=s, leaky=leaky)
        
        # Apply ablation filter if requested
        if subset_features is not None:
            indices = [feature_names.index(f) for f in subset_features]
            X_train = X_train[:, indices]
            X_test = X_test[:, indices]

        if model_type == "xgb":
            model = build_xgboost(random_state=s)
            model.fit(X_train, y_train)
        elif model_type == "lr":
            model = build_logistic_regression(random_state=s)
            model.fit(X_train, y_train)
        elif model_type == "mlp":
            model = train_mlp(X_train, y_train, X_test, y_test, seed=s, verbose=verbose)
            
        res = evaluate_model(model, X_test, y_test, name=f"{model_type}_{s}")
        accs.append(res["accuracy"])
        aucs.append(res["roc_auc"])
        
        # Keep track of the first seed's model for plotting/analysis
        if best_model is None:
            best_model = model
            best_test_data = (X_test, y_test)
            
    if verbose:
        print(f"      Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"      ROC-AUC:  {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        
    return np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs), best_model, best_test_data


def run_phase_3():
    """Main pipeline for rigorous Phase 3 evaluation."""
    print("=" * 60)
    print(" PHASE 3: RIGOROUS ACCEPTANCE RATE PREDICTION")
    print("=" * 60)
    
    df = build_dataset(RESULTS_DIR)
    feature_names = get_feature_names(df)
    
    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Data Leakage Quantification
    # Hypothesis: Token-level random splitting artificially inflates performance.
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 1] Data Leakage vs Rigorous Grouping (XGBoost, 3 seeds)")
    print("  Running with Leaky Split (Random Tokens)...")
    _, _, leaky_auc, leaky_std, _, _ = run_experiment("xgb", df, feature_names, SEEDS, leaky=True)
    
    print("  Running with Rigorous Split (Grouped by Prompt)...")
    _, _, group_auc, group_std, xgb_model, test_data = run_experiment("xgb", df, feature_names, SEEDS, leaky=False)
    
    print(f"\n  Leakage Inflation: ROC-AUC dropped from {leaky_auc:.3f} to {group_auc:.3f} when properly grouped.")
    
    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Multi-Seed Model Benchmark
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 2] Multi-Seed Benchmark on Rigorous Split (3 seeds)")
    print("  Model: Logistic Regression")
    run_experiment("lr", df, feature_names, SEEDS, leaky=False, verbose=True)
    
    print("  Model: XGBoost")
    run_experiment("xgb", df, feature_names, SEEDS, leaky=False, verbose=True)
    
    print("  Model: PyTorch MLP")
    run_experiment("mlp", df, feature_names, SEEDS, leaky=False, verbose=True)

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Feature Ablation (Using XGBoost)
    # Hypothesis: Divergence metrics (KL) drive prediction more than Context.
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 3] Feature Ablation Studies (XGBoost)")
    
    # Define feature groups
    divergence_feats = ["kl_divergence", "cross_entropy"]
    confidence_feats = ["p_draft", "draft_log_prob", "draft_entropy", "draft_top5_mass", "confidence_margin"]
    context_feats = ["position", "rolling_acceptance", "token_length", "is_punctuation", "is_whitespace"] + [c for c in feature_names if c.startswith("domain_")]
    
    ablations = {
        "No Divergence": [f for f in feature_names if f not in divergence_feats],
        "No Draft Confidence": [f for f in feature_names if f not in confidence_feats],
        "No Context/Domain": [f for f in feature_names if f not in context_feats],
        "Only KL Divergence": ["kl_divergence"]
    }
    
    for name, feats in ablations.items():
        print(f"  Ablation: {name} ({len(feats)} features)")
        acc_m, acc_s, auc_m, auc_s, _, _ = run_experiment("xgb", df, feature_names, SEEDS, leaky=False, subset_features=feats)
        print(f"    -> ROC-AUC: {auc_m:.3f} ± {auc_s:.3f} (Delta: {auc_m - group_auc:+.3f})")

    # -------------------------------------------------------------------------
    # EXPERIMENT 4: Interpretability & Error Analysis
    # -------------------------------------------------------------------------
    X_test, y_test = test_data
    
    # Feature Analysis on Best XGBoost Model
    analyze_features(xgb_model, X_test, y_test, feature_names)
    
    # Error Stratification
    y_pred = xgb_model.predict(X_test)
    analyze_errors(X_test, y_test, y_pred, feature_names)
    
    # -------------------------------------------------------------------------
    # EXPERIMENT 5: Adaptive Decoding Simulation
    # -------------------------------------------------------------------------
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    simulate_adaptive_decoding(y_prob_xgb, y_test, threshold=0.5)
    simulate_adaptive_decoding(y_prob_xgb, y_test, threshold=0.7)

    print("\nRigorous Phase 3 Evaluation Completed.")


if __name__ == "__main__":
    run_phase_3()
