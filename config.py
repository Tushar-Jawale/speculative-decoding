"""Central configuration for the speculative decoding research project.

All paths, model names, hyperparameters, and seeds are defined here.
No hardcoded paths elsewhere in the codebase.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTOR_DIR = RESULTS_DIR / "predictor"

# Auto-create output directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────
# Device
# ────────────────────────────────────────────────────────────
def get_device(requested: str = "auto") -> str:
    """Determine compute device.

    Args:
        requested: One of 'cuda', 'cpu', or 'auto'.

    Returns:
        Device string suitable for ``torch.device``.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


DEVICE = get_device()

# ────────────────────────────────────────────────────────────
# Model pairs
# ────────────────────────────────────────────────────────────
# Primary pair (requires A100-class GPU)
PRIMARY_DRAFT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PRIMARY_TARGET_MODEL = "meta-llama/Llama-3-8B-Instruct"

# Fallback pair (always publicly available, CPU-friendly)
FALLBACK_DRAFT_MODEL = "distilgpt2"
FALLBACK_TARGET_MODEL = "gpt2-xl"

# Select based on available hardware
if DEVICE == "cuda":
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    except Exception:
        vram_gb = 0.0
else:
    vram_gb = 0.0

if vram_gb >= 24:
    DRAFT_MODEL = PRIMARY_DRAFT_MODEL
    TARGET_MODEL = PRIMARY_TARGET_MODEL
    USING_FALLBACK = False
else:
    DRAFT_MODEL = FALLBACK_DRAFT_MODEL
    TARGET_MODEL = FALLBACK_TARGET_MODEL
    USING_FALLBACK = True

# ────────────────────────────────────────────────────────────
# Hyperparameters
# ────────────────────────────────────────────────────────────
K = 4                       # Draft tokens per speculation step
MAX_NEW_TOKENS = 128        # Max tokens to generate per prompt
PROMPTS_PER_DOMAIN = 50     # Prompts per domain in Phase 2

# ────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────
SEED = 42

# ────────────────────────────────────────────────────────────
# Predictor (Phase 3) hyperparameters
# ────────────────────────────────────────────────────────────
MLP_HIDDEN_SIZES = (128, 64)    # MLP hidden layer sizes
MLP_EPOCHS = 80                 # Training epochs for MLP
MLP_LR = 1e-3                   # MLP learning rate
MLP_DROPOUT = 0.3               # Dropout probability
CV_FOLDS = 5                    # Cross-validation folds
ROLLING_WINDOW = 8              # Window for rolling acceptance rate


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ────────────────────────────────────────────────────────────
# W&B
# ────────────────────────────────────────────────────────────
WANDB_ENABLED = bool(os.environ.get("WANDB_API_KEY"))
RUN_LOG_PATH = RESULTS_DIR / "run_log.jsonl"

# ────────────────────────────────────────────────────────────
# Cleanup helper
# ────────────────────────────────────────────────────────────
def cleanup_gpu() -> None:
    """Clear GPU cache to free memory between phases."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
