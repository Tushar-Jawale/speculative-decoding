"""Cross-domain benchmark for speculative decoding.

Runs all prompts from each domain through the speculative engine,
collects per-token metrics, computes aggregate statistics, and
produces the four required visualisations:

1. Box plot of acceptance rate by domain
2. Scatter plot: KL divergence vs acceptance rate
3. Line plot: per-position acceptance rate
4. Heatmap: domain × acceptance rate bins
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEVICE,
    DRAFT_MODEL,
    K,
    MAX_NEW_TOKENS,
    PLOTS_DIR,
    RESULTS_DIR,
    SEED,
    TARGET_MODEL,
    USING_FALLBACK,
    set_seed,
)
from engine.draft_model import DraftModel
from engine.kv_cache import KVCache
from engine.logger import MetricsLogger
from engine.spec_engine import SpeculativeEngine
from engine.target_model import TargetModel
from experiments.datasets_loader import load_datasets


# ────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────

# Use a modern aesthetic
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})

DOMAIN_PALETTE = {
    "code": "#4361ee",
    "math": "#f72585",
    "instruct": "#4cc9f0",
    "creative": "#7209b7",
    "long": "#f77f00",
}


def plot_acceptance_by_domain(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot 1 — Box plot of acceptance rate by domain."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = df.groupby("domain")["mean_alpha"].median().sort_values().index.tolist()
    sns.boxplot(
        data=df,
        x="domain",
        y="mean_alpha",
        order=order,
        palette=DOMAIN_PALETTE,
        ax=ax,
        width=0.5,
        linewidth=1.5,
    )
    ax.set_xlabel("Domain", fontsize=13)
    ax.set_ylabel("Mean Acceptance Rate (α)", fontsize=13)
    ax.set_title("Token Acceptance Rate by Prompt Domain", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(save_dir / "01_acceptance_by_domain.png", dpi=150)
    plt.close(fig)
    print("  ✓ Plot 1: acceptance_by_domain.png")


def plot_kl_vs_acceptance(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot 2 — Scatter plot: KL divergence vs acceptance rate."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for domain, colour in DOMAIN_PALETTE.items():
        subset = df[df["domain"] == domain]
        ax.scatter(
            subset["mean_kl"],
            subset["mean_alpha"],
            c=colour,
            label=domain,
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )

    # Trend line
    valid = df.dropna(subset=["mean_kl", "mean_alpha"])
    if len(valid) > 2:
        z = np.polyfit(valid["mean_kl"], valid["mean_alpha"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(valid["mean_kl"].min(), valid["mean_kl"].max(), 100)
        ax.plot(x_range, p(x_range), "--", color="gray", alpha=0.6, linewidth=2)

    ax.set_xlabel("Mean KL Divergence (nats)", fontsize=13)
    ax.set_ylabel("Mean Acceptance Rate (α)", fontsize=13)
    ax.set_title("KL Divergence vs Token Acceptance Rate", fontsize=15, fontweight="bold")
    ax.legend(title="Domain", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_dir / "02_kl_vs_acceptance.png", dpi=150)
    plt.close(fig)
    print("  ✓ Plot 2: kl_vs_acceptance.png")


def plot_acceptance_by_position(all_records: List[Dict], save_dir: Path) -> None:
    """Plot 3 — Per-position acceptance rate across all prompts."""
    from collections import defaultdict

    pos_data: Dict[int, List[bool]] = defaultdict(list)
    for rec in all_records:
        # Use position relative to speculation window, not absolute
        pos_data[rec["position"]].append(rec["accepted"])

    # Convert absolute positions to relative (0-based from start of generation)
    min_pos = min(pos_data.keys()) if pos_data else 0
    relative: Dict[int, float] = {}
    for pos, vals in sorted(pos_data.items()):
        rel = pos - min_pos
        if rel <= 200:  # cap for readability
            relative[rel] = sum(vals) / len(vals)

    if not relative:
        print("  ⚠ Plot 3: no position data available")
        return

    positions = sorted(relative.keys())
    rates = [relative[p] for p in positions]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(positions, rates, color="#4361ee", alpha=0.8, linewidth=1.5)
    # Smoothed trend
    if len(rates) > 10:
        window = min(10, len(rates) // 3)
        smoothed = pd.Series(rates).rolling(window=window, center=True).mean()
        ax.plot(positions, smoothed, color="#f72585", linewidth=2.5, label="Smoothed")
        ax.legend()

    ax.set_xlabel("Position in Generated Sequence", fontsize=13)
    ax.set_ylabel("Acceptance Rate", fontsize=13)
    ax.set_title("Acceptance Rate vs Sequence Position", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(save_dir / "03_acceptance_by_position.png", dpi=150)
    plt.close(fig)
    print("  ✓ Plot 3: acceptance_by_position.png")


def plot_domain_heatmap(df: pd.DataFrame, save_dir: Path) -> None:
    """Plot 4 — Heatmap: domain × acceptance rate bin."""
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    df = df.copy()
    df["alpha_bin"] = pd.cut(df["mean_alpha"], bins=bins, labels=bin_labels, include_lowest=True)

    pivot = df.groupby(["domain", "alpha_bin"], observed=False).size().unstack(fill_value=0)
    # Normalise per domain (row)
    row_sums = pivot.sum(axis=1)
    pivot_norm = pivot.div(row_sums.replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot_norm,
        annot=pivot.values,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Fraction of prompts"},
    )
    ax.set_xlabel("Acceptance Rate Bin", fontsize=13)
    ax.set_ylabel("Domain", fontsize=13)
    ax.set_title("Distribution of Acceptance Rates by Domain", fontsize=15, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "04_domain_heatmap.png", dpi=150)
    plt.close(fig)
    print("  ✓ Plot 4: domain_heatmap.png")


# ────────────────────────────────────────────────────────────
# Main benchmark runner
# ────────────────────────────────────────────────────────────


def run_benchmark(
    k: int = K,
    max_new_tokens: int = MAX_NEW_TOKENS,
    prompts_per_domain: int = None,
    device: str = DEVICE,
) -> pd.DataFrame:
    """Run the full cross-domain benchmark.

    Args:
        k: Draft tokens per speculation cycle.
        max_new_tokens: Max tokens to generate per prompt.
        prompts_per_domain: Override for number of prompts per domain.
        device: Compute device.

    Returns:
        DataFrame of benchmark results (one row per prompt).
    """
    from config import PROMPTS_PER_DOMAIN
    if prompts_per_domain is None:
        prompts_per_domain = PROMPTS_PER_DOMAIN

    set_seed()

    if USING_FALLBACK:
        print(f"⚠  Using fallback models: {DRAFT_MODEL} / {TARGET_MODEL}")
    print(f"Device: {device} | K={k} | max_new_tokens={max_new_tokens}")
    print(f"Prompts per domain: {prompts_per_domain}\n")

    # ── Load models ──
    print("Loading draft model...", flush=True)
    draft = DraftModel(DRAFT_MODEL, device)
    print("Loading target model...", flush=True)
    target = TargetModel(TARGET_MODEL, device)
    tokenizer = draft.tokenizer

    engine = SpeculativeEngine(
        draft_model=draft,
        target_model=target,
        tokenizer=tokenizer,
        k=k,
        device=device,
    )

    # ── Load datasets ──
    datasets = load_datasets(prompts_per_domain)

    # ── Run benchmark ──
    results: List[Dict[str, Any]] = []
    all_records: List[Dict] = []

    for domain, prompts in datasets.items():
        print(f"\n{'='*60}")
        print(f"Domain: {domain} ({len(prompts)} prompts)")
        print(f"{'='*60}")

        for idx, prompt_text in enumerate(prompts):
            set_seed(SEED + idx + hash(domain) % 1000)  # per-prompt variation

            # Prepare logger for this prompt
            logger = MetricsLogger(domain=domain)
            engine.logger = logger

            # Tokenize
            input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
            # Truncate very long prompts
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, :512]

            print(f"  [{idx+1}/{len(prompts)}] {prompt_text[:60]}...", end=" ", flush=True)

            try:
                result = engine.generate(input_ids, max_new_tokens=max_new_tokens)

                # Compute metrics from logger
                records = logger.records
                all_records.extend(records)

                n_total_tokens = len(records)
                n_accepted = sum(1 for r in records if r["accepted"])
                mean_alpha = n_accepted / max(n_total_tokens, 1)
                mean_kl = (
                    sum(r["kl_divergence"] for r in records) / max(n_total_tokens, 1)
                )
                tps = result["n_generated"] / max(result["wall_time"], 1e-6)

                results.append({
                    "domain": domain,
                    "prompt_id": idx,
                    "mean_alpha": round(mean_alpha, 4),
                    "mean_kl": round(mean_kl, 4),
                    "tokens_per_sec": round(tps, 2),
                    "wall_time_s": round(result["wall_time"], 2),
                    "n_generated": result["n_generated"],
                    "n_steps": result["n_steps"],
                    "prompt_len": input_ids.shape[1],
                })

                print(f"α={mean_alpha:.3f} KL={mean_kl:.3f} tps={tps:.1f}")

            except Exception as exc:
                print(f"ERROR: {exc}")
                results.append({
                    "domain": domain,
                    "prompt_id": idx,
                    "mean_alpha": float("nan"),
                    "mean_kl": float("nan"),
                    "tokens_per_sec": 0.0,
                    "wall_time_s": 0.0,
                    "n_generated": 0,
                    "n_steps": 0,
                    "prompt_len": input_ids.shape[1],
                })

            # Save log
            logger.save()

    # ── Save results CSV ──
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    # ── Generate plots ──
    print("\nGenerating plots...")
    plot_acceptance_by_domain(df, PLOTS_DIR)
    plot_kl_vs_acceptance(df, PLOTS_DIR)
    plot_acceptance_by_position(all_records, PLOTS_DIR)
    plot_domain_heatmap(df, PLOTS_DIR)

    # ── Compute & print correlations ──
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    valid = df.dropna(subset=["mean_kl", "mean_alpha"])
    if len(valid) > 2:
        pearson_r, pearson_p = stats.pearsonr(valid["mean_kl"], valid["mean_alpha"])
        spearman_r, spearman_p = stats.spearmanr(valid["mean_kl"], valid["mean_alpha"])
        print(f"Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.4e})")
        print(f"Spearman ρ = {spearman_r:.4f}  (p = {spearman_p:.4e})")
    else:
        print("Not enough data for correlation analysis.")

    print(f"\n{'Domain':<12} {'Mean α':>10} {'Std α':>10} {'Mean KL':>10}")
    print("-" * 48)
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain]
        print(
            f"{domain:<12} {sub['mean_alpha'].mean():>10.4f} "
            f"{sub['mean_alpha'].std():>10.4f} {sub['mean_kl'].mean():>10.4f}"
        )

    # Best/worst domain
    domain_means = df.groupby("domain")["mean_alpha"].mean()
    best = domain_means.idxmax()
    worst = domain_means.idxmin()
    print(f"\nHighest acceptance: {best} ({domain_means[best]:.4f})")
    print(f"Lowest acceptance:  {worst} ({domain_means[worst]:.4f})")

    return df


if __name__ == "__main__":
    run_benchmark()
