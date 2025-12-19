"""CLI entrypoint for the speculative decoding research project.

Usage:
    python main.py --phase 1          # Run Phase 1 (engine verification)
    python main.py --phase 2          # Run Phase 2 (cross-domain benchmark)
    python main.py --phase 3          # Run Phase 3 (acceptance predictor)
    python main.py --phase all        # Run all phases sequentially
    python main.py --phase 1 --k 6    # Use K=6 draft tokens
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Set output encoding to UTF-8 to handle Unicode symbols on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import set_seed, cleanup_gpu


def run_phase1(k: int, device: str) -> None:
    """Phase 1 — Instrumented Inference Engine verification.

    Loads the draft/target models, runs 10 prompts through speculative
    decoding, and prints acceptance statistics.
    """
    from engine.draft_model import DraftModel
    from engine.target_model import TargetModel
    from engine.spec_engine import SpeculativeEngine
    from engine.logger import MetricsLogger

    set_seed()

    print("=" * 60)
    print("PHASE 1 — Instrumented Inference Engine")
    print("=" * 60)

    if config.USING_FALLBACK:
        print(f"⚠  Using fallback models: {config.DRAFT_MODEL} / {config.TARGET_MODEL}")

    print(f"Device: {device} | K={k}")
    print()

    # Load models
    print("Loading draft model...", flush=True)
    draft = DraftModel(config.DRAFT_MODEL, device)
    print("Loading target model...", flush=True)
    target = TargetModel(config.TARGET_MODEL, device)

    tokenizer = draft.tokenizer
    engine = SpeculativeEngine(
        draft_model=draft,
        target_model=target,
        tokenizer=tokenizer,
        k=k,
        device=device,
    )

    # Run 10 prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time in a land far away,",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n",
        "The capital of France is",
        "In machine learning, gradient descent is",
        "To be or not to be, that is the",
        "The weather today is expected to be",
        "import numpy as np\nimport pandas as pd\n",
        "The history of computing began with",
        "Water is composed of two hydrogen atoms and",
    ]

    print(f"\nRunning {len(prompts)} prompts with max_new_tokens=32...\n")

    acceptance_rates = []
    for i, prompt in enumerate(prompts):
        set_seed(42 + i)
        logger = MetricsLogger(domain="phase1_test")
        engine.logger = logger

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        result = engine.generate(input_ids, max_new_tokens=32)

        alpha = logger.acceptance_rate
        acceptance_rates.append(alpha)

        output_text = tokenizer.decode(
            result["output_ids"][0, input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        print(f"  [{i+1}] α={alpha:.3f} | {result['n_generated']} tokens | "
              f"{result['wall_time']:.1f}s")
        print(f"      → {output_text[:80]}...")

        logger.save()

    import numpy as np
    mean_alpha = np.mean(acceptance_rates)
    print(f"\n{'─'*40}")
    print(f"Mean acceptance rate: {mean_alpha:.4f}")
    print(f"Expected range: [0.5, 0.85] (may vary with fallback models)")
    print(f"{'─'*40}")

    cleanup_gpu()
    print("\n✓ Phase 1 complete.\n")


def run_phase2(k: int, device: str, prompts_per_domain: int = None) -> None:
    """Phase 2 — Empirical Study Across Domains."""
    from experiments.benchmark import run_benchmark

    set_seed()

    print("=" * 60)
    print("PHASE 2 — Empirical Study Across Domains")
    print("=" * 60)

    run_benchmark(k=k, device=device, prompts_per_domain=prompts_per_domain)

    cleanup_gpu()
    print("\n✓ Phase 2 complete.\n")


def run_phase3() -> None:
    """Phase 3 — Acceptance Rate Prediction."""
    from predictor.run_predictor import run_phase_3

    set_seed()
    run_phase_3()
    cleanup_gpu()



def run_k_sweep(device: str, prompts_per_domain: int = 3) -> None:
    """K-sensitivity analysis: sweep K ∈ {2, 4, 6, 8}.

    A core theoretical prediction of speculative decoding is that
    efficiency is non-monotone in K — larger K amortises more target
    forward passes when α is high, but wastes computation when α is low
    (more rejected draft tokens per cycle).  This sweep provides
    empirical evidence for that tradeoff.
    """
    import numpy as np
    import pandas as pd
    from experiments.benchmark import run_benchmark

    set_seed()

    print("=" * 60)
    print("K-SENSITIVITY SWEEP")
    print("=" * 60)
    print(f"K values: [2, 4, 6, 8]")
    print(f"Prompts per domain: {prompts_per_domain}")
    print()

    k_values = [2, 4, 6, 8]
    sweep_rows = []

    for k_val in k_values:
        print(f"\n{'─'*40}")
        print(f"Running K = {k_val}")
        print(f"{'─'*40}")
        set_seed()  # reset for fair comparison
        df = run_benchmark(
            k=k_val,
            device=device,
            prompts_per_domain=prompts_per_domain,
        )
        for domain in sorted(df["domain"].unique()):
            sub = df[df["domain"] == domain]
            sweep_rows.append({
                "K": k_val,
                "domain": domain,
                "mean_alpha": round(sub["mean_alpha"].mean(), 4),
                "mean_tps": round(sub["tokens_per_sec"].mean(), 2),
                "mean_kl": round(sub["mean_kl"].mean(), 4),
            })
        cleanup_gpu()

    # ── Summary table ──
    sweep_df = pd.DataFrame(sweep_rows)

    print("\n" + "=" * 70)
    print("K-SENSITIVITY SUMMARY")
    print("=" * 70)
    print(f"\n{'K':>4} {'Domain':<12} {'Mean α':>10} {'Tok/s':>8} {'Mean KL':>10}")
    print("-" * 50)
    for _, row in sweep_df.iterrows():
        print(
            f"{row['K']:>4} {row['domain']:<12} {row['mean_alpha']:>10.4f} "
            f"{row['mean_tps']:>8.2f} {row['mean_kl']:>10.4f}"
        )

    # Per-K aggregate
    print(f"\n{'K':>4} {'Avg α':>10} {'Avg Tok/s':>10}")
    print("-" * 28)
    for k_val in k_values:
        sub = sweep_df[sweep_df["K"] == k_val]
        print(
            f"{k_val:>4} {sub['mean_alpha'].mean():>10.4f} "
            f"{sub['mean_tps'].mean():>10.2f}"
        )

    # Save
    from config import RESULTS_DIR
    csv_path = RESULTS_DIR / "k_sweep_results.csv"
    sweep_df.to_csv(csv_path, index=False)
    print(f"\n✓ K-sweep results saved to {csv_path}")
    print("✓ K-sweep complete.\n")


def main() -> None:
    """Parse CLI arguments and dispatch to the requested phase."""
    parser = argparse.ArgumentParser(
        description="Speculative Decoding Research — CLI Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --phase 1
  python main.py --phase 2 --k 6
  python main.py --phase 3
  python main.py --phase all --device cpu
  python main.py --sweep-k               # K-sensitivity analysis
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=config.K,
        help=f"Speculation window size (default: {config.K})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=config.DEVICE,
        help=f"Compute device (default: {config.DEVICE})",
    )
    parser.add_argument(
        "--prompts-per-domain",
        type=int,
        default=None,
        help="Number of prompts per domain in Phase 2 (default: from config)",
    )
    parser.add_argument(
        "--sweep-k",
        action="store_true",
        default=False,
        help="Run K-sensitivity sweep (K ∈ {2, 4, 6, 8}, 3 prompts/domain)",
    )

    args = parser.parse_args()

    print(f"\n{'━'*60}")
    print("  Speculative Decoding Research")
    print(f"  Draft Model Alignment & Acceptance Rate Analysis")
    print(f"{'━'*60}\n")

    start = time.time()

    if args.sweep_k:
        run_k_sweep(args.device)
    else:
        if args.phase in ("1", "all"):
            run_phase1(args.k, args.device)

        if args.phase in ("2", "all"):
            run_phase2(args.k, args.device, args.prompts_per_domain)

        if args.phase in ("3", "all"):
            run_phase3()

    elapsed = time.time() - start
    print(f"\nTotal elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
