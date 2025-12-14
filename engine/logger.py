"""Per-token metrics logger for speculative decoding runs.

Records acceptance decisions, draft/target probabilities, KL divergence,
and other per-token diagnostics.  Results are saved as JSONL files under
``results/`` and optionally streamed to Weights & Biases.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import RESULTS_DIR, WANDB_ENABLED


class MetricsLogger:
    """Collects and persists per-token speculative decoding metrics.

    Each logged entry contains:
    - ``token_id``, ``token_text``
    - ``accepted`` (bool)
    - ``p_draft``, ``p_target`` (float)
    - ``kl_divergence`` (float)
    - ``position`` (int) — absolute position in the generated sequence
    - ``step`` (int) — speculation cycle index

    Attributes:
        records: In-memory buffer of all logged entries.
    """

    def __init__(self, domain: str = "unknown", run_id: Optional[str] = None) -> None:
        self.domain = domain
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records: List[Dict[str, Any]] = []

        # W&B integration (lazy init)
        self._wandb_run = None
        if WANDB_ENABLED:
            try:
                import wandb  # type: ignore

                self._wandb_run = wandb.init(
                    project="spec-decoding-research",
                    name=f"{self.domain}_{self.run_id}",
                    reinit=True,
                )
            except Exception:
                self._wandb_run = None

    # ── logging ───────────────────────────────────────────────

    def log_token(
        self,
        token_id: int,
        token_text: str,
        accepted: bool,
        p_draft: float,
        p_target: float,
        kl_divergence: float,
        position: int,
        step: int,
        **extra_features: float,
    ) -> None:
        """Record metrics for a single token decision.

        Args:
            token_id: Vocabulary index of the token.
            token_text: Decoded string representation.
            accepted: Whether the target model accepted the draft token.
            p_draft: Draft model probability for this token.
            p_target: Target model probability for this token.
            kl_divergence: KL(target‖draft) over the full vocabulary at this position.
            position: Absolute position in the generated sequence.
            step: Speculation cycle index (0-based).
            **extra_features: Additional per-token features for predictor
                training (e.g. ``draft_entropy``, ``draft_top5_mass``,
                ``confidence_margin``, ``cross_entropy``, ``draft_log_prob``,
                ``token_length``, ``is_punctuation``, ``is_whitespace``,
                ``rolling_acceptance``, ``domain``).
        """
        entry: Dict[str, Any] = {
            "token_id": token_id,
            "token_text": token_text,
            "accepted": accepted,
            "p_draft": round(p_draft, 6),
            "p_target": round(p_target, 6),
            "kl_divergence": round(kl_divergence, 6),
            "position": position,
            "step": step,
        }
        # Merge any additional features (Phase 3 predictor data)
        for key, val in extra_features.items():
            entry[key] = round(val, 6) if isinstance(val, float) else val
        self.records.append(entry)

        if self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.log(entry)
            except Exception:
                pass

    # ── persistence ───────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Flush all buffered records to a JSONL file.

        Args:
            path: Explicit output path.  If *None*, uses the default
                  ``results/run_{run_id}_{domain}.jsonl``.

        Returns:
            The path the file was written to.
        """
        if path is None:
            path = RESULTS_DIR / f"run_{self.run_id}_{self.domain}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for record in self.records:
                fh.write(json.dumps(record) + "\n")
        return path

    # ── summary helpers ───────────────────────────────────────

    @property
    def acceptance_rate(self) -> float:
        """Fraction of tokens that were accepted by the target model."""
        if not self.records:
            return 0.0
        accepted = sum(1 for r in self.records if r["accepted"])
        return accepted / len(self.records)

    @property
    def mean_kl(self) -> float:
        """Mean KL divergence across all logged positions."""
        if not self.records:
            return 0.0
        return sum(r["kl_divergence"] for r in self.records) / len(self.records)

    def get_per_position_acceptance(self) -> Dict[int, float]:
        """Return acceptance rate at each relative position (within speculation window)."""
        from collections import defaultdict

        counts: Dict[int, List[bool]] = defaultdict(list)
        for r in self.records:
            counts[r["position"]].append(r["accepted"])
        return {pos: sum(v) / len(v) for pos, v in sorted(counts.items())}

    def reset(self) -> None:
        """Clear in-memory records (e.g. between prompts)."""
        self.records.clear()

    def close(self) -> None:
        """Finalise the logger (close W&B run if active)."""
        if self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                wandb.finish()
            except Exception:
                pass
