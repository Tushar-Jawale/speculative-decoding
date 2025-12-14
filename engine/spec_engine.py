"""Main speculative decoding inference engine.

Implements the full speculative decoding loop from Leviathan et al. (2023):

1. Draft model proposes *K* candidate tokens autoregressively.
2. Target model evaluates all *K* candidates in one forward pass.
3. Acceptance/rejection sampling preserves the exact target distribution.
4. On rejection, a corrected token is sampled from the *adjusted*
   distribution: ``normalize(max(0, p_target − p_draft))``.
5. KV caches are rolled back to the accepted prefix.

Also provides ``standard_generate()`` for autoregressive baseline comparison.
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from engine.draft_model import DraftModel
from engine.kv_cache import KVCache
from engine.logger import MetricsLogger
from engine.target_model import TargetModel

try:
    from config import ROLLING_WINDOW
except ImportError:
    ROLLING_WINDOW = 8


def compute_kl_divergence(
    q_probs: torch.Tensor, p_probs: torch.Tensor, epsilon: float = 1e-10
) -> float:
    """Compute KL(q ‖ p) = Σ q(x) · log(q(x) / p(x)).

    Convention: call with ``(target, draft)`` to measure how much
    information the target distribution loses when approximated by the
    draft — i.e. KL(P_target ‖ P_draft).  This is the standard
    direction in acceptance-rate theory (Leviathan et al., 2023) since
    the target is the reference distribution.

    Args:
        q_probs: Reference (first) distribution, shape ``(V,)``.
        p_probs: Approximating (second) distribution, shape ``(V,)``.
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar KL divergence value (nats).
    """
    q = q_probs.clamp(min=epsilon)
    p = p_probs.clamp(min=epsilon)
    # PyTorch's kl_div expects the input (first arg) to be log-probabilities
    # and the target (second arg) to be probabilities.
    # So we compute KL(target || draft) by passing log(draft) and target.
    # Note: F.kl_div computes l_n = y_n \cdot (\log y_n - x_n)
    return F.kl_div(p.log(), q, reduction='sum').item()


class SpeculativeEngine:
    """Orchestrates the full speculative decoding loop.

    Args:
        draft_model: The small, fast draft model.
        target_model: The large, accurate target model.
        tokenizer: Tokenizer shared by both models (or at least compatible).
        k: Number of draft tokens per speculation cycle.
        device: Compute device.
        logger: Optional metrics logger.
    """

    def __init__(
        self,
        draft_model: DraftModel,
        target_model: TargetModel,
        tokenizer: object,
        k: int = 4,
        device: str = "cpu",
        logger: Optional[MetricsLogger] = None,
    ) -> None:
        self.draft = draft_model
        self.target = target_model
        self.tokenizer = tokenizer
        self.k = k
        self.device = device
        self.logger = logger

    # ────────────────────────────────────────────────────────────
    # Main speculative decoding generation
    # ────────────────────────────────────────────────────────────

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> Dict:
        """Generate tokens using speculative decoding.

        Args:
            input_ids: Prompt token IDs, shape ``(1, prompt_len)``.
            max_new_tokens: Maximum number of new tokens to generate.

        Returns:
            Dict with keys:
            - ``output_ids``: Full generated sequence (prompt + new tokens).
            - ``n_generated``: Number of new tokens actually generated.
            - ``n_accepted``: Total accepted draft tokens.
            - ``n_rejected``: Total rejected draft tokens.
            - ``n_steps``: Number of speculation cycles.
            - ``wall_time``: Wall-clock time in seconds.
        """
        start_time = time.time()

        generated = input_ids.to(self.device)
        prompt_len = generated.shape[1]

        # Initialize target model KV cache on the prompt
        target_cache = KVCache()
        next_target_logits, target_cache = self.target.init_cache(
            generated, target_cache
        )

        # Initialize draft model KV cache on the prompt
        draft_cache = KVCache()
        _, draft_past = self.draft.forward(generated)
        draft_cache.update(draft_past)

        n_generated = 0
        total_accepted = 0
        total_rejected = 0
        step = 0
        recent_decisions: List[float] = []  # for rolling acceptance rate

        while n_generated < max_new_tokens:
            remaining = max_new_tokens - n_generated
            k = min(self.k, remaining)

            # Save cache checkpoints for potential rollback
            target_cache.save_checkpoint()
            draft_cache.save_checkpoint()
            cache_len_before = len(target_cache)

            # ── DRAFT PHASE ──
            # Reuse persistent draft cache across cycles for efficiency
            candidates, draft_probs, draft_cache = self.draft.generate_candidates(
                generated, k, draft_cache
            )

            candidate_ids = torch.cat(candidates, dim=-1)  # (1, k)

            # ── VERIFICATION PHASE ──
            # Target evaluates all K candidates in a single forward pass
            verify_logits, target_cache = self.target.verify(
                candidate_ids, target_cache
            )
            # verify_logits shape: (1, k, vocab_size)

            # ── ACCEPTANCE PHASE ──
            n_accepted = 0
            adjusted_token = None

            for i in range(k):
                # Target logits for verifying candidate i:
                #   i == 0: use saved logits from previous cycle (predicts
                #           what comes right after the current prefix)
                #   i >= 1: use verify_logits[:, i-1, :] (predicts what comes
                #           after prefix + candidates[0..i-1])
                if i == 0:
                    t_logits = next_target_logits
                else:
                    t_logits = verify_logits[:, i - 1, :]

                t_probs = F.softmax(t_logits, dim=-1)  # (1, V)
                d_probs = draft_probs[i]  # (1, V)

                token_id = candidates[i].item()
                p_t = t_probs[0, token_id].item()
                p_d = d_probs[0, token_id].item()

                # KL(target ‖ draft): measures how well the draft
                # approximates the target — the reference distribution.
                kl = compute_kl_divergence(t_probs[0], d_probs[0])

                # Acceptance test: r < min(1, p_target / p_draft)
                r = random.random()
                accept_prob = min(1.0, p_t / max(p_d, 1e-10))
                accepted = r < accept_prob

                # Log metrics (with enriched features for predictor)
                if self.logger:
                    # ── Compute enriched features ──
                    d_clamped = d_probs[0].clamp(min=1e-10)
                    t_clamped = t_probs[0].clamp(min=1e-10)

                    # Draft entropy: H(draft) = -Σ p·log(p)
                    draft_entropy = -(d_clamped * d_clamped.log()).sum().item()

                    # Top-5 probability mass
                    draft_top5_mass = d_probs[0].topk(5).values.sum().item()

                    # Confidence margin: top-1 minus top-2 probability
                    top2_vals = d_probs[0].topk(2).values
                    confidence_margin = (top2_vals[0] - top2_vals[1]).item()

                    # Cross-entropy: H(target, draft) = -Σ t·log(d)
                    cross_entropy = -(t_clamped * d_clamped.log()).sum().item()

                    # Draft log probability
                    draft_log_prob = math.log(max(p_d, 1e-10))

                    # Token metadata
                    token_text = self.tokenizer.decode([token_id])
                    token_length = len(token_text)
                    is_punct = 1.0 if (
                        token_text.strip() != ""
                        and all(c in ".,;:!?()[]{}\"'-/\\@#$%^&*" for c in token_text.strip())
                    ) else 0.0
                    is_ws = 1.0 if token_text.strip() == "" else 0.0

                    # Rolling acceptance rate
                    rolling_alpha = (
                        sum(recent_decisions[-ROLLING_WINDOW:])
                        / max(len(recent_decisions[-ROLLING_WINDOW:]), 1)
                        if recent_decisions else 0.5  # prior
                    )

                    self.logger.log_token(
                        token_id=token_id,
                        token_text=token_text,
                        accepted=accepted,
                        p_draft=p_d,
                        p_target=p_t,
                        kl_divergence=kl,
                        position=prompt_len + n_generated + i,
                        step=step,
                        # ── Enriched features for predictor ──
                        draft_entropy=draft_entropy,
                        draft_log_prob=draft_log_prob,
                        draft_top5_mass=draft_top5_mass,
                        confidence_margin=confidence_margin,
                        cross_entropy=cross_entropy,
                        token_length=token_length,
                        is_punctuation=is_punct,
                        is_whitespace=is_ws,
                        rolling_acceptance=rolling_alpha,
                    )

                # Track for rolling window
                recent_decisions.append(1.0 if accepted else 0.0)

                if accepted:
                    n_accepted += 1
                else:
                    # Reject — sample from adjusted distribution:
                    # p_adjusted = normalize(max(0, p_target − p_draft))
                    adjusted = torch.clamp(t_probs - d_probs, min=0)
                    adj_sum = adjusted.sum()
                    if adj_sum > 1e-10:
                        adjusted = adjusted / adj_sum
                    else:
                        # Edge case: distributions match perfectly → use target
                        adjusted = t_probs
                    adjusted_token = torch.multinomial(adjusted, num_samples=1)  # (1, 1)
                    total_rejected += 1
                    break

            if n_accepted == k:
                # ── ALL ACCEPTED + BONUS TOKEN ──
                bonus_logits = verify_logits[:, -1, :]  # (1, V)
                bonus_probs = F.softmax(bonus_logits, dim=-1)
                bonus_token = torch.multinomial(bonus_probs, num_samples=1)  # (1, 1)

                generated = torch.cat(
                    [generated, candidate_ids, bonus_token], dim=-1
                )
                n_generated += k + 1
                total_accepted += k

                # Extend target cache with the bonus token
                bonus_logits_new, past = self.target.forward(
                    bonus_token, target_cache.get()
                )
                target_cache.update(past)
                next_target_logits = bonus_logits_new[:, -1, :]

                # Extend draft cache with the bonus token
                _, past_d = self.draft.forward(
                    bonus_token.to(self.device), draft_cache.get()
                )
                draft_cache.update(past_d)

                # NOTE: Bonus tokens are NOT logged.  They are sampled
                # purely from the target distribution (p_draft is undefined),
                # so they are always accepted by definition (α=1).  Logging
                # them with kl=0.0 / p_draft=0.0 would artificially inflate
                # acceptance rates and deflate mean KL.

            else:
                # ── PARTIAL ACCEPTANCE ──
                accepted_tokens = candidate_ids[:, :n_accepted]
                total_accepted += n_accepted

                generated = torch.cat(
                    [generated, accepted_tokens, adjusted_token], dim=-1
                )
                n_generated += n_accepted + 1

                # Rollback target cache: currently has cache_len_before + k,
                # need cache_len_before + n_accepted
                rollback_n = k - n_accepted
                target_cache.rollback(rollback_n)

                # Extend target cache with the adjusted token
                adj_logits, past = self.target.forward(
                    adjusted_token, target_cache.get()
                )
                target_cache.update(past)
                next_target_logits = adj_logits[:, -1, :]

                # Rollback draft cache
                draft_cache.rollback(rollback_n)

                # Extend draft cache with the adjusted token
                _, past_d = self.draft.forward(
                    adjusted_token.to(self.device), draft_cache.get()
                )
                draft_cache.update(past_d)

            step += 1

        wall_time = time.time() - start_time

        return {
            "output_ids": generated,
            "n_generated": n_generated,
            "n_accepted": total_accepted,
            "n_rejected": total_rejected,
            "n_steps": step,
            "wall_time": wall_time,
        }

    # ────────────────────────────────────────────────────────────
    # Standard autoregressive generation (baseline)
    # ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def standard_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
    ) -> Dict:
        """Standard autoregressive generation from the target model.

        Used as a correctness baseline: speculative decoding must produce
        the same token distribution as this method.

        Args:
            input_ids: Prompt token IDs, shape ``(1, prompt_len)``.
            max_new_tokens: Maximum new tokens to generate.

        Returns:
            Dict with ``output_ids``, ``n_generated``, ``wall_time``.
        """
        start_time = time.time()
        generated = input_ids.to(self.device)

        outputs = self.target.model(generated, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        for _ in range(max_new_tokens):
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated = torch.cat([generated, token], dim=-1)

            outputs = self.target.model(
                token, past_key_values=past, use_cache=True
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        wall_time = time.time() - start_time
        return {
            "output_ids": generated,
            "n_generated": max_new_tokens,
            "wall_time": wall_time,
        }
