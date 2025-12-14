"""Draft model wrapper for speculative decoding.

Wraps a small causal language model (e.g. ``distilgpt2``) and provides
an interface for generating *K* candidate tokens autoregressively
along with their draft probabilities.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_cache import KVCache


class DraftModel:
    """Lightweight draft model used for speculative token proposals.

    Args:
        model_name: HuggingFace model identifier.
        device: Compute device string (``'cpu'`` or ``'cuda'``).
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception:
            warnings.warn(
                f"Failed to load {model_name}. Falling back to distilgpt2.",
                RuntimeWarning,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

        self.model.to(self.device)
        self.model.eval()

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[object] = None,
    ) -> Tuple[torch.Tensor, object]:
        """Single forward pass returning logits and updated KV cache.

        Args:
            input_ids: Token IDs, shape ``(1, seq_len)``.
            past_key_values: HuggingFace-style past key values.

        Returns:
            Tuple of ``(logits, new_past_key_values)``.
        """
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def generate_candidates(
        self,
        input_ids: torch.Tensor,
        k: int,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], KVCache]:
        """Generate *k* candidate tokens autoregressively.

        Args:
            input_ids: Full sequence so far, shape ``(1, seq_len)``.
            k: Number of draft tokens to generate.
            kv_cache: Optional existing KV cache. If *None* or empty,
                      the full ``input_ids`` are processed from scratch.

        Returns:
            Tuple of:
            - ``candidates``: list of *k* token tensors, each ``(1, 1)``
            - ``draft_probs``: list of *k* probability tensors over vocab, each ``(1, V)``
            - ``updated_cache``: KV cache including the drafted tokens
        """
        cache = kv_cache or KVCache()

        if cache.seq_len == 0:
            # Process the full sequence from scratch
            logits, past = self.forward(input_ids)
            cache.update(past)
            current_logits = logits[:, -1, :]  # (1, V)
        else:
            # Only process tokens that are not yet cached
            new_tokens = input_ids[:, cache.seq_len :]
            if new_tokens.shape[1] > 0:
                logits, past = self.forward(new_tokens, cache.get())
                cache.update(past)
                current_logits = logits[:, -1, :]
            else:
                # Cache covers the input exactly; rollback 1 token and
                # re-run it to get fresh logits without duplication.
                assert cache.seq_len == input_ids.shape[1], (
                    f"Cache ({cache.seq_len}) longer than input "
                    f"({input_ids.shape[1]}); state corruption detected"
                )
                cache.rollback(1)
                logits, past = self.forward(input_ids[:, -1:], cache.get())
                cache.update(past)
                current_logits = logits[:, -1, :]

        candidates: List[torch.Tensor] = []
        draft_probs: List[torch.Tensor] = []

        for _ in range(k):
            probs = F.softmax(current_logits, dim=-1)  # (1, V)
            token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            candidates.append(token)
            draft_probs.append(probs)

            # Autoregressive next step
            logits, past = self.forward(token, cache.get())
            cache.update(past)
            current_logits = logits[:, -1, :]

        return candidates, draft_probs, cache
