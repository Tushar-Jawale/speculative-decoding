"""Target model wrapper for speculative decoding.

Wraps the large causal language model (e.g. ``gpt2-xl``) and provides
a verification interface that evaluates all *K* draft candidates in a
single forward pass.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.kv_cache import KVCache


class TargetModel:
    """Large target model used for verifying draft proposals.

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
                f"Failed to load {model_name}. Falling back to gpt2-xl.",
                RuntimeWarning,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2-xl")

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[object] = None,
    ) -> Tuple[torch.Tensor, object]:
        """Single forward pass.

        Args:
            input_ids: Token IDs, shape ``(1, seq_len)``.
            past_key_values: Existing KV cache.

        Returns:
            ``(logits, new_past_key_values)``
        """
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values

    @torch.no_grad()
    def verify(
        self,
        candidate_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> Tuple[torch.Tensor, KVCache]:
        """Verify *K* draft candidates in one parallel forward pass.

        The model processes all candidate tokens at once (thanks to causal
        masking), returning logits at each position.

        Args:
            candidate_ids: Draft token IDs, shape ``(1, K)``.
            kv_cache: KV cache containing the prefix up to (but not
                      including) the candidates.

        Returns:
            Tuple of ``(logits, updated_kv_cache)`` where logits has shape
            ``(1, K, vocab_size)``.
        """
        logits, new_past = self.forward(candidate_ids, kv_cache.get())
        kv_cache.update(new_past)
        return logits, kv_cache

    @torch.no_grad()
    def init_cache(
        self, input_ids: torch.Tensor, kv_cache: KVCache
    ) -> Tuple[torch.Tensor, KVCache]:
        """Process the prompt through the target model to build the initial KV cache.

        Args:
            input_ids: Prompt token IDs, shape ``(1, prompt_len)``.
            kv_cache: Empty or partially filled KV cache.

        Returns:
            Tuple of ``(last_logits, updated_cache)`` where ``last_logits``
            has shape ``(1, vocab_size)`` and represents the target model's
            prediction for the next token after the prompt.
        """
        logits, new_past = self.forward(input_ids, kv_cache.get())
        kv_cache.update(new_past)
        return logits[:, -1, :], kv_cache
