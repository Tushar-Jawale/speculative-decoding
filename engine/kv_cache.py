"""Custom KV cache with rollback and checkpoint support.

Wraps HuggingFace's ``past_key_values`` tuple-of-tuples format and adds:
- ``rollback(n_tokens)``: discard the last *n* KV entries without recomputation
- ``save_checkpoint()`` / ``restore_checkpoint()``: snapshot state before each
  speculation cycle so we can undo on rejection
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# Type alias: each layer stores (key, value) tensors shaped
# (batch, n_heads, seq_len, head_dim).
PastKeyValues = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


class KVCache:
    """Key-Value cache manager for transformer models.

    Provides efficient rollback and checkpointing over the standard
    HuggingFace ``past_key_values`` representation.
    """

    def __init__(self) -> None:
        self._cache: Optional[PastKeyValues] = None
        self._checkpoint: Optional[PastKeyValues] = None

    # ── properties ────────────────────────────────────────────

    @property
    def seq_len(self) -> int:
        """Current number of cached tokens (sequence length dimension)."""
        if self._cache is None:
            return 0
        return self._cache[0][0].shape[2]

    @property
    def num_layers(self) -> int:
        """Number of transformer layers in the cache."""
        if self._cache is None:
            return 0
        return len(self._cache)

    def __len__(self) -> int:
        return self.seq_len

    # ── core operations ───────────────────────────────────────

    def update(self, new_past_key_values: PastKeyValues) -> None:
        """Replace the cache with new past_key_values from a model forward pass.

        Args:
            new_past_key_values: HuggingFace-style ``past_key_values``.
        """
        self._cache = new_past_key_values

    def get(self) -> Optional[PastKeyValues]:
        """Return the raw ``past_key_values`` for use in model forward calls."""
        return self._cache

    def rollback(self, n_tokens: int) -> None:
        """Discard the last *n_tokens* from every layer.

        This is the key operation for speculative decoding: when the
        target model rejects a draft token at position *i*, we truncate
        the cache back to the accepted prefix without re-running any
        earlier computation.

        Args:
            n_tokens: Number of trailing tokens to remove.

        Raises:
            ValueError: If *n_tokens* exceeds the current cache length.
        """
        if self._cache is None or n_tokens <= 0:
            return
        if n_tokens > self.seq_len:
            raise ValueError(
                f"Cannot rollback {n_tokens} tokens from cache of length {self.seq_len}"
            )
        end = self.seq_len - n_tokens
        self._cache = tuple(
            (k[:, :, :end, :], v[:, :, :end, :]) for k, v in self._cache
        )

    def trim_to(self, length: int) -> None:
        """Trim the cache so that exactly *length* tokens remain.

        Args:
            length: Desired sequence length.
        """
        if self._cache is None:
            return
        self._cache = tuple(
            (k[:, :, :length, :], v[:, :, :length, :]) for k, v in self._cache
        )

    # ── checkpointing ─────────────────────────────────────────

    def save_checkpoint(self) -> None:
        """Deep-copy the current cache state for later restoration.

        Call this at the start of each speculation cycle.
        """
        if self._cache is None:
            self._checkpoint = None
        else:
            self._checkpoint = tuple(
                (k.clone(), v.clone()) for k, v in self._cache
            )

    def restore_checkpoint(self) -> None:
        """Restore the cache to the last saved checkpoint.

        Useful when the entire speculation cycle must be abandoned.
        """
        if self._checkpoint is not None:
            self._cache = self._checkpoint
            self._checkpoint = None

    def clear(self) -> None:
        """Discard the cache and any saved checkpoint."""
        self._cache = None
        self._checkpoint = None
