"""Tests for the KV cache rollback and checkpoint functionality.

Verification criteria:
- After rollback, cache length equals the expected prefix length
- Checkpoint save/restore correctly reverts state
- Rollback of 0 tokens is a no-op
- Rollback beyond cache length raises ValueError
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.kv_cache import KVCache


def _make_cache(n_layers: int = 4, n_heads: int = 8, seq_len: int = 16, head_dim: int = 64) -> KVCache:
    """Create a KVCache with synthetic tensors."""
    cache = KVCache()
    past = tuple(
        (
            torch.randn(1, n_heads, seq_len, head_dim),
            torch.randn(1, n_heads, seq_len, head_dim),
        )
        for _ in range(n_layers)
    )
    cache.update(past)
    return cache


class TestKVCacheBasic:
    """Basic KV cache operations."""

    def test_empty_cache(self) -> None:
        cache = KVCache()
        assert len(cache) == 0
        assert cache.get() is None
        assert cache.num_layers == 0

    def test_update_and_length(self) -> None:
        cache = _make_cache(seq_len=10)
        assert len(cache) == 10
        assert cache.num_layers == 4

    def test_update_replaces(self) -> None:
        cache = _make_cache(seq_len=10)
        assert len(cache) == 10
        cache2 = _make_cache(seq_len=20)
        cache.update(cache2.get())
        assert len(cache) == 20


class TestKVCacheRollback:
    """Tests for the rollback method — the critical operation for spec decoding."""

    def test_rollback_basic(self) -> None:
        """Rollback 4 tokens from a 16-token cache → 12 tokens remain."""
        cache = _make_cache(seq_len=16)
        cache.rollback(4)
        assert len(cache) == 12

    def test_rollback_to_zero(self) -> None:
        """Rollback all tokens → cache is empty."""
        cache = _make_cache(seq_len=8)
        cache.rollback(8)
        assert len(cache) == 0

    def test_rollback_zero_is_noop(self) -> None:
        """Rollback 0 tokens should not change anything."""
        cache = _make_cache(seq_len=16)
        original_values = cache.get()[0][0].clone()
        cache.rollback(0)
        assert len(cache) == 16
        assert torch.equal(cache.get()[0][0], original_values)

    def test_rollback_preserves_prefix(self) -> None:
        """After rollback, the first N tokens are unchanged."""
        cache = _make_cache(seq_len=16)
        original_prefix = cache.get()[0][0][:, :, :12, :].clone()
        cache.rollback(4)
        assert torch.equal(cache.get()[0][0], original_prefix)

    def test_rollback_exceeds_length_raises(self) -> None:
        """Rolling back more tokens than exist should raise ValueError."""
        cache = _make_cache(seq_len=8)
        with pytest.raises(ValueError, match="Cannot rollback"):
            cache.rollback(10)

    def test_rollback_after_speculation_scenario(self) -> None:
        """Simulate: 10-token prefix → extend to 14 (K=4) → reject at position 2.
        
        Expected: cache should have 10 + 2 = 12 tokens after rollback of 2.
        """
        # Start with a 10-token prefix
        cache = _make_cache(seq_len=10)
        assert len(cache) == 10

        # Simulate target model verification extending cache by K=4 tokens
        new_past = tuple(
            (
                torch.randn(1, 8, 14, 64),
                torch.randn(1, 8, 14, 64),
            )
            for _ in range(4)
        )
        cache.update(new_past)
        assert len(cache) == 14

        # Reject at position 2 of 4 → accepted 2, need to rollback 2
        rollback_n = 4 - 2  # K - n_accepted
        cache.rollback(rollback_n)
        assert len(cache) == 12  # prefix (10) + accepted (2)

    def test_rollback_all_rejected(self) -> None:
        """If the first draft token is rejected, rollback all K=4 tokens."""
        cache = _make_cache(seq_len=10)

        # Extend by K=4
        new_past = tuple(
            (torch.randn(1, 8, 14, 64), torch.randn(1, 8, 14, 64))
            for _ in range(4)
        )
        cache.update(new_past)
        assert len(cache) == 14

        # All rejected → rollback all K
        cache.rollback(4)
        assert len(cache) == 10


class TestKVCacheCheckpoint:
    """Tests for checkpoint save/restore."""

    def test_save_and_restore(self) -> None:
        """Save checkpoint, modify cache, restore → original state."""
        cache = _make_cache(seq_len=10)
        original_key = cache.get()[0][0].clone()

        cache.save_checkpoint()

        # Extend the cache (simulating draft generation)
        new_past = tuple(
            (torch.randn(1, 8, 14, 64), torch.randn(1, 8, 14, 64))
            for _ in range(4)
        )
        cache.update(new_past)
        assert len(cache) == 14

        # Restore
        cache.restore_checkpoint()
        assert len(cache) == 10
        assert torch.equal(cache.get()[0][0], original_key)

    def test_restore_without_save_is_noop(self) -> None:
        """Restoring without a prior save should not crash."""
        cache = _make_cache(seq_len=10)
        cache.restore_checkpoint()  # no checkpoint saved → no-op
        assert len(cache) == 10

    def test_checkpoint_independence(self) -> None:
        """Modifying cache after save does not affect the checkpoint."""
        cache = _make_cache(seq_len=10)
        cache.save_checkpoint()

        # Modify in-place via rollback
        cache.rollback(5)
        assert len(cache) == 5

        # Restore should give back the original 10
        cache.restore_checkpoint()
        assert len(cache) == 10


class TestKVCacheTrimTo:
    """Tests for trim_to()."""

    def test_trim_to_shorter(self) -> None:
        cache = _make_cache(seq_len=16)
        cache.trim_to(8)
        assert len(cache) == 8

    def test_trim_to_same_length(self) -> None:
        cache = _make_cache(seq_len=16)
        cache.trim_to(16)
        assert len(cache) == 16

    def test_trim_to_zero(self) -> None:
        cache = _make_cache(seq_len=16)
        cache.trim_to(0)
        assert len(cache) == 0


class TestKVCacheClear:
    """Tests for clear()."""

    def test_clear(self) -> None:
        cache = _make_cache(seq_len=16)
        cache.save_checkpoint()
        cache.clear()
        assert len(cache) == 0
        assert cache.get() is None
