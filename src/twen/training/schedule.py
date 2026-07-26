"""Token-based schedules that can be serialized without framework state."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SparseTopKSchedule:
    """Use all experts, then half, then the target top-k during warm sparsification."""

    num_experts: int = 8
    target_top_k: int = 2
    anneal_fraction: float = 0.2

    def value(self, consumed_tokens: int, max_tokens: int) -> int:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if consumed_tokens < 0:
            raise ValueError("consumed_tokens cannot be negative")
        if not 0.0 < self.anneal_fraction <= 1.0:
            raise ValueError("anneal_fraction must be in (0, 1]")
        progress = min(consumed_tokens / max_tokens, 1.0)
        first_cut = self.anneal_fraction / 3.0
        second_cut = 2.0 * self.anneal_fraction / 3.0
        if progress < first_cut:
            return self.num_experts
        if progress < second_cut:
            return max(self.target_top_k, self.num_experts // 2)
        return self.target_top_k


@dataclass(frozen=True, slots=True)
class TokenCosineSchedule:
    """Linear warmup followed by cosine decay, parameterized by token count."""

    warmup_tokens: int
    max_tokens: int
    min_ratio: float = 0.1

    def ratio(self, consumed_tokens: int) -> float:
        if self.warmup_tokens < 0 or self.max_tokens <= 0:
            raise ValueError("invalid token schedule")
        if self.warmup_tokens >= self.max_tokens:
            raise ValueError("warmup_tokens must be smaller than max_tokens")
        tokens = max(0, min(consumed_tokens, self.max_tokens))
        if tokens < self.warmup_tokens:
            return tokens / max(1, self.warmup_tokens)
        progress = (tokens - self.warmup_tokens) / (self.max_tokens - self.warmup_tokens)
        return self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True, slots=True)
class TokenWarmupStableDecaySchedule:
    """Linear warmup, a stable plateau, then a token-exact cosine cooldown.

    ``decay_tokens`` counts backwards from ``max_tokens``.  This makes the
    stable/decay boundary independent of optimizer-step or microbatch geometry
    while still landing exactly on ``min_ratio`` at the configured token
    budget.
    """

    warmup_tokens: int
    max_tokens: int
    decay_tokens: int
    min_ratio: float = 0.1

    def ratio(self, consumed_tokens: int) -> float:
        if self.warmup_tokens < 0 or self.max_tokens <= 0:
            raise ValueError("invalid token schedule")
        if self.warmup_tokens >= self.max_tokens:
            raise ValueError("warmup_tokens must be smaller than max_tokens")
        if self.decay_tokens <= 0:
            raise ValueError("decay_tokens must be positive")
        if self.decay_tokens > self.max_tokens - self.warmup_tokens:
            raise ValueError("decay_tokens overlaps the warmup interval")
        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError("min_ratio must be in [0, 1]")

        tokens = max(0, min(consumed_tokens, self.max_tokens))
        if tokens < self.warmup_tokens:
            return tokens / max(1, self.warmup_tokens)

        decay_start = self.max_tokens - self.decay_tokens
        if tokens <= decay_start:
            return 1.0

        progress = (tokens - decay_start) / self.decay_tokens
        return self.min_ratio + (1.0 - self.min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
