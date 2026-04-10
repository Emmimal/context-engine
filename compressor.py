"""
compressor.py
-------------
Context Compression & Token Budget Enforcement.

The core problem context engineering solves:
You have 10,000+ characters of potentially relevant context.
Your model can only use ~2,000. What do you keep?

Three strategies:
- TRUNCATE   → Fastest, lossy (head truncation)
- SENTENCE   → Preserves sentence boundaries
- EXTRACTIVE → Query-aware, best quality (recommended)

Pure Python. No external dependencies.
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Literal

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────

Strategy = Literal["truncate", "sentence", "extractive"]


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using punctuation boundaries.
    Preserves the closing punctuation.
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _sentence_score(sentence: str, query: str) -> float:
    """Score sentence by token overlap with the query."""
    if not query or not query.strip():
        return 0.0
    q_tokens = set(query.lower().split())
    s_tokens = set(sentence.lower().split())
    return len(q_tokens & s_tokens) / len(q_tokens) if q_tokens else 0.0


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """
    Rough token estimation (1 token ≈ 4 characters for English prose).
    Pass chars_per_token=3 for code, =2 for dense languages.
    """
    if chars_per_token < 1:
        raise ValueError(f"chars_per_token must be >= 1, got {chars_per_token}.")
    return len(text) // chars_per_token


# ─────────────────────────────────────────────
# Compression Result
# ─────────────────────────────────────────────

@dataclass
class CompressionResult:
    text: str
    original_chars: int
    compressed_chars: int
    strategy_used: str
    estimated_tokens_saved: int

    @property
    def compression_ratio(self) -> float:
        """Fraction of original size retained (1.0 = no compression)."""
        if self.original_chars == 0:
            return 1.0
        return round(self.compressed_chars / self.original_chars, 3)

    def __repr__(self) -> str:
        return (
            f"CompressionResult("
            f"ratio={self.compression_ratio:.1%}, "
            f"chars={self.original_chars}→{self.compressed_chars}, "
            f"tokens_saved≈{self.estimated_tokens_saved})"
        )


# ─────────────────────────────────────────────
# Compressor
# ─────────────────────────────────────────────

class Compressor:
    """
    Compresses retrieved documents to fit within a token/character budget.
    """

    VALID_STRATEGIES: tuple = ("truncate", "sentence", "extractive")

    def __init__(
        self,
        max_chars: int = 1500,
        strategy: Strategy = "extractive",
        min_chunk: int = 20,
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.VALID_STRATEGIES}, got {strategy!r}."
            )
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}.")

        self.max_chars = max_chars
        self.strategy = strategy
        self.min_chunk = min_chunk

    def compress(self, chunks: List[str], query: str = "") -> CompressionResult:
        """
        Main compression method.
        """
        valid_chunks = [c for c in chunks if c and len(c) >= self.min_chunk]

        if not valid_chunks:
            logger.debug("compress(): no valid chunks after filtering.")
            return CompressionResult(
                text="",
                original_chars=0,
                compressed_chars=0,
                strategy_used="none (empty input)",
                estimated_tokens_saved=0,
            )

        original = "\n\n".join(valid_chunks)
        original_chars = len(original)

        # No compression needed
        if original_chars <= self.max_chars:
            logger.debug("compress(): input already fits budget.")
            return CompressionResult(
                text=original,
                original_chars=original_chars,
                compressed_chars=original_chars,
                strategy_used="none (fits budget)",
                estimated_tokens_saved=0,
            )

        # Apply chosen strategy
        if self.strategy == "truncate":
            compressed = self._truncate(valid_chunks)
        elif self.strategy == "sentence":
            compressed = self._sentence(valid_chunks)
        else:  # extractive
            compressed = self._extractive(valid_chunks, query)

        compressed_chars = len(compressed)
        tokens_saved = estimate_tokens(original) - estimate_tokens(compressed)

        logger.debug(
            "compress(): %s | %d → %d chars | ~%d tokens saved",
            self.strategy, original_chars, compressed_chars, tokens_saved
        )

        return CompressionResult(
            text=compressed,
            original_chars=original_chars,
            compressed_chars=compressed_chars,
            strategy_used=self.strategy,
            estimated_tokens_saved=max(0, tokens_saved),
        )

    # ── Strategy implementations ──────────────

    def _truncate(self, chunks: List[str]) -> str:
        """Simple head truncation."""
        budget_per_chunk = self.max_chars // max(len(chunks), 1)
        result = "\n\n".join(chunk[:budget_per_chunk] for chunk in chunks)
        return result[:self.max_chars]

    def _sentence(self, chunks: List[str]) -> str:
        """Greedy sentence-level truncation."""
        all_sentences: List[str] = []
        for chunk in chunks:
            all_sentences.extend(_split_sentences(chunk))

        parts: List[str] = []
        used = 0
        for sentence in all_sentences:
            needed = len(sentence) + (1 if parts else 0)
            if used + needed > self.max_chars:
                break
            parts.append(sentence)
            used += needed

        return " ".join(parts).strip()

    def _extractive(self, chunks: List[str], query: str) -> str:
        """
        Query-aware extractive compression — best quality strategy.
        Improved: consistent separator and cost calculation.
        """
        # Build indexed sentences: (chunk_idx, sent_idx, score, text)
        indexed: List[tuple] = []
        for c_idx, chunk in enumerate(chunks):
            for s_idx, sent in enumerate(_split_sentences(chunk)):
                score = _sentence_score(sent, query)
                indexed.append((c_idx, s_idx, score, sent))

        # Rank by relevance
        ranked = sorted(indexed, key=lambda x: x[2], reverse=True)

        # Greedily select within budget
        budget = self.max_chars
        selected_keys: List[tuple] = []  # (c_idx, s_idx)
        separator = "  "   # Consistent with final join

        for c_idx, s_idx, score, sent in ranked:
            # Cost = sentence length + separator length (except for first sentence)
            cost = len(sent) + (len(separator) if selected_keys else 0)
            if cost <= budget:
                selected_keys.append((c_idx, s_idx))
                budget -= cost
            if budget <= 10:   # Small safety margin
                break

        if not selected_keys:
            logger.warning("_extractive: budget too tight, falling back to truncate.")
            return self._truncate(chunks)

        # Restore original order
        selected_set = set(selected_keys)
        ordered_text = [
            sent
            for c_idx, s_idx, score, sent in indexed
            if (c_idx, s_idx) in selected_set
        ]

        return separator.join(ordered_text).strip()


# ─────────────────────────────────────────────
# Token Budget Enforcer
# ─────────────────────────────────────────────

class TokenBudget:
    """
    Enforces token budget across different context slots.
    """

    def __init__(self, total: int = 2048) -> None:
        if total < 1:
            raise ValueError(f"TokenBudget total must be >= 1, got {total}.")
        self.total = total
        self._used: dict = {}

    def reserve(self, slot: str, tokens: int) -> None:
        if tokens < 0:
            raise ValueError(f"Token reservation for '{slot}' must be >= 0.")
        self._used[slot] = tokens

    def reserve_text(self, slot: str, text: str) -> None:
        self._used[slot] = estimate_tokens(text)

    def remaining(self) -> int:
        return max(0, self.total - sum(self._used.values()))

    def remaining_chars(self) -> int:
        return self.remaining() * 4

    def summary(self) -> dict:
        usage = dict(self._used)
        usage["_remaining"] = self.remaining()
        usage["_total"] = self.total
        return usage

    def __repr__(self) -> str:
        return (
            f"TokenBudget(total={self.total}, "
            f"used={sum(self._used.values())}, "
            f"remaining={self.remaining()})"
        )
