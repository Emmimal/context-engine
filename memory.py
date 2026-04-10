"""
memory.py
---------
Conversational Memory with Decay, Deduplication, and Smart Importance Scoring.

New in v1.1.0:
- Automatic importance scoring based on content + query relevance
- last_accessed timestamp for freshness boost
- final_score = importance × recency × freshness
- importance can now be > 1.0 for very important turns
- Smart _calculate_importance() heuristic
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """One conversational exchange (user or assistant)."""
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    last_accessed: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant", "system"):
            raise ValueError(
                f"Turn.role must be 'user', 'assistant', or 'system', got {self.role!r}."
            )
        # Allow importance > 1.0 for high-value turns
        self.importance = max(0.0, self.importance)

    def age_seconds(self) -> float:
        """Seconds elapsed since this turn was created."""
        return time.time() - self.timestamp

    def touch(self) -> None:
        """Update last_accessed timestamp."""
        self.last_accessed = time.time()

    def __repr__(self) -> str:
        preview = self.content[:50].replace("\n", " ")
        return f"Turn(role={self.role!r}, importance={self.importance:.2f}, preview={preview!r}...)"


class Memory:
    """
    Conversational memory with short-term / long-term split and smart decay.

    Parameters
    ----------
    short_term_size  : int
        Number of most-recent turns always included.
    decay_rate       : float
        Exponential decay rate for long-term memory.
    min_importance   : float
        Minimum effective score for long-term turns to be included.
    dedup_threshold  : float
        Jaccard similarity threshold for deduplication.
    """

    def __init__(
            self,
            short_term_size: int = 4,
            decay_rate: float = 0.001,
            min_importance: float = 0.1,
            dedup_threshold: float = 0.72,
    ) -> None:
        if short_term_size < 1:
            raise ValueError(f"short_term_size must be >= 1, got {short_term_size}.")
        if decay_rate < 0.0:
            raise ValueError(f"decay_rate must be >= 0.0, got {decay_rate}.")
        if min_importance < 0.0:
            raise ValueError(f"min_importance must be >= 0.0, got {min_importance}.")
        if not (0.0 <= dedup_threshold <= 1.0):
            raise ValueError(f"dedup_threshold must be in [0.0, 1.0], got {dedup_threshold}.")

        self.short_term_size = short_term_size
        self.decay_rate = decay_rate
        self.min_importance = min_importance
        self.dedup_threshold = dedup_threshold
        self._history: List[Turn] = []

    # ── Public API ────────────────────────────

    def add(self, role: str, content: str, importance: Optional[float] = None) -> bool:
        """
        Add a conversation turn to memory.
        If importance is None, it will be automatically calculated.
        Returns True if stored, False if deduplicated.
        """
        if importance is None:
            importance = self._calculate_importance(content)

        if self._is_duplicate(content):
            logger.debug("Memory.add(): duplicate skipped — role=%s", role)
            return False

        self._history.append(Turn(role=role, content=content, importance=importance))
        return True

    def get_recent(self, n: Optional[int] = None) -> List[Turn]:
        """Return the n most recent turns."""
        n = n if n is not None else self.short_term_size
        return list(self._history[-n:])

    def get_weighted(self, query: Optional[str] = None) -> List[Turn]:
        """
        Return weighted memory turns.
        Short-term turns are always included.
        Long-term turns use: importance × recency × freshness
        """
        if not self._history:
            return []

        short_term = self._history[-self.short_term_size:]
        long_term = self._history[:-self.short_term_size]

        weighted_long_term: List[Turn] = []
        current_time = time.time()

        for turn in long_term:
            age = turn.age_seconds()
            recency = math.exp(-self.decay_rate * age)

            # Freshness boost if recently accessed
            freshness = math.exp(-0.01 * (current_time - turn.last_accessed))

            effective = turn.importance * recency * freshness

            # Query relevance boost
            if query:
                boost = self._relevance_boost(query, turn.content)
                effective += boost

            if effective >= self.min_importance:
                weighted_long_term.append(turn)
                turn.touch()  # update last_accessed

        return weighted_long_term + short_term

    def summary(self) -> dict:
        """Snapshot of memory state for diagnostics."""
        total = len(self._history)
        short = min(self.short_term_size, total)
        long = max(0, total - short)
        return {
            "total_turns": total,
            "short_term": short,
            "long_term_candidates": long,
            "decay_rate": self.decay_rate,
            "min_importance": self.min_importance,
            "dedup_threshold": self.dedup_threshold,
        }

    def clear(self) -> None:
        """Erase all stored turns."""
        self._history.clear()
        logger.debug("Memory cleared.")

    # ── Internal helpers ──────────────────────

    def _calculate_importance(self, content: str, query: Optional[str] = None) -> float:
        """
        Automatic importance scoring.
        Higher score = more likely to be retained in long-term memory.
        """
        score = 1.0
        lower_content = content.lower()
        words = lower_content.split()

        # Length bonus (longer, more informative turns)
        score += min(len(words) / 70, 0.8)

        # Keyword bonus
        key_terms = {
            "how", "why", "explain", "important", "key", "remember", "definition",
            "difference", "advantage", "memory", "context", "rag", "embedding",
            "hybrid", "decay", "tfidf", "compression"
        }
        score += 0.4 * sum(1 for term in key_terms if term in lower_content)

        # Query relevance (if provided)
        if query:
            q_tokens = set(query.lower().split())
            c_tokens = set(words)
            if q_tokens:
                overlap = len(q_tokens & c_tokens) / len(q_tokens)
                score += overlap * 0.9

        return min(2.5, score)  # Cap at 2.5x importance

    def _is_duplicate(self, content: str) -> bool:
        """Robust deduplication logic (your excellent original version)."""
        if not self._history:
            return False

        new = content.strip().lower()
        if not new:
            return True

        new_tokens = set(new.split())

        for turn in self._history[-12:]:
            old = turn.content.strip().lower()
            if not old:
                continue

            # 1. Exact containment
            if new in old or old in new:
                return True

            # 2. Strong prefix overlap
            half = len(new) // 2
            if len(new) > 10 and half > 0 and new[:half] == old[:half]:
                return True

            # 3. Jaccard similarity
            old_tokens = set(old.split())
            union = len(new_tokens | old_tokens)
            if union > 0:
                jaccard = len(new_tokens & old_tokens) / union
                if jaccard >= self.dedup_threshold:
                    return True

        return False

    def _relevance_boost(self, query: str, content: str) -> float:
        """Small relevance boost for query-aware retrieval."""
        q_tokens = set(query.lower().split())
        c_tokens = set(content.lower().split())
        if not q_tokens or not c_tokens:
            return 0.0
        return len(q_tokens & c_tokens) / len(q_tokens) * 0.35

    # ── Dunder helpers ────────────────────────

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"Memory(turns={len(self._history)}, "
            f"decay_rate={self.decay_rate}, "
            f"short_term_size={self.short_term_size})"
        )
