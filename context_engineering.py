"""
context_engineering.py
----------------------
The Orchestrator — ties retrieval, memory, and compression together.

v1.1.0 Final - Hybrid retrieval as default + Re-ranking
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from retriever import Document, Retriever, ScoredDocument
from memory import Memory, Turn
from compressor import Compressor, TokenBudget, estimate_tokens

logger = logging.getLogger(__name__)


@dataclass
class ContextPacket:
    query: str
    retrieved_docs: List[ScoredDocument]
    memory_turns: List[Turn]
    compressed_text: str
    budget_summary: dict
    metadata: dict = field(default_factory=dict)

    def to_prompt_string(self, include_history: bool = True) -> str:
        parts: List[str] = []

        parts.append("=== CONTEXT ===")
        parts.append(self.compressed_text if self.compressed_text else "(no relevant context found)")

        if include_history and self.memory_turns:
            parts.append("\n=== CONVERSATION HISTORY ===")
            for turn in self.memory_turns:
                parts.append(f"[{turn.role.upper()}]: {turn.content}")

        parts.append("\n=== QUERY ===")
        parts.append(self.query)

        return "\n".join(parts)

    def diagnostics(self) -> str:
        lines = [
            f"Query           : {self.query!r}",
            f"Docs retrieved  : {len(self.retrieved_docs)}",
            f"History turns   : {len(self.memory_turns)}",
            f"Compressed chars: {len(self.compressed_text)}",
            "Token budget    :",
        ]
        for slot, tokens in self.budget_summary.items():
            if not str(slot).startswith("_"):
                lines.append(f"  {slot:<20} {tokens} tokens")
        if self.metadata:
            lines.append("Metadata        :")
            for k, v in self.metadata.items():
                lines.append(f"  {k:<20} {v}")
        return "\n".join(lines)


class ContextEngine:
    _SYSTEM_SLOT = "system_prompt"
    _HISTORY_SLOT = "history"
    _DOCS_SLOT = "retrieved_docs"

    def __init__(
        self,
        documents: Optional[List[Document]] = None,
        total_token_budget: int = 2048,
        system_prompt: str = "You are a helpful assistant.",
        retrieval_top_k: int = 5,
        retrieval_mode: str = "hybrid",           # Changed to hybrid as default
        compression_strategy: str = "extractive",
        memory_short_term: int = 4,
        memory_decay_rate: float = 0.001,
        hybrid_alpha: float = 0.65,
    ) -> None:
        if total_token_budget < 64:
            raise ValueError(f"total_token_budget must be >= 64, got {total_token_budget}.")
        if retrieval_top_k < 1:
            raise ValueError(f"retrieval_top_k must be >= 1, got {retrieval_top_k}.")

        self.system_prompt = system_prompt
        self.total_token_budget = total_token_budget
        self.retrieval_top_k = retrieval_top_k
        self.compression_strategy = compression_strategy
        self.hybrid_alpha = hybrid_alpha

        self._retriever = Retriever(
            documents=documents or [],
            mode=retrieval_mode,
        )
        self._memory = Memory(
            short_term_size=memory_short_term,
            decay_rate=memory_decay_rate,
        )

        logger.info(
            "ContextEngine initialised: budget=%d, docs=%d, mode=%s, strategy=%s, alpha=%.2f",
            total_token_budget, len(documents or []), retrieval_mode, compression_strategy, hybrid_alpha
        )

    def add_document(self, doc: Document) -> None:
        self._retriever.add_document(doc)

    def remember(self, role: str, content: str, importance: Optional[float] = None) -> bool:
        return self._memory.add(role=role, content=content, importance=importance)

    def build(self, query: str) -> ContextPacket:
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        budget = TokenBudget(total=self.total_token_budget)
        budget.reserve_text(self._SYSTEM_SLOT, self.system_prompt)

        # Retrieve + Re-rank
        scored_docs = self._retriever.retrieve(
            query,
            top_k=self.retrieval_top_k * 2,
            alpha=self.hybrid_alpha
        )
        scored_docs = self._rerank(scored_docs, query)[:self.retrieval_top_k]

        raw_chunks = [sd.document.content for sd in scored_docs]

        memory_turns = self._memory.get_weighted(query=query)
        history_text = " ".join(t.content for t in memory_turns)

        budget.reserve_text(self._HISTORY_SLOT, history_text)

        remaining_chars = budget.remaining_chars()
        compressor = Compressor(max_chars=max(remaining_chars, 1), strategy=self.compression_strategy)
        compression_result = compressor.compress(chunks=raw_chunks, query=query)

        budget.reserve_text(self._DOCS_SLOT, compression_result.text)

        return ContextPacket(
            query=query,
            retrieved_docs=scored_docs,
            memory_turns=memory_turns,
            compressed_text=compression_result.text,
            budget_summary=budget.summary(),
            metadata={
                "compression_ratio": compression_result.compression_ratio,
                "tokens_saved": compression_result.estimated_tokens_saved,
                "strategy_used": compression_result.strategy_used,
                "memory_turns_total": len(self._memory),
                "retrieval_mode": self._retriever.mode,
            },
        )

    def _rerank(self, scored_docs: List[ScoredDocument], query: str) -> List[ScoredDocument]:
        """Multi-factor re-ranking"""
        if not scored_docs:
            return []

        reranked = []
        for sd in scored_docs:
            doc = sd.document
            base = sd.score
            importance = 1.4 if any(tag in doc.tags for tag in ["memory", "context", "rag", "embedding"]) else 1.0
            final_score = base * 0.68 + importance * 0.32

            reranked.append(ScoredDocument(
                document=doc,
                score=round(final_score, 4),
                match_reason=sd.match_reason + " → reranked"
            ))

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    def memory_summary(self) -> dict:
        return self._memory.summary()

    def clear_memory(self) -> None:
        self._memory.clear()

    def __repr__(self) -> str:
        return f"ContextEngine(docs={len(self._retriever.documents)}, memory={len(self._memory)} turns, budget={self.total_token_budget} tokens, mode={self._retriever.mode})"
