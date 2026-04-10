"""
Context Engineering — Pure Python context management for LLMs.

Exposes the full public API from a single import:

    from context_engineering import (
        ContextEngine,
        ContextPacket,
        Document,
        Retriever,
        ScoredDocument,
        Memory,
        Turn,
        Compressor,
        TokenBudget,
        CompressionResult,
        estimate_tokens
    )

Version: 1.1.0
"""

from .compressor import (
    Compressor,
    TokenBudget,
    CompressionResult,
    estimate_tokens
)
from .context_engineering import ContextEngine, ContextPacket
from .memory import Memory, Turn
from .retriever import Document, Retriever, ScoredDocument

__version__ = "1.1.0"

__all__ = [
    # Main Orchestrator
    "ContextEngine",
    "ContextPacket",

    # Retrieval
    "Document",
    "Retriever",
    "ScoredDocument",

    # Memory
    "Memory",
    "Turn",

    # Compression
    "Compressor",
    "TokenBudget",
    "CompressionResult",
    "estimate_tokens",
]
