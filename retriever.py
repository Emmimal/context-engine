"""
retriever.py
------------
Context Retrieval & Ranking Engine.

Now supports three modes:
1. keyword  — fast exact matching
2. tfidf    — classic TF-IDF cosine similarity
3. hybrid   — TF-IDF + Vector Embeddings (best of both worlds)

Optional dependency: sentence-transformers (graceful fallback to random embeddings).
"""

import math
import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Optional Embeddings Support
# ─────────────────────────────────────────────

EMBEDDINGS_AVAILABLE = False
SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    pass  # Will use random embeddings fallback
if EMBEDDINGS_AVAILABLE:
    # Optional: silence transformers logging globally when embeddings are used
    logging.getLogger("transformers").setLevel(logging.ERROR)
# ─────────────────────────────────────────────
# Singleton Embedding Engine
# ─────────────────────────────────────────────

_embedding_instance: Optional["EmbeddingEngine"] = None


class EmbeddingEngine:
    """Singleton embedding engine — loads the model only once."""

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        global _embedding_instance
        if _embedding_instance is None:
            _embedding_instance = super().__new__(cls)
            if EMBEDDINGS_AVAILABLE:
                _embedding_instance.model = SentenceTransformer(model_name)
                _embedding_instance.dim = _embedding_instance.model.get_sentence_embedding_dimension()
                logger.info(f"✅ Loaded embedding model: {model_name} ({_embedding_instance.dim} dims)")
            else:
                _embedding_instance.model = None
                _embedding_instance.dim = 384
                logger.warning("⚠️ sentence-transformers not installed. Using random embeddings fallback.")
        return _embedding_instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # __init__ is only called once due to singleton
        pass

    def embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        if self.model is not None:
            return self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
        else:
            # Fallback: random embeddings for demo/testing
            return np.random.rand(len(texts), self.dim).astype(np.float32)

    def similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity (assumes normalized embeddings)."""
        return np.dot(doc_embs, query_emb.T).flatten()


# ─────────────────────────────────────────────
# Stopwords — improves TF-IDF signal quality
# ─────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "are",
    "was", "be", "by", "from", "as", "has", "have", "had", "its",
    "they", "them", "their", "we", "you", "he", "she", "i", "my",
    "your", "our", "how", "what", "which", "who", "when", "where",
    "do", "does", "did", "will", "would", "can", "could", "should",
    "may", "might", "also", "so", "if", "about", "into", "than",
    "more", "such", "both", "each", "all", "no", "any", "there",
})


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class Document:
    """A unit of knowledge in the context engine."""
    id: str
    content: str
    source: str = "unknown"
    tags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Document.id must not be empty.")
        if not self.content or not self.content.strip():
            raise ValueError(f"Document '{self.id}' has empty content.")

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Document(id={self.id!r}, preview={preview!r}...)"


@dataclass
class ScoredDocument:
    """A document paired with its relevance score."""
    document: Document
    score: float
    match_reason: str = ""

    def __repr__(self) -> str:
        return f"ScoredDocument(id={self.document.id!r}, score={self.score:.4f})"


# ─────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [
        t for t in text.split()
        if len(t) > 1 and t not in _STOPWORDS
    ]


def _term_frequency(tokens: List[str]) -> Dict[str, float]:
    """Raw term frequency: count / total tokens."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _cosine_sim(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in vec_a)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
# Retriever — Main Class
# ─────────────────────────────────────────────

class Retriever:
    """
    Supports 'keyword', 'tfidf', and 'hybrid' modes.
    """

    VALID_MODES = ("keyword", "tfidf", "hybrid")

    def __init__(self, documents: Optional[List[Document]] = None, mode: str = "tfidf") -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}.")

        self.mode = mode
        self.documents: List[Document] = []
        self.embedding_engine: Optional[EmbeddingEngine] = None

        # TF-IDF incremental state
        self._doc_freq: Counter = Counter()
        self._n_docs: int = 0

        # Cache for document embeddings (only used in hybrid mode)
        self._doc_embeddings: Optional[np.ndarray] = None

        if mode == "hybrid":
            self.embedding_engine = EmbeddingEngine()

        # Add initial documents
        for doc in documents or []:
            self.add_document(doc)

        logger.debug(
            "Retriever initialised: mode=%s, docs=%d, vocab=%d",
            mode, self._n_docs, len(self._doc_freq)
        )

    # ── Indexing ──────────────────────────────

    def _index_document(self, doc: Document) -> None:
        """Index document for TF-IDF and (if hybrid) cache its embedding."""
        self.documents.append(doc)
        self._n_docs += 1

        # TF-IDF indexing
        for term in set(_tokenize(doc.content)):
            self._doc_freq[term] += 1

        # Hybrid: Update cached embeddings if engine exists
        if self.embedding_engine and self._doc_embeddings is not None:
            # Recompute all embeddings when a new doc is added (simple but correct)
            self._compute_doc_embeddings()

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        return math.log((1 + self._n_docs) / (1 + df)) + 1.0

    def _compute_doc_embeddings(self) -> None:
        """Pre-compute and cache embeddings for all documents (hybrid mode only)."""
        if not self.embedding_engine:
            return
        contents = [doc.content for doc in self.documents]
        self._doc_embeddings = self.embedding_engine.embed(contents)

    # ── Public API ────────────────────────────

    def add_document(self, doc: Document) -> None:
        """Incrementally add a document and update indexes/caches."""
        self._index_document(doc)
        logger.debug("Document added: id=%s, total=%d", doc.id, self._n_docs)

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.65) -> List[ScoredDocument]:
        """
        Main retrieval method.
        alpha: weight for embeddings when using hybrid mode (0.0 = pure TF-IDF, 1.0 = pure embeddings)
        """
        if not query or not query.strip():
            logger.warning("retrieve() called with empty query.")
            return []

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        if self.mode == "keyword":
            return self._keyword_retrieve(query, top_k)
        elif self.mode == "hybrid":
            return self._hybrid_retrieve(query, top_k, alpha)
        else:  # tfidf
            return self._tfidf_retrieve(query, top_k)

    # ── Retrieval Implementations ─────────────

    def _keyword_retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        results: List[ScoredDocument] = []
        for doc in self.documents:
            doc_tokens = set(_tokenize(doc.content))
            overlap = query_tokens & doc_tokens
            if overlap:
                score = len(overlap) / len(query_tokens)
                results.append(ScoredDocument(
                    document=doc,
                    score=round(score, 4),
                    match_reason=f"keyword overlap: {sorted(overlap)[:5]}"
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _tfidf_retrieve(self, query: str, top_k: int) -> List[ScoredDocument]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_tf = _term_frequency(query_tokens)
        query_vec = {term: tf * self._idf(term) for term, tf in query_tf.items()}

        results: List[ScoredDocument] = []
        for doc in self.documents:
            doc_tokens = _tokenize(doc.content)
            doc_tf = _term_frequency(doc_tokens)
            doc_vec = {term: tf * self._idf(term) for term, tf in doc_tf.items()}

            score = _cosine_sim(query_vec, doc_vec)
            if score > 0.0:
                results.append(ScoredDocument(
                    document=doc,
                    score=round(score, 4),
                    match_reason="tfidf cosine similarity"
                ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _hybrid_retrieve(self, query: str, top_k: int, alpha: float = 0.65) -> List[ScoredDocument]:
        """Hybrid TF-IDF + Embedding retrieval with cached document embeddings."""
        if not self.embedding_engine:
            logger.warning("Hybrid mode requested but embeddings unavailable. Falling back to TF-IDF.")
            return self._tfidf_retrieve(query, top_k)

        # Lazy compute document embeddings (only once)
        if self._doc_embeddings is None:
            self._compute_doc_embeddings()

        # Get TF-IDF scores for candidate boosting
        tfidf_results = self._tfidf_retrieve(query, top_k=12)
        tfidf_dict = {r.document.id: r.score for r in tfidf_results}

        # Embedding scores
        query_emb = self.embedding_engine.embed([query])[0]
        emb_scores = self.embedding_engine.similarity(query_emb, self._doc_embeddings)

        results = []
        for i, doc in enumerate(self.documents):
            tf_score = tfidf_dict.get(doc.id, 0.0)
            emb_score = float(emb_scores[i])
            hybrid_score = alpha * emb_score + (1 - alpha) * tf_score

            results.append(ScoredDocument(
                document=doc,
                score=round(hybrid_score, 4),
                match_reason=f"hybrid (α={alpha:.2f})"
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
