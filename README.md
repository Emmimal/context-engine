# context-engine
A pure-Python context management layer for LLM systems — retrieval, re-ranking, memory decay, and token-budget enforcement in one pipeline.

# context-engine

> A pure-Python context management layer for LLM systems — retrieval, re-ranking, memory decay, and token-budget enforcement in one pipeline.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.1.0-green)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

Most RAG tutorials stop at: retrieve documents, stuff them into a prompt, call the model.
This library handles what comes next — deciding *what* the model actually sees, *how much*
of it, and in *what order*, under real token constraints.

Read the full write-up on Towards Data Science → **[RAG Isn’t Enough — I Built the Missing Layer That Makes LLM Systems Work
]**

---

## What It Does

```
Documents → Retriever → Re-ranker → Compressor → TokenBudget → ContextPacket → LLM
                                         ↑
                                      Memory
```

Five components, one `build()` call:

| Component     | Job                                                              |
|---------------|------------------------------------------------------------------|
| `Retriever`   | keyword / TF-IDF / hybrid (embedding + TF-IDF) retrieval        |
| Re-ranker     | tag-weighted score blending to promote domain-relevant docs      |
| `Memory`      | exponential decay, auto-importance scoring, deduplication        |
| `Compressor`  | truncate / sentence / extractive query-aware compression         |
| `TokenBudget` | slot-based budget enforcer (system → history → docs)             |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/context-engine.git
cd context-engine
pip install numpy                        # required
pip install sentence-transformers        # optional — enables hybrid retrieval
```

No other dependencies. All core functionality runs on the Python standard library + numpy.
If `sentence-transformers` is not installed, hybrid mode falls back to random embeddings
with a warning — useful for development and testing.

---

## Quick Start

```python
from context_engineering import ContextEngine, Document

docs = [
    Document(id="doc-1", content="RAG grounds models in external knowledge.", tags=["rag"]),
    Document(id="doc-2", content="Memory decay prevents context bloat.", tags=["memory"]),
]

engine = ContextEngine(
    documents=docs,
    total_token_budget=800,
    retrieval_mode="hybrid",          # "keyword" | "tfidf" | "hybrid"
    compression_strategy="extractive" # "truncate" | "sentence" | "extractive"
)

# First turn
packet = engine.build("How does memory decay work?")
print(packet.to_prompt_string())
engine.remember("user", "How does memory decay work?")
engine.remember("assistant", "Memory decay reduces the weight of older turns over time.")

# Second turn — memory now competes for budget; compression tightens automatically
packet = engine.build("What happens to irrelevant turns?")
print(packet.diagnostics())
```

---

## Running the Demos

Seven runnable demos covering every component:

```bash
python demo.py
```

| Demo | What It Shows                                      |
|------|----------------------------------------------------|
| 1    | Keyword vs TF-IDF retrieval on the same query      |
| 2    | All three compression strategies side by side      |
| 3    | Memory decay and deduplication                     |
| 4    | Token budget slot enforcement                      |
| 5    | Full engine under tight token pressure             |
| 6    | Prompt engineering vs context engineering contrast |
| 7    | Hybrid retrieval + re-ranking + auto-importance    |

---

## Configuration Reference

```python
ContextEngine(
    documents=[],                  # Initial document list (add more with .add_document())
    total_token_budget=2048,       # Total token budget across all slots
    system_prompt="...",           # Fixed overhead reserved first
    retrieval_top_k=5,             # Documents to keep after re-ranking
    retrieval_mode="hybrid",       # "keyword" | "tfidf" | "hybrid"
    compression_strategy="extractive",
    memory_short_term=4,           # Turns always included regardless of decay
    memory_decay_rate=0.001,       # Exponential decay rate (per second)
    hybrid_alpha=0.65,             # 0.0 = pure TF-IDF, 1.0 = pure embeddings
)
```

**Tuning `hybrid_alpha`:**

| Query type                        | Suggested alpha |
|-----------------------------------|-----------------|
| Exact term lookup                 | 0.3 – 0.4       |
| General / mixed                   | 0.6 – 0.7       |
| Conceptual / paraphrase-heavy     | 0.8 – 0.9       |

---

## Project Structure

```
context-engine/
├── __init__.py               # Public API surface
├── retriever.py              # Retriever + EmbeddingEngine + Document / ScoredDocument
├── memory.py                 # Memory + Turn (decay, dedup, auto-importance)
├── compressor.py             # Compressor + TokenBudget + CompressionResult
├── context_engineering.py    # ContextEngine + ContextPacket (orchestrator)
└── demo.py                   # Seven runnable demos
```

---

## Performance (CPU only, 5-doc knowledge base)

| Operation              | Latency  |
|------------------------|----------|
| Keyword retrieval      | ~0.8 ms  |
| TF-IDF retrieval       | ~2.1 ms  |
| Hybrid retrieval       | ~85 ms   |
| Re-ranking (5 docs)    | ~0.3 ms  |
| Extractive compression | ~4.2 ms  |
| Full `engine.build()`  | ~92 ms   |

Hybrid retrieval dominates latency. For sub-50ms requirements, use `tfidf` or `keyword` mode.
Embedding results are cached after the first call — subsequent queries on the same document
set drop to ~2ms for the embedding step.

---

## When to Use This

**Worth it when you have:**
- Multi-turn conversations where context accumulates across turns
- A large knowledge base where retrieval noise degrades quality
- A tight token budget and quality requirements that outweigh ~92ms overhead

**Skip it when you have:**
- Single-turn queries against a small fixed dataset
- Hard latency requirements under 50ms
- Fully deterministic domains where keyword retrieval is sufficient

---

## Known Limitations

- Token estimation uses 1 token ≈ 4 characters (English prose). Misfires for code and
  non-Latin scripts. Swap in `tiktoken` in `compressor.py` for exact counts.
- The extractive compressor scores sentences by query-token recall overlap, not semantic
  similarity. Sentences that paraphrase the query without sharing tokens score zero.
- `Memory` is in-process only — no persistence across sessions.
- `hybrid_alpha=0.65` is empirically tuned on a small query set. Tune it for your domain.

---

## License

MIT
