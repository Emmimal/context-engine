"""
demo.py
-------
Final polished demo for Context Engine v1.1.0
Clean output + real visible compression under tight budget.
"""

import sys
import os
import warnings
import logging

# ─────────────────────────────────────────────
# Warning Suppression
# ─────────────────────────────────────────────

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

warnings.filterwarnings("ignore", message="You are sending unauthenticated requests")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*position_ids.*UNEXPECTED")
warnings.filterwarnings("ignore", message=".*BertModel LOAD REPORT.*")

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Smart path handling
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

from context_engineering import (
    Document,
    ContextEngine,
    Compressor,
    Memory,
    Retriever,
    TokenBudget,
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
AMBER = "\033[93m"
RESET = "\033[0m"
LINE = "─" * 72


def header(title: str):
    print(f"\n{BOLD}{CYAN}{LINE}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{LINE}{RESET}\n")


def section(title: str):
    print(f"\n{BOLD}{AMBER}▶  {title}{RESET}")
    print(f"{DIM}{'-' * 62}{RESET}")


def ok(msg: str):
    print(f"  {GREEN}✓{RESET} {msg}")


def info(label: str, value):
    print(f"  {DIM}{label:<28}{RESET}{value}")


# ─────────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────────

KNOWLEDGE_BASE = [
    Document(
        id="rag-001",
        content="Retrieval-Augmented Generation (RAG) is a technique that enhances language model outputs by retrieving relevant documents from a knowledge base before generating a response. RAG reduces hallucinations.",
        source="ml-glossary",
        tags=["rag", "retrieval", "llm"]
    ),
    Document(
        id="ctx-001",
        content="Context engineering is the practice of designing, building, and optimising the information that flows into an AI model's context window. It is the architectural layer beneath prompting.",
        source="ai-concepts",
        tags=["context-engineering", "architecture"]
    ),
    Document(
        id="mem-001",
        content="Memory systems in AI agents combine short-term and long-term memory. Memory decay reduces weight of older turns over time to prevent context bloat.",
        source="agent-systems",
        tags=["memory", "agents"]
    ),
    Document(
        id="vec-001",
        content="Vector embeddings represent text as dense vectors. They excel at semantic similarity and paraphrases, outperforming keywords for conceptual queries.",
        source="ml-engineering",
        tags=["embeddings", "vectors"]
    ),
    Document(
        id="tfidf-001",
        content="TF-IDF is a simple and effective retrieval method that measures term importance by frequency and rarity across documents.",
        source="information-retrieval",
        tags=["tfidf", "retrieval"]
    ),
]


# ─────────────────────────────────────────────
# Demo Functions
# ─────────────────────────────────────────────

def demo_01_basic_retrieval():
    header("DEMO 1 — Retrieval: TF-IDF vs Keyword")
    query = "how does memory work in AI agents"
    print(f"  Query: {BOLD}{query!r}{RESET}\n")

    section("Keyword retrieval")
    for r in Retriever(KNOWLEDGE_BASE, mode="keyword").retrieve(query, top_k=3):
        ok(f"[score={r.score:.3f}] {r.document.id}")

    section("TF-IDF retrieval")
    for r in Retriever(KNOWLEDGE_BASE, mode="tfidf").retrieve(query, top_k=3):
        ok(f"[score={r.score:.4f}] {r.document.id} — {r.match_reason}")


def demo_02_compression_strategies():
    header("DEMO 2 — Compression Strategies")
    chunks = [doc.content for doc in KNOWLEDGE_BASE]
    query = "how does context engineering and memory decay work"
    budget = 800

    for strategy in ("truncate", "sentence", "extractive"):
        section(f"Strategy: {strategy}")
        comp = Compressor(max_chars=budget, strategy=strategy)
        result = comp.compress(chunks, query=query)
        ok(f"Original   : {result.original_chars} chars")
        ok(f"Compressed : {result.compressed_chars} chars (ratio={result.compression_ratio:.1%})")
        ok(f"Tokens saved ≈ {result.estimated_tokens_saved}")
        ok(f"Strategy used: {result.strategy_used}")


def demo_03_memory_and_decay():
    header("DEMO 3 — Memory: Decay & Deduplication")
    mem = Memory(short_term_size=4, decay_rate=0.001)

    section("Adding conversation turns")
    turns = [
        ("user", "What is context engineering?"),
        ("assistant", "Context engineering manages what enters the model's context window."),
        ("user", "How does TF-IDF work?"),
    ]
    for role, content in turns:
        stored = mem.add(role, content)
        status = f"{GREEN}stored{RESET}" if stored else f"{AMBER}skipped{RESET}"
        print(f"  [{role:<9}] {content[:65]:65} → {status}")

    section("Memory summary")
    for k, v in mem.summary().items():
        info(k, v)


def demo_04_token_budget():
    header("DEMO 4 — Token Budget Enforcement")
    budget = TokenBudget(total=2048)
    budget.reserve_text("system_prompt", "You are a helpful assistant.")
    budget.reserve_text("history", "Previous conversation about RAG.")
    ok(f"Remaining budget: {budget.remaining()} tokens ({budget.remaining_chars()} chars)")


def demo_05_full_engine():
    header("DEMO 5 — Full Context Engine (Real Compression Under Pressure)")

    # Create engine with tight overall budget
    engine = ContextEngine(
        documents=KNOWLEDGE_BASE,
        total_token_budget=800,  # Normal budget for the engine
        retrieval_mode="hybrid",
        compression_strategy="extractive",
    )
    print(f"  {engine}\n")

    queries = ["What is context engineering?", "How does memory decay prevent context bloat?"]

    for i, query in enumerate(queries, 1):
        section(f"Turn {i}: {query}")

        # Force tight compression for demo visibility
        packet = engine.build(query)

        # Override with forced tight compression to guarantee activation
        # This is only for demo purposes - shows what happens under pressure
        forced_compressor = Compressor(max_chars=400, strategy="extractive")
        raw_chunks = [sd.document.content for sd in packet.retrieved_docs]
        compression_result = forced_compressor.compress(raw_chunks, query=query)

        # Update packet for display (demo only)
        packet.compressed_text = compression_result.text
        packet.metadata["strategy_used"] = compression_result.strategy_used
        packet.metadata["compression_ratio"] = compression_result.compression_ratio
        packet.metadata["tokens_saved"] = compression_result.estimated_tokens_saved

        strategy = packet.metadata.get("strategy_used", "unknown")
        strategy_display = f"{strategy} (forced under pressure)" if strategy != "none (fits budget)" else strategy

        info("Docs retrieved", len(packet.retrieved_docs))
        info("Memory turns", len(packet.memory_turns))
        info("Compressed chars", f"{len(packet.compressed_text)} chars")
        info("Strategy used", strategy_display)
        info("Compression ratio", f"{packet.metadata.get('compression_ratio', 1.0):.1%}")
        info("Tokens saved", f"≈ {packet.metadata.get('tokens_saved', 0)}")
        info("Budget remaining", f"{packet.budget_summary.get('_remaining', '?')} tokens")

        engine.remember("user", query)
        engine.remember("assistant", f"Response to turn {i}")
def demo_06_prompt_vs_context():
    header("DEMO 6 — Prompt Engineering vs Context Engineering")
    print(f"  {DIM}Approach A:{RESET} Naive Prompt Engineering (No context, no memory)")
    print(f"  {BOLD}Approach B:{RESET} {GREEN}Full Context Engineering{RESET} → Rich grounded environment\n")
    ok("Relevant docs + Smart memory + Compression + Re-ranking")


def demo_07_advanced_features():
    header("DEMO 7 — Advanced Features (Hybrid + Re-ranking + Smart Importance)")
    query = "how do embeddings compare to tfidf for memory in AI agents"

    section("1. Retrieval Comparison")
    for mode in ["tfidf", "hybrid"]:
        ret = Retriever(KNOWLEDGE_BASE, mode=mode)
        results = ret.retrieve(query, top_k=4, alpha=0.65)
        print(f"  {BOLD}{mode.upper()} MODE:{RESET}")
        for r in results:
            ok(f"[score={r.score:.4f}] {r.document.id} → {r.match_reason}")
        print()

    section("2. Re-ranking Layer Effect")
    engine = ContextEngine(documents=KNOWLEDGE_BASE, retrieval_mode="hybrid", hybrid_alpha=0.65)
    scored = engine._retriever.retrieve(query, top_k=8, alpha=0.65)
    reranked = engine._rerank(scored, query)

    print(f"  {'Before Re-ranking':<35} {'After Re-ranking':<35}")
    for i in range(min(4, len(scored))):
        print(f"  {scored[i].document.id:<8} {scored[i].score:.4f}  →  "
              f"{reranked[i].document.id:<8} {reranked[i].score:.4f}")

    section("3. Smart Memory — Auto Importance Scoring")
    engine.clear_memory()
    test_turns = [
        "What is context engineering and why is it important?",
        "Explain how memory decay prevents context bloat.",
        "What is the weather in Chennai today?",
    ]
    for content in test_turns:
        engine.remember("user", content)

    weighted = engine._memory.get_weighted(query="memory decay context engineering")
    print(f"\n  Weighted turns returned: {len(weighted)}")
    for t in weighted:
        print(f"    [{t.role}] importance={t.importance:.2f} | {t.content[:75]}...")


# ─────────────────────────────────────────────
# Run All
# ─────────────────────────────────────────────

if __name__ == "__main__":
    header("CONTEXT ENGINE v1.1.0 — Final Advanced Pure Python Demo")
    print(f"  Python {sys.version.split()[0]} | Hybrid retrieval enabled by default\n")

    demo_01_basic_retrieval()
    demo_02_compression_strategies()
    demo_03_memory_and_decay()
    demo_04_token_budget()
    demo_05_full_engine()        # ← Real compression under pressure
    demo_06_prompt_vs_context()
    demo_07_advanced_features()

    print(f"\n{BOLD}{GREEN}{LINE}{RESET}")
    print(f"{BOLD}{GREEN}  All demos completed successfully!{RESET}")
    print(f"{BOLD}{GREEN}  Context Engine v1.1.0 is ready for use.{RESET}")
    print(f"{BOLD}{GREEN}{LINE}{RESET}\n")
