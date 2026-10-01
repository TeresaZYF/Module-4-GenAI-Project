"""
Demo 02: Embedding Model Comparison for Canadian Bank Filings
=============================================================
Instructor demo script - compares OpenAI and local
Sentence-Transformers embeddings for Canadian bank filing retrieval.

This script demonstrates:
- Generating embeddings with OpenAI text-embedding-3-small
- Generating embeddings with local all-MiniLM-L6-v2
- Comparing retrieval quality on Canadian bank filing excerpts
- Cost analysis for production deployment

Prerequisites:
    pip install openai sentence-transformers numpy
    Set OPENAI_API_KEY environment variable

Usage:
    python demos/02_embedding_comparison.py
"""

import os
import json
import time
import numpy as np
from typing import List, Dict

# ──────────────────────────────────────────────────────────
# SAMPLE DATA (Canadian bank filing excerpts)
# ──────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {"text": "Credit risk is the risk of loss associated with a counterparty's inability to fulfill its payment obligations. The provision for credit losses was $3.4 billion for fiscal 2024, an increase from $2.9 billion in the prior year. Gross impaired loans increased to $5.8 billion from $4.2 billion.",
     "bank": "Royal Bank of Canada", "section": "MD&A - Credit Risk"},
    {"text": "Market risk is the risk of loss arising from adverse changes in underlying market factors including interest rates, foreign exchange rates, equity prices, and credit spreads. Average daily VaR for fiscal 2024 was $42 million, compared to $38 million in the prior year.",
     "bank": "Royal Bank of Canada", "section": "MD&A - Market Risk"},
    {"text": "Net interest income for the year was $23.8 billion, an increase of 12% from $21.3 billion in the prior year, driven by higher interest rates and strong volume growth in personal and commercial banking. Net interest margin expanded by 14 basis points to 1.72%.",
     "bank": "Toronto-Dominion Bank", "section": "MD&A - Financial Performance"},
    {"text": "Operational risk is the risk of loss resulting from inadequate or failed internal processes, people, and systems, or from external events. This includes cybersecurity incidents, technology failures, fraud, and third-party vendor dependencies. The bank invested over $800 million in cybersecurity during fiscal 2024.",
     "bank": "Bank of Montreal", "section": "MD&A - Operational Risk"},
    {"text": "The bank's total high-quality liquid assets (HQLA) were $298 billion at October 31, 2024. The Liquidity Coverage Ratio was 132%, well above the OSFI regulatory minimum of 100%. Net Stable Funding Ratio was 114%.",
     "bank": "Royal Bank of Canada", "section": "MD&A - Liquidity Risk"},
    {"text": "In accordance with OSFI Guideline B-15, the bank has enhanced its climate risk management framework. The bank committed to net-zero greenhouse gas emissions in lending and investment portfolios by 2050. Physical risk exposure is concentrated in real estate portfolios vulnerable to flooding and wildfire.",
     "bank": "Bank of Nova Scotia", "section": "MD&A - Climate Risk"},
    {"text": "The residential mortgage portfolio totalled $372 billion, representing 45% of total loans. The proportion of uninsured mortgages was 62%. The bank's exposure to variable-rate mortgages increased as borrowers faced renewal at higher rates.",
     "bank": "Canadian Imperial Bank of Commerce", "section": "MD&A - Credit Risk"},
    {"text": "Cybersecurity threats continue to evolve in sophistication and frequency. The bank invests significantly in cybersecurity defense capabilities including threat detection, incident response, identity management, and employee security awareness training programs.",
     "bank": "Toronto-Dominion Bank", "section": "MD&A - Operational Risk"},
]

TEST_QUERIES = [
    "What is the bank's credit risk exposure and provision for losses?",
    "Tell me about operational risk and cybersecurity",
    "What are the climate risk disclosures?",
    "How much are high-quality liquid assets and what is the LCR?",
    "What is the mortgage portfolio size and variable rate exposure?",
]


# ──────────────────────────────────────────────────────────
# EMBEDDING FUNCTIONS
# ──────────────────────────────────────────────────────────

def embed_openai(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Generate embeddings using OpenAI API."""
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(model=model, input=texts)
    embeddings = np.array([d.embedding for d in response.data])
    return embeddings


def embed_local(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings using local Sentence-Transformers model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of vectors."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def search(query_emb: np.ndarray, doc_embs: np.ndarray,
           chunks: List[Dict], top_k: int = 3) -> List[Dict]:
    """Search for most similar chunks to query."""
    sims = cosine_similarity(query_emb.reshape(1, -1), doc_embs)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "similarity": float(sims[idx]),
            "bank": chunks[idx]["bank"],
            "section": chunks[idx]["section"],
            "text_preview": chunks[idx]["text"][:100],
        })
    return results


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Embedding Model Comparison: Canadian Bank Filings")
    print("=" * 60)

    texts = [c["text"] for c in SAMPLE_CHUNKS]
    print(f"\n  Chunks: {len(texts)}")
    print(f"  Queries: {len(TEST_QUERIES)}")

    # Try local model first (always available)
    local_available = False
    try:
        print("\n  Loading local model (all-MiniLM-L6-v2)...")
        t0 = time.time()
        local_doc_embs = embed_local(texts)
        local_query_embs = embed_local(TEST_QUERIES)
        local_time = time.time() - t0
        local_available = True
        print(f"  Local embedding: {local_doc_embs.shape[1]} dimensions, {local_time:.2f}s")
    except Exception as e:
        print(f"  Local model not available: {e}")
        print("  Install: pip install sentence-transformers")

    # Try OpenAI
    openai_available = False
    if os.environ.get("OPENAI_API_KEY"):
        try:
            print("\n  Calling OpenAI API (text-embedding-3-small)...")
            t0 = time.time()
            openai_doc_embs = embed_openai(texts)
            openai_query_embs = embed_openai(TEST_QUERIES)
            openai_time = time.time() - t0
            openai_available = True
            print(f"  OpenAI embedding: {openai_doc_embs.shape[1]} dimensions, {openai_time:.2f}s")
        except Exception as e:
            print(f"  OpenAI API error: {e}")
    else:
        print("\n  OpenAI skipped (no OPENAI_API_KEY)")

    # Show retrieval results
    for label, doc_embs, query_embs, available in [
        ("LOCAL (all-MiniLM-L6-v2)", locals().get("local_doc_embs"), locals().get("local_query_embs"), local_available),
        ("OPENAI (text-embedding-3-small)", locals().get("openai_doc_embs"), locals().get("openai_query_embs"), openai_available),
    ]:
        if not available:
            continue
        print(f"\n  {'=' * 55}")
        print(f"  RETRIEVAL RESULTS: {label}")
        print(f"  {'=' * 55}")

        for i, query in enumerate(TEST_QUERIES):
            results = search(query_embs[i], doc_embs, SAMPLE_CHUNKS, top_k=3)
            print(f"\n  Q: {query}")
            for r in results:
                print(f"    [{r['similarity']:.3f}] {r['bank']} - {r['section']}")

    # Cost comparison
    print(f"\n{'=' * 60}")
    print("COST COMPARISON")
    print(f"{'=' * 60}")
    total_tokens = sum(len(t.split()) * 1.3 for t in texts)
    print(f"  Total estimated tokens: {int(total_tokens)}")
    print(f"  OpenAI text-embedding-3-small: ${total_tokens * 0.02 / 1_000_000:.6f}")
    print(f"  Local all-MiniLM-L6-v2: $0.00 (runs locally)")
    print(f"\n  At scale (10,000 filing chunks):")
    print(f"  OpenAI: ~${10000 * 500 * 0.02 / 1_000_000:.2f}")
    print(f"  Local: $0.00 (but slower, lower quality)")

    # Key takeaways
    print(f"\n{'=' * 60}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 60}")
    print("1. OpenAI embeddings (1536-dim) vs local (384-dim).")
    print("2. OpenAI is higher quality but costs money.")
    print("3. Local models are free and run offline - good for")
    print("   development and environments where data cannot")
    print("   leave the premises (OSFI B-10 compliance).")
    print("4. Always test on YOUR domain data before choosing.")
    print("5. Canadian financial terms (PCL, OSFI, LCR, AIF)")
    print("   may not be well-represented in general embeddings.")
    print("6. Cost is minimal for most projects - quality matters more.")


if __name__ == "__main__":
    main()
