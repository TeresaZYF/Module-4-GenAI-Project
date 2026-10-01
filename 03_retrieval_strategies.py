"""
Demo 03: Retrieval Strategies for Canadian Bank Filing Search
=============================================================
Instructor demo script - demonstrates top-k, MMR, and metadata-filtered
retrieval using ChromaDB for Canadian bank filing chunks.

This script demonstrates:
- Setting up ChromaDB with Canadian bank filing metadata
- Top-k similarity search
- Maximal Marginal Relevance (MMR) for diverse results
- Metadata-filtered search (by bank, filing type, section)

Prerequisites:
    pip install chromadb sentence-transformers

Usage:
    python demos/03_retrieval_strategies.py
"""

import json
import time
import numpy as np
from typing import List, Dict

# ──────────────────────────────────────────────────────────
# SAMPLE CANADIAN BANK FILING CHUNKS
# ──────────────────────────────────────────────────────────

CHUNKS = [
    {"id": "rbc_credit_001", "text": "Credit risk is the risk of loss associated with a counterparty's inability to fulfill its payment obligations. The provision for credit losses was $3.4 billion for fiscal 2024, an increase from $2.9 billion in the prior year.",
     "bank": "Royal Bank of Canada", "section": "Credit Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "rbc_market_001", "text": "Market risk is the risk of loss arising from adverse changes in underlying market factors. Average daily VaR for fiscal 2024 was $42 million, compared to $38 million in the prior year.",
     "bank": "Royal Bank of Canada", "section": "Market Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "rbc_cyber_001", "text": "The bank invested over $800 million in cybersecurity and technology resilience capabilities during fiscal 2024. Cybersecurity risk remains a top priority with increasing sophistication of threat actors.",
     "bank": "Royal Bank of Canada", "section": "Operational Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "rbc_liquidity_001", "text": "The bank's total high-quality liquid assets (HQLA) were $298 billion at October 31, 2024. The Liquidity Coverage Ratio was 132%, well above the OSFI regulatory minimum of 100%.",
     "bank": "Royal Bank of Canada", "section": "Liquidity Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "td_credit_001", "text": "Total gross impaired loans were $6.2 billion at October 31, 2024, up from $4.8 billion in 2023. The increase was primarily driven by the US retail and commercial real estate portfolios.",
     "bank": "Toronto-Dominion Bank", "section": "Credit Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "td_climate_001", "text": "TD has committed to achieving net-zero greenhouse gas emissions in its operations and financing activities by 2050. The bank established interim 2030 targets for the energy and power generation sectors.",
     "bank": "Toronto-Dominion Bank", "section": "Climate Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "bmo_credit_001", "text": "Provision for credit losses was $2.3 billion in fiscal 2024, up from $1.5 billion in 2023. The increase reflects integration of Bank of the West portfolio and normalization of credit conditions.",
     "bank": "Bank of Montreal", "section": "Credit Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "bmo_ai_001", "text": "The bank has deployed over 200 AI/ML models in production. In anticipation of OSFI Guideline E-23 (Model Risk Management), the bank established an AI Ethics Committee and enhanced model validation procedures.",
     "bank": "Bank of Montreal", "section": "Operational Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "bns_intl_001", "text": "International Banking revenue grew 8% to $4.1 billion, driven by strong performance in Mexico, Peru, Chile, and Colombia. The Pacific Alliance countries remain core markets for long-term growth.",
     "bank": "Bank of Nova Scotia", "section": "International Banking", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "cibc_mortgage_001", "text": "The residential mortgage portfolio totalled $372 billion, representing 45% of total loans. The proportion of uninsured mortgages was 62%. The bank's exposure to variable-rate mortgages increased as borrowers faced renewal at higher rates.",
     "bank": "Canadian Imperial Bank of Commerce", "section": "Credit Risk", "filing_type": "Annual Report", "fiscal_year": "2024"},
    {"id": "rbc_credit_q1_001", "text": "Provision for credit losses was $813 million in Q1 2025, compared to $720 million in Q1 2024. The increase was driven by higher provisions in the personal lending and credit card portfolios.",
     "bank": "Royal Bank of Canada", "section": "Credit Risk", "filing_type": "Quarterly Report", "fiscal_year": "2025-Q1"},
    {"id": "rbc_climate_q1_001", "text": "The bank published its second Climate Report aligned with TCFD recommendations. OSFI Guideline B-15 climate risk management implementation is on track for full compliance.",
     "bank": "Royal Bank of Canada", "section": "Climate Risk", "filing_type": "Quarterly Report", "fiscal_year": "2025-Q1"},
]


def setup_chromadb(chunks, collection_name="canadian_bank_filings"):
    """Create and populate a ChromaDB collection."""
    import chromadb
    client = chromadb.Client()

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "bank": c["bank"],
            "section": c["section"],
            "filing_type": c["filing_type"],
            "fiscal_year": c["fiscal_year"],
        } for c in chunks],
    )

    print(f"  Collection '{collection_name}': {collection.count()} chunks")
    return collection


def print_results(results, label):
    """Display search results."""
    print(f"\n  {label}")
    print(f"  {'-' * 50}")
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for i in range(len(docs)):
        meta = metas[i]
        sim = 1 - dists[i] if dists[i] <= 1 else dists[i]
        print(f"    {i+1}. [{sim:.3f}] {meta['bank']} | {meta['section']} | {meta['filing_type']} {meta['fiscal_year']}")
        print(f"       {docs[i][:80]}...")


def main():
    print("=" * 60)
    print("Retrieval Strategies: Canadian Bank Filing Search")
    print("=" * 60)

    try:
        import chromadb
    except ImportError:
        print("  ChromaDB not installed. Install: pip install chromadb")
        return

    collection = setup_chromadb(CHUNKS)

    # Strategy 1: Top-k
    query = "What are the credit risk exposures and provisions?"
    print(f"\n  Query: '{query}'")

    results_topk = collection.query(query_texts=[query], n_results=5)
    print_results(results_topk, "Top-k (k=5)")

    # Strategy 2: Metadata-filtered
    results_rbc = collection.query(query_texts=[query], n_results=5,
                                    where={"bank": "Royal Bank of Canada"})
    print_results(results_rbc, "Top-k FILTERED (RBC only)")

    results_quarterly = collection.query(query_texts=[query], n_results=5,
                                          where={"filing_type": "Quarterly Report"})
    print_results(results_quarterly, "Top-k FILTERED (Quarterly Reports only)")

    # Strategy 3: Different query types
    queries = [
        "climate risk and net-zero commitments",
        "cybersecurity investment and operational risk",
        "mortgage portfolio and housing exposure",
        "OSFI guidelines and regulatory compliance",
    ]

    print(f"\n{'=' * 60}")
    print("MULTI-QUERY RETRIEVAL")
    print(f"{'=' * 60}")

    for q in queries:
        results = collection.query(query_texts=[q], n_results=3)
        print_results(results, f"Query: '{q}'")

    # Key takeaways
    print(f"\n{'=' * 60}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 60}")
    print("1. Top-k returns the most similar chunks regardless of bank.")
    print("2. Metadata filtering restricts search to specific banks,")
    print("   filing types, or time periods - essential for compliance.")
    print("3. ChromaDB uses its own embedding model by default.")
    print("4. For production, specify your embedding model explicitly.")
    print("5. Canadian bank metadata should include: bank name,")
    print("   filing type, fiscal year, quarter, and section.")
    print("6. MMR (not shown in ChromaDB default) balances relevance")
    print("   with diversity - useful for broad risk overview queries.")


if __name__ == "__main__":
    main()
