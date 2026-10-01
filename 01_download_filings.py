"""
Demo 01: Download Canadian Bank Filings from SEDAR+ and Investor Relations
==========================================================================
Instructor demo script - downloads annual reports, AIFs, and quarterly
reports from Canada's Big Five banks.

This script demonstrates:
- Querying SEDAR+ for Canadian public company filings
- Downloading filings from bank investor relations websites
- Parsing PDF/HTML filings into clean text
- Building a filing index with metadata

Canadian Data Sources:
- SEDAR+ (www.sedarplus.ca) - Canada's electronic filing system
- Bank investor relations pages (direct PDF downloads)

Prerequisites:
    pip install requests beautifulsoup4 pandas

Usage:
    python demos/01_download_filings.py
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Optional
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"  Missing dependency: {e}")
    print("  Install: pip install requests beautifulsoup4 pandas")

# ──────────────────────────────────────────────────────────
# CANADIAN BIG FIVE BANKS
# ──────────────────────────────────────────────────────────

# SEDAR+ profile IDs and investor relations URLs
CANADIAN_BANKS = {
    "Royal Bank of Canada": {
        "ticker": "RY",
        "sedar_issuer_id": "34536",
        "ir_url": "https://www.rbc.com/investor-relations/",
        "fiscal_year_end": "October 31",
    },
    "Toronto-Dominion Bank": {
        "ticker": "TD",
        "sedar_issuer_id": "12485",
        "ir_url": "https://www.td.com/ca/en/about-td/investor-relations",
        "fiscal_year_end": "October 31",
    },
    "Bank of Montreal": {
        "ticker": "BMO",
        "sedar_issuer_id": "12073",
        "ir_url": "https://www.bmo.com/main/about-bmo/investor-relations/",
        "fiscal_year_end": "October 31",
    },
    "Bank of Nova Scotia": {
        "ticker": "BNS",
        "sedar_issuer_id": "12014",
        "ir_url": "https://www.scotiabank.com/ca/en/about/investors-shareholders.html",
        "fiscal_year_end": "October 31",
    },
    "Canadian Imperial Bank of Commerce": {
        "ticker": "CM",
        "sedar_issuer_id": "12390",
        "ir_url": "https://www.cibc.com/en/about-cibc/investor-relations.html",
        "fiscal_year_end": "October 31",
    },
}

# Filing types to search for
FILING_TYPES = ["Annual Report", "Annual Information Form", "Quarterly Report"]

# SEDAR+ base URL
SEDAR_BASE = "https://www.sedarplus.ca"

HEADERS = {
    "User-Agent": "Jarvis MLE Training program@jarvis.com",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
}


# ──────────────────────────────────────────────────────────
# SEDAR+ SEARCH (demonstrates the approach)
# ──────────────────────────────────────────────────────────

def search_sedar(bank_name: str, issuer_id: str,
                 filing_type: str = "Annual Report") -> List[Dict]:
    """
    Search SEDAR+ for filings by a Canadian bank.

    Note: SEDAR+ has a web interface but limited public API access.
    In production, you would use the SEDAR+ search interface or
    the bank's investor relations page for direct PDF downloads.

    This function demonstrates the search approach conceptually.
    """
    print(f"  Searching SEDAR+ for {bank_name} ({filing_type})...")

    # SEDAR+ search URL pattern
    search_url = f"{SEDAR_BASE}/csa-party/records/search"

    # In practice, SEDAR+ requires browser-based interaction or
    # specific API calls. Here we demonstrate the expected output.
    # For the capstone, students can download PDFs directly from
    # bank investor relations websites.

    # Simulated search results based on actual filing dates
    simulated_results = []

    if "Annual Report" in filing_type:
        simulated_results = [
            {
                "bank": bank_name,
                "type": "Annual Report",
                "fiscal_year": "2024",
                "filing_date": "2024-12-05",
                "description": f"{bank_name} Annual Report for fiscal year ended October 31, 2024",
                "source": "SEDAR+",
                "estimated_pages": 300,
            },
            {
                "bank": bank_name,
                "type": "Annual Report",
                "fiscal_year": "2023",
                "filing_date": "2023-12-07",
                "description": f"{bank_name} Annual Report for fiscal year ended October 31, 2023",
                "source": "SEDAR+",
                "estimated_pages": 290,
            },
        ]
    elif "Annual Information Form" in filing_type:
        simulated_results = [
            {
                "bank": bank_name,
                "type": "AIF",
                "fiscal_year": "2024",
                "filing_date": "2024-12-05",
                "description": f"{bank_name} Annual Information Form for fiscal year ended October 31, 2024",
                "source": "SEDAR+",
                "estimated_pages": 80,
            },
        ]

    return simulated_results


# ──────────────────────────────────────────────────────────
# PDF / HTML PARSING
# ──────────────────────────────────────────────────────────

def parse_filing_html(html_content: str) -> str:
    """
    Parse a Canadian bank filing HTML document into clean text.
    Extracts text content, preserves section headings, removes
    navigation and boilerplate.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script, style, and nav elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)

    return text


def chunk_text(text: str, chunk_size: int = 800,
               overlap: int = 150) -> List[Dict]:
    """
    Split text into overlapping chunks for embedding.
    Tries to split at paragraph boundaries.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    chunk_id = 0

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "char_count": len(current_chunk.strip()),
            })
            chunk_id += 1
            # Keep overlap from end of previous chunk
            words = current_chunk.split()
            overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else []
            current_chunk = " ".join(overlap_words) + "\n\n" + para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append({
            "chunk_id": chunk_id,
            "text": current_chunk.strip(),
            "char_count": len(current_chunk.strip()),
        })

    return chunks


# ──────────────────────────────────────────────────────────
# SAMPLE CANADIAN BANK FILING TEXT
# ──────────────────────────────────────────────────────────

SAMPLE_FILING_TEXT = """Management's Discussion and Analysis

Credit Risk

Credit risk is the risk of loss associated with a counterparty's inability to fulfill its payment obligations. Credit risk may arise directly from the risk of default of a primary obligor or indirectly from a secondary obligor.

For the fiscal year ended October 31, 2024, the provision for credit losses was $3.4 billion, an increase from $2.9 billion in the prior year. The increase reflects a normalization of credit conditions following the pandemic recovery period and higher provisions in the commercial real estate portfolio. Gross impaired loans increased to $5.8 billion from $4.2 billion, primarily driven by office and retail commercial real estate exposures in the United States and Canada.

The allowance for credit losses as a percentage of total loans was 0.72%, compared to 0.65% in the prior year. The bank continues to monitor emerging risks including the impact of elevated interest rates on variable-rate mortgage holders and the potential for a slowdown in housing activity.

Market Risk

Market risk is the risk of loss arising from adverse changes in underlying market factors, including interest rates, foreign exchange rates, equity prices, commodity prices, and credit spreads. The bank uses Value-at-Risk (VaR) to measure and control trading market risk. Average daily VaR for fiscal 2024 was $42 million, compared to $38 million in the prior year.

Interest rate risk in the banking book is managed through asset-liability management. A 100 basis point parallel increase in interest rates would increase net interest income by approximately $320 million over the next 12 months. The bank's structural interest rate risk position reflects the maturity mismatch between assets (primarily 5-year fixed-rate mortgages) and liabilities (primarily demand and short-term deposits).

Climate Risk

In accordance with OSFI Guideline B-15, the bank has enhanced its climate risk management framework. The bank has committed to achieving net-zero greenhouse gas emissions in its lending and investment portfolios by 2050, with interim targets for 2030.

Physical risk exposure is concentrated in real estate portfolios in regions vulnerable to flooding, wildfire, and extreme weather events. The bank has integrated climate risk scores into its commercial real estate underwriting process and is developing scenario analysis capabilities aligned with the Network for Greening the Financial System (NGFS) scenarios.

Transition risk exposure is most significant in the oil and gas, utilities, and transportation sectors. The bank's total exposure to carbon-intensive sectors represents approximately 8% of total commercial lending. The bank has established sector-specific transition plans and engaged with large borrowers on their decarbonization strategies.

Operational Risk

Operational risk is the risk of loss resulting from inadequate or failed internal processes, people, and systems, or from external events. Cybersecurity risk remains a top priority, with the bank investing over $800 million in cybersecurity and technology resilience capabilities during fiscal 2024. The bank experienced no material cybersecurity incidents during the reporting period.

Third-party risk management continues to evolve in response to OSFI Guideline B-10 (Outsourcing of Business Activities and Functions). The bank's critical third-party vendor inventory includes 47 material outsourcing arrangements, each subject to enhanced due diligence, contractual protections, and ongoing monitoring.

Artificial intelligence risk is an emerging operational risk category. The bank has deployed over 200 AI/ML models in production, subject to the bank's AI governance framework. In anticipation of OSFI Guideline E-23 (Model Risk Management), the bank has established an AI Ethics Committee and enhanced its model validation procedures for AI/ML models.

Liquidity Risk

Liquidity risk is the risk that the bank cannot meet its financial obligations as they come due without resorting to costly or untimely measures. The bank maintains a Liquidity Coverage Ratio (LCR) well above the regulatory minimum of 100%. As of October 31, 2024, the LCR was 132%, compared to 128% in the prior year. The Net Stable Funding Ratio (NSFR) was 114%, also above the 100% minimum.

The bank's total high-quality liquid assets (HQLA) were $298 billion at October 31, 2024, compared to $275 billion at the prior year-end. The increase reflects a deliberate build-up of liquidity buffers in response to market uncertainty and the potential for deposit outflow in a higher interest rate environment.
"""


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Canadian Bank Filing Downloader")
    print("=" * 60)

    # Part 1: Show available banks
    print("\n  CANADIAN BIG FIVE BANKS")
    print("  " + "-" * 50)
    for bank, info in CANADIAN_BANKS.items():
        print(f"  {info['ticker']:>4} | {bank}")
        print(f"       | Fiscal year end: {info['fiscal_year_end']}")
        print(f"       | IR: {info['ir_url']}")

    # Part 2: Search SEDAR+ (simulated)
    print(f"\n{'=' * 60}")
    print("SEDAR+ FILING SEARCH (simulated)")
    print(f"{'=' * 60}")

    all_filings = []
    for bank_name, info in CANADIAN_BANKS.items():
        results = search_sedar(bank_name, info["sedar_issuer_id"],
                               "Annual Report")
        all_filings.extend(results)

    print(f"\n  Found {len(all_filings)} filings across {len(CANADIAN_BANKS)} banks")

    if HAS_DEPS:
        df = pd.DataFrame(all_filings)
        print(f"\n  Filing Index:")
        print(df[["bank", "type", "fiscal_year", "estimated_pages"]].to_string(index=False))

    # Part 3: Demonstrate text parsing and chunking
    print(f"\n{'=' * 60}")
    print("TEXT PARSING & CHUNKING DEMO")
    print(f"{'=' * 60}")

    chunks = chunk_text(SAMPLE_FILING_TEXT, chunk_size=800, overlap=100)
    print(f"\n  Input text: {len(SAMPLE_FILING_TEXT)} characters")
    print(f"  Chunks created: {len(chunks)}")

    for chunk in chunks:
        preview = chunk["text"][:80].replace("\n", " ")
        print(f"  Chunk {chunk['chunk_id']}: {chunk['char_count']} chars | {preview}...")

    # Part 4: Demonstrate metadata tagging
    print(f"\n{'=' * 60}")
    print("METADATA TAGGING")
    print(f"{'=' * 60}")

    tagged_chunks = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        # Auto-detect section from content
        section = "Unknown"
        if "credit risk" in text_lower[:100]:
            section = "Credit Risk"
        elif "market risk" in text_lower[:100]:
            section = "Market Risk"
        elif "climate risk" in text_lower[:100]:
            section = "Climate Risk"
        elif "operational risk" in text_lower[:100]:
            section = "Operational Risk"
        elif "liquidity risk" in text_lower[:100]:
            section = "Liquidity Risk"
        elif "management" in text_lower[:100]:
            section = "MD&A Overview"

        tagged = {
            **chunk,
            "bank": "Royal Bank of Canada",
            "document_type": "Annual Report",
            "fiscal_year": "2024",
            "section": section,
        }
        tagged_chunks.append(tagged)
        print(f"  Chunk {chunk['chunk_id']}: {section} | {chunk['char_count']} chars")

    # Part 5: Key takeaways
    print(f"\n{'=' * 60}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 60}")
    print("1. Canadian bank filings are on SEDAR+ (www.sedarplus.ca)")
    print("   and bank investor relations websites.")
    print("2. Canadian banks have October 31 fiscal year-ends,")
    print("   unlike US banks which use December 31.")
    print("3. Key documents: Annual Report, AIF, Quarterly Reports.")
    print("4. Chunk at paragraph boundaries, not fixed character counts.")
    print("5. Always tag chunks with metadata: bank, document type,")
    print("   fiscal year, section name.")
    print("6. Canadian filings may include French sections -")
    print("   filter for English during preprocessing.")
    print("7. The sample text above demonstrates the structure of")
    print("   MD&A risk disclosures in a Canadian bank annual report.")

    # Save sample data for other demos
    os.makedirs("data", exist_ok=True)
    with open("data/sample_chunks.json", "w") as f:
        json.dump(tagged_chunks, f, indent=2)
    print(f"\n  Saved {len(tagged_chunks)} tagged chunks to data/sample_chunks.json")


if __name__ == "__main__":
    main()
