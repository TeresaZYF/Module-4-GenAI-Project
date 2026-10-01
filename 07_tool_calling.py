"""
Demo 07: Tool Calling for Canadian Bank Filing AI Agent
========================================================
Instructor demo - demonstrates LLM function calling with
tools for searching Canadian bank filings.

Prerequisites:
    pip install openai
    Set OPENAI_API_KEY environment variable

Usage:
    python demos/07_tool_calling.py
"""

import os, json, time
from typing import List, Dict, Optional

FILING_DATABASE = {
    "Royal Bank of Canada": {
        "Annual Report": {
            "2024": {
                "Credit Risk": [
                    "Provision for credit losses was $3.4 billion in fiscal 2024, up from $2.9 billion in 2023.",
                    "Gross impaired loans increased to $5.8 billion from $4.2 billion, driven by commercial real estate.",
                    "Residential mortgage portfolio: $421 billion (42% of total loans). Uninsured: 58%.",
                ],
                "Climate Risk": [
                    "Committed to net-zero by 2050 in accordance with OSFI Guideline B-15.",
                    "Physical risk concentrated in real estate vulnerable to flooding and wildfire.",
                ],
            },
            "2023": {
                "Credit Risk": [
                    "Provision for credit losses was $2.9 billion in fiscal 2023.",
                    "Gross impaired loans were $4.2 billion. Allowance for credit losses was 0.65% of total loans.",
                ],
            },
        },
    },
    "Toronto-Dominion Bank": {
        "Annual Report": {
            "2024": {
                "Credit Risk": [
                    "Gross impaired loans were $6.2 billion, up from $4.8 billion in 2023.",
                    "US retail portfolio showed elevated delinquencies in credit cards.",
                ],
                "Climate Risk": [
                    "Committed to net-zero by 2050. Interim 2030 targets for energy and power sectors.",
                ],
            },
        },
    },
    "Bank of Montreal": {
        "Annual Report": {
            "2024": {
                "Operational Risk": [
                    "Deployed over 200 AI/ML models in production.",
                    "Established AI Ethics Committee for OSFI E-23 readiness.",
                    "Invested $800 million in cybersecurity capabilities.",
                ],
            },
        },
    },
}

def search_filings(query, bank=None, filing_type=None, section=None):
    results = []
    for bk, bk_data in FILING_DATABASE.items():
        if bank and bank.lower() not in bk.lower(): continue
        for ft, ft_data in bk_data.items():
            if filing_type and filing_type.lower() not in ft.lower(): continue
            for yr, yr_data in ft_data.items():
                for sec, passages in yr_data.items():
                    if section and section.lower() not in sec.lower(): continue
                    for p in passages:
                        if any(w.lower() in p.lower() for w in query.split()):
                            results.append({"bank": bk, "filing_type": ft, "fiscal_year": yr, "section": sec, "text": p})
    return json.dumps({"status": "success" if results else "no_results", "results": results[:5]})

def get_filing_list(bank=None):
    filings = []
    for bk, bk_data in FILING_DATABASE.items():
        if bank and bank.lower() not in bk.lower(): continue
        for ft, ft_data in bk_data.items():
            for yr in ft_data:
                filings.append({"bank": bk, "filing_type": ft, "fiscal_year": yr, "sections": list(ft_data[yr].keys())})
    return json.dumps({"filings": filings})

def compare_sections(bank, section, year1, year2):
    bk_data = FILING_DATABASE.get(bank, {}).get("Annual Report", {})
    d1 = bk_data.get(year1, {}).get(section, [])
    d2 = bk_data.get(year2, {}).get(section, [])
    return json.dumps({"bank": bank, "section": section, "period_1": {"year": year1, "passages": d1}, "period_2": {"year": year2, "passages": d2}, "status": "success" if d1 or d2 else "no_data"})

TOOL_FUNCTIONS = {"search_filings": search_filings, "get_filing_list": get_filing_list, "compare_sections": compare_sections}

TOOLS = [
    {"type": "function", "function": {"name": "search_filings", "description": "Search Canadian bank filings for passages matching a query.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "bank": {"type": "string"}, "filing_type": {"type": "string"}, "section": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_filing_list", "description": "List available Canadian bank filings.", "parameters": {"type": "object", "properties": {"bank": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "compare_sections", "description": "Compare the same section across two fiscal years.", "parameters": {"type": "object", "properties": {"bank": {"type": "string"}, "section": {"type": "string"}, "year1": {"type": "string"}, "year2": {"type": "string"}}, "required": ["bank", "section", "year1", "year2"]}}},
]

def run_agent(user_query, max_iterations=5):
    try:
        from openai import OpenAI
        client = OpenAI()
    except ImportError:
        print("  OpenAI not installed."); return None

    messages = [
        {"role": "system", "content": "You are a Canadian bank compliance analyst with access to filing search tools. Cite the bank, document, and section for every claim."},
        {"role": "user", "content": user_query},
    ]
    total_tokens = 0
    for iteration in range(max_iterations):
        print(f"\n  --- Iteration {iteration + 1} ---")
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOLS, tool_choice="auto", temperature=0)
        total_tokens += response.usage.total_tokens
        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"  Tool: {tc.function.name}({json.dumps(args)})")
                result = TOOL_FUNCTIONS[tc.function.name](**args) if tc.function.name in TOOL_FUNCTIONS else json.dumps({"error": "unknown"})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            print(f"\n  Final Answer: {msg.content[:300]}")
            print(f"  Tokens: {total_tokens}, Iterations: {iteration + 1}")
            return msg.content
    return None

def main():
    print("=" * 60)
    print("Tool Calling: Canadian Bank Filing AI Agent")
    print("=" * 60)
    queries = [
        "What are RBC's key credit risk factors in their 2024 annual report?",
        "How did RBC's credit risk change from 2023 to 2024?",
        "Which Canadian banks discuss AI risk in their filings?",
    ]
    for q in queries:
        print(f"\n{'=' * 60}\nQuery: {q}\n{'=' * 60}")
        run_agent(q)

    print(f"\n{'=' * 60}\nKEY TAKEAWAYS\n{'=' * 60}")
    print("1. Tools let the LLM search Canadian bank filings programmatically.")
    print("2. The LLM decides which tool to call and with what arguments.")
    print("3. compare_sections enables cross-year analysis.")
    print("4. Always log tool calls for OSFI E-23 audit compliance.")

if __name__ == "__main__":
    main()
