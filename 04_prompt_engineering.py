"""
Demo 04: Prompt Engineering for Canadian Bank Filing Q&A
========================================================
Instructor demo script - demonstrates zero-shot, few-shot,
chain-of-thought prompting for Canadian bank filing analysis.

Prerequisites:
    pip install openai
    Set OPENAI_API_KEY environment variable

Usage:
    python demos/04_prompt_engineering.py
"""

import os
import json
import time
from typing import List, Dict

SAMPLE_CONTEXT = [
    {"text": "Credit risk is the risk of loss associated with a counterparty's inability to fulfill its payment obligations. The provision for credit losses was $3.4 billion for fiscal 2024, an increase from $2.9 billion in the prior year. Gross impaired loans increased to $5.8 billion from $4.2 billion.",
     "source": "Royal Bank of Canada Annual Report 2024, MD&A - Credit Risk"},
    {"text": "The residential mortgage portfolio totalled $421 billion, representing 42% of total loans. The proportion of uninsured mortgages was 58%. The bank monitors the impact of elevated interest rates on variable-rate mortgage holders.",
     "source": "Royal Bank of Canada Annual Report 2024, MD&A - Credit Risk"},
    {"text": "In accordance with OSFI Guideline B-15, the bank has enhanced its climate risk management framework. Physical risk exposure is concentrated in real estate portfolios vulnerable to flooding, wildfire, and extreme weather.",
     "source": "Royal Bank of Canada Annual Report 2024, MD&A - Climate Risk"},
]

SAMPLE_QUERY = "What are RBC's key credit risk factors and how large is the mortgage portfolio?"

# ── PROMPT VERSIONS ──

SYSTEM_PROMPT_V1 = """You are a financial compliance analyst specializing in Canadian bank filing analysis.
Answer questions based ONLY on the provided filing excerpts.
If the context does not contain the answer, say so."""

SYSTEM_PROMPT_V2 = """You are a senior financial compliance analyst at a major Canadian bank.

INSTRUCTIONS:
1. Answer ONLY based on the provided Canadian bank filing excerpts.
2. Cite the specific bank, document, and section for every factual claim.
3. If the context does not contain the information, explicitly state: "The provided documents do not contain information about [topic]."
4. Use precise Canadian financial terminology (PCL, OSFI, LCR, AIF).
5. When citing dollar amounts, always include the fiscal year and period.
6. Structure your response with clear sections when the question covers multiple topics."""

FEW_SHOT_EXAMPLES = [
    {
        "question": "What was TD's provision for credit losses?",
        "context": "Toronto-Dominion Bank reported a provision for credit losses of $3.6 billion for fiscal 2024, compared to $2.8 billion in the prior year.",
        "answer": "According to TD's fiscal 2024 annual report, the provision for credit losses was $3.6 billion, an increase from $2.8 billion in fiscal 2023 (Source: Toronto-Dominion Bank Annual Report 2024, MD&A - Credit Risk)."
    },
]

COT_INSTRUCTION = """Think step by step:
1. Identify which parts of the context are relevant to the question
2. Extract specific numbers, metrics, and facts
3. Note the source document and section for each fact
4. Synthesize into a clear, cited answer
5. Flag any information gaps"""

def build_context_string(chunks):
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n".join(parts)

def call_llm(system_prompt, user_message, model="gpt-4o-mini", temperature=0):
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model, temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content, response.usage.total_tokens
    except Exception as e:
        return f"API not available: {e}", 0

def main():
    print("=" * 60)
    print("Prompt Engineering: Canadian Bank Filing Q&A")
    print("=" * 60)

    context_str = build_context_string(SAMPLE_CONTEXT)

    # Version 1: Basic
    print("\n  VERSION 1: Basic Prompt")
    print("  " + "-" * 50)
    user_msg = f"Context:\n{context_str}\n\nQuestion: {SAMPLE_QUERY}"
    answer, tokens = call_llm(SYSTEM_PROMPT_V1, user_msg)
    print(f"  Answer: {answer[:300]}...")
    print(f"  Tokens: {tokens}")

    # Version 2: Structured with citations
    print("\n  VERSION 2: Structured with Citations")
    print("  " + "-" * 50)
    answer2, tokens2 = call_llm(SYSTEM_PROMPT_V2, user_msg)
    print(f"  Answer: {answer2[:300]}...")
    print(f"  Tokens: {tokens2}")

    # Version 3: Chain-of-thought
    print("\n  VERSION 3: Chain-of-Thought")
    print("  " + "-" * 50)
    cot_msg = f"Context:\n{context_str}\n\n{COT_INSTRUCTION}\n\nQuestion: {SAMPLE_QUERY}"
    answer3, tokens3 = call_llm(SYSTEM_PROMPT_V2, cot_msg)
    print(f"  Answer: {answer3[:400]}...")
    print(f"  Tokens: {tokens3}")

    # Version 4: Structured JSON output
    print("\n  VERSION 4: Structured JSON Output")
    print("  " + "-" * 50)
    json_prompt = SYSTEM_PROMPT_V2 + "\n\nRespond in JSON format with keys: answer, sources (list of {bank, document, section, year}), confidence (high/medium/low), caveats (list)."
    answer4, tokens4 = call_llm(json_prompt, user_msg)
    print(f"  Answer: {answer4[:400]}...")
    print(f"  Tokens: {tokens4}")

    print(f"\n{'=' * 60}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 60}")
    print("1. The system prompt controls citation quality and format.")
    print("2. Few-shot examples teach the model your expected output style.")
    print("3. Chain-of-thought produces more thorough analysis but uses more tokens.")
    print("4. JSON output enables downstream parsing and UI integration.")
    print("5. Canadian-specific terminology (OSFI, PCL, AIF) should be")
    print("   included in the system prompt to set domain context.")
    print("6. Always include an 'I don't know' instruction to prevent hallucination.")

if __name__ == "__main__":
    main()
