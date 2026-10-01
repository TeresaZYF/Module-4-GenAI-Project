"""
Demo 09: LLM-as-Judge Evaluation Pipeline
==========================================
Instructor demo script - demonstrates how to use an LLM to evaluate
the quality of RAG system outputs across multiple dimensions.

This script demonstrates:
- Defining evaluation rubrics for LLM judges
- Faithfulness scoring (is the answer grounded in context?)
- Relevance scoring (does the answer address the question?)
- Completeness scoring (does the answer cover all key points?)
- Running batch evaluations with cost tracking
- Comparing evaluation results across RAG configurations

Prerequisites:
    pip install openai pandas
    Set OPENAI_API_KEY environment variable

Usage:
    python demos/09_llm_judge.py
"""

import os
import json
import time
from typing import List, Dict, Optional
from datetime import datetime

# If pandas is not installed, we handle gracefully
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ──────────────────────────────────────────────────────────
# SAMPLE RAG OUTPUTS TO EVALUATE
# ──────────────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "id": "eval_001",
        "question": "What are Royal Bank of Canada's key credit risk factors in 2023?",
        "retrieved_context": (
            "Credit risk is the risk of financial loss from borrower default. "
            "Total loans were $3.4 billion for fiscal 2024. "
            "The provision for credit losses was $9.8 billion in 2023, "
            "up from $2.9 billion in 2023."
        ),
        "rag_answer": (
            "Royal Bank of Canada's key credit risk factors in 2023 include a total loan "
            "portfolio of $3.4 billion and provision for credit losses of $3.4 "
            "billion, which increased from $2.9 billion in 2023. This increase "
            "in provisioning suggests the bank is preparing for potential "
            "deterioration in credit quality."
        ),
        "reference_answer": (
            "Royal Bank of Canada's credit risk factors include a $3.4 billion loan "
            "portfolio and $9.8 billion in credit loss provisions (up from "
            "$6.4 billion in 2022)."
        ),
    },
    {
        "id": "eval_002",
        "question": "How much did Toronto-Dominion Bank invest in cybersecurity?",
        "retrieved_context": (
            "Cybersecurity: Invested over $800 million in cybersecurity "
            "capabilities in 2023. Climate risk: The company committed to "
            "net-zero emissions in financing activities by 2050."
        ),
        "rag_answer": (
            "Toronto-Dominion Bank invested approximately $3 billion in "
            "cybersecurity during 2023, making it one of the largest "
            "cybersecurity spenders in the financial sector."
        ),
        "reference_answer": (
            "Toronto-Dominion Bank invested over $800 million in cybersecurity "
            "capabilities in 2023."
        ),
    },
    {
        "id": "eval_003",
        "question": "What is Bank of Montreal' position on climate risk?",
        "retrieved_context": (
            "Credit risk: Allowance for credit losses was $14.6 billion "
            "for fiscal 2024. Market risk is managed through VaR models."
        ),
        "rag_answer": (
            "Based on the available filings, I could not find specific "
            "information about Bank of Montreal' position on climate risk. "
            "The retrieved documents discuss credit risk and market risk "
            "but do not address climate-related disclosures for Bank of Montreal."
        ),
        "reference_answer": (
            "No information about Bank of Montreal' climate risk position "
            "is available in the retrieved context."
        ),
    },
]


# ──────────────────────────────────────────────────────────
# EVALUATION RUBRICS
# ──────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are evaluating whether a RAG system's answer is faithful to the retrieved context.

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

RAG ANSWER:
{answer}

Score the answer on FAITHFULNESS (is every claim in the answer supported by the context?):
- 5: Every claim is directly supported by the context. No hallucinated information.
- 4: Almost all claims are supported. Minor inferences are reasonable.
- 3: Most claims are supported, but some unsupported claims are present.
- 2: Significant unsupported claims or mild hallucinations.
- 1: Major hallucinations. The answer contains fabricated information not in the context.

Respond with ONLY a JSON object (no markdown, no backticks):
{{"score": <1-5>, "reasoning": "<brief explanation>", "hallucinated_claims": ["<list any claims not in context>"]}}"""

RELEVANCE_PROMPT = """You are evaluating whether a RAG system's answer is relevant to the question.

QUESTION:
{question}

RAG ANSWER:
{answer}

Score the answer on RELEVANCE (does it address the question?):
- 5: Directly and completely addresses the question.
- 4: Mostly addresses the question with minor tangents.
- 3: Partially addresses the question. Some relevant, some irrelevant content.
- 2: Mostly irrelevant with only tangential connection to the question.
- 1: Completely irrelevant or does not attempt to answer.

Respond with ONLY a JSON object (no markdown, no backticks):
{{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

COMPLETENESS_PROMPT = """You are evaluating whether a RAG system's answer is complete.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference}

RAG ANSWER:
{answer}

Score the answer on COMPLETENESS (does it cover all key points from the reference?):
- 5: Covers all key points from the reference answer.
- 4: Covers most key points. One minor point missing.
- 3: Covers some key points but misses important details.
- 2: Covers very few key points from the reference.
- 1: Misses all key points or provides wrong information.

Respond with ONLY a JSON object (no markdown, no backticks):
{{"score": <1-5>, "reasoning": "<brief explanation>", "missing_points": ["<list any missing key points>"]}}"""


# ──────────────────────────────────────────────────────────
# LLM JUDGE CLASS
# ──────────────────────────────────────────────────────────

class LLMJudge:
    """
    Uses an LLM to evaluate RAG system outputs across multiple
    quality dimensions: faithfulness, relevance, and completeness.
    """

    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.model = model
        self.temperature = temperature
        self.total_tokens = 0
        self.total_cost = 0.0
        self.eval_results = []

    def _call_llm(self, prompt: str) -> Dict:
        """Make a single LLM call and parse JSON response."""
        try:
            from openai import OpenAI
            client = OpenAI()
        except (ImportError, Exception) as e:
            return {"score": 0, "reasoning": f"API not available: {e}"}

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        self.total_tokens += response.usage.total_tokens

        text = response.choices[0].message.content.strip()
        # Clean potential markdown fences
        text = text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"score": 0, "reasoning": f"Failed to parse: {text[:200]}"}

    def evaluate_faithfulness(self, question: str, context: str,
                              answer: str) -> Dict:
        """Score whether the answer is grounded in the retrieved context."""
        prompt = FAITHFULNESS_PROMPT.format(
            context=context, question=question, answer=answer
        )
        result = self._call_llm(prompt)
        result["dimension"] = "faithfulness"
        return result

    def evaluate_relevance(self, question: str, answer: str) -> Dict:
        """Score whether the answer addresses the question."""
        prompt = RELEVANCE_PROMPT.format(
            question=question, answer=answer
        )
        result = self._call_llm(prompt)
        result["dimension"] = "relevance"
        return result

    def evaluate_completeness(self, question: str, answer: str,
                               reference: str) -> Dict:
        """Score whether the answer covers all key points."""
        prompt = COMPLETENESS_PROMPT.format(
            question=question, answer=answer, reference=reference
        )
        result = self._call_llm(prompt)
        result["dimension"] = "completeness"
        return result

    def evaluate_sample(self, sample: Dict) -> Dict:
        """Run all three evaluations on a single sample."""
        print(f"  Evaluating {sample['id']}...")

        faithfulness = self.evaluate_faithfulness(
            sample["question"], sample["retrieved_context"],
            sample["rag_answer"]
        )
        relevance = self.evaluate_relevance(
            sample["question"], sample["rag_answer"]
        )
        completeness = self.evaluate_completeness(
            sample["question"], sample["rag_answer"],
            sample["reference_answer"]
        )

        result = {
            "id": sample["id"],
            "question": sample["question"],
            "faithfulness": faithfulness,
            "relevance": relevance,
            "completeness": completeness,
            "avg_score": round(
                (faithfulness.get("score", 0) +
                 relevance.get("score", 0) +
                 completeness.get("score", 0)) / 3, 2
            ),
        }

        self.eval_results.append(result)
        return result

    def evaluate_batch(self, samples: List[Dict]) -> List[Dict]:
        """Evaluate a batch of samples."""
        results = []
        for sample in samples:
            result = self.evaluate_sample(sample)
            results.append(result)
        return results

    def get_summary(self) -> Dict:
        """Compute aggregate statistics across all evaluations."""
        if not self.eval_results:
            return {"status": "no_evaluations"}

        n = len(self.eval_results)
        avg_faith = sum(
            r["faithfulness"].get("score", 0) for r in self.eval_results
        ) / n
        avg_rel = sum(
            r["relevance"].get("score", 0) for r in self.eval_results
        ) / n
        avg_comp = sum(
            r["completeness"].get("score", 0) for r in self.eval_results
        ) / n
        avg_overall = sum(
            r["avg_score"] for r in self.eval_results
        ) / n

        # Estimate cost (gpt-4o-mini pricing)
        self.total_cost = self.total_tokens * 0.15 / 1_000_000

        return {
            "num_samples": n,
            "avg_faithfulness": round(avg_faith, 2),
            "avg_relevance": round(avg_rel, 2),
            "avg_completeness": round(avg_comp, 2),
            "avg_overall": round(avg_overall, 2),
            "total_tokens": self.total_tokens,
            "estimated_cost": round(self.total_cost, 6),
        }


# ──────────────────────────────────────────────────────────
# DEMO: SIMULATED EVALUATION (no API needed)
# ──────────────────────────────────────────────────────────

def run_simulated_evaluation():
    """
    Demonstrates the evaluation pipeline with pre-computed scores
    so the demo runs without an API key.
    """
    print("\n  SIMULATED EVALUATION (no API key required)")
    print("  " + "-" * 50)

    # Pre-computed evaluation results (what an LLM judge would return)
    simulated_results = [
        {
            "id": "eval_001",
            "question": "What are Royal Bank of Canada's key credit risk factors in 2023?",
            "faithfulness": {
                "score": 4,
                "reasoning": (
                    "Most claims are supported. The inference about "
                    "'preparing for deterioration' is reasonable but "
                    "not stated in the context."
                ),
                "hallucinated_claims": [
                    "The inference about preparing for credit quality "
                    "deterioration is not explicitly in the context."
                ],
                "dimension": "faithfulness",
            },
            "relevance": {
                "score": 5,
                "reasoning": "Directly addresses credit risk factors.",
                "dimension": "relevance",
            },
            "completeness": {
                "score": 5,
                "reasoning": "Covers all key points from reference.",
                "missing_points": [],
                "dimension": "completeness",
            },
            "avg_score": 4.67,
        },
        {
            "id": "eval_002",
            "question": "How much did Toronto-Dominion Bank invest in cybersecurity?",
            "faithfulness": {
                "score": 1,
                "reasoning": (
                    "HALLUCINATION: Answer says $3 billion but context "
                    "says $800 million. This is a critical factual error."
                ),
                "hallucinated_claims": [
                    "$3 billion investment figure (context says $800 million)",
                    "'one of the largest cybersecurity spenders' claim "
                    "is not in the context",
                ],
                "dimension": "faithfulness",
            },
            "relevance": {
                "score": 5,
                "reasoning": "Directly addresses the question topic.",
                "dimension": "relevance",
            },
            "completeness": {
                "score": 1,
                "reasoning": "Provides wrong dollar amount.",
                "missing_points": ["Correct $800 million figure"],
                "dimension": "completeness",
            },
            "avg_score": 2.33,
        },
        {
            "id": "eval_003",
            "question": "What is Bank of Montreal' position on climate risk?",
            "faithfulness": {
                "score": 5,
                "reasoning": (
                    "Answer correctly states the information is not "
                    "available. No hallucination."
                ),
                "hallucinated_claims": [],
                "dimension": "faithfulness",
            },
            "relevance": {
                "score": 5,
                "reasoning": (
                    "Appropriately addresses the question by noting "
                    "the absence of relevant information."
                ),
                "dimension": "relevance",
            },
            "completeness": {
                "score": 5,
                "reasoning": (
                    "Matches reference: correctly identifies that no "
                    "Bank of Montreal climate data is available."
                ),
                "missing_points": [],
                "dimension": "completeness",
            },
            "avg_score": 5.0,
        },
    ]

    # Display results
    for result in simulated_results:
        print(f"\n  --- {result['id']} ---")
        print(f"  Q: {result['question']}")
        print(f"  Faithfulness: {result['faithfulness']['score']}/5"
              f" - {result['faithfulness']['reasoning'][:80]}")
        print(f"  Relevance:    {result['relevance']['score']}/5"
              f" - {result['relevance']['reasoning'][:80]}")
        print(f"  Completeness: {result['completeness']['score']}/5"
              f" - {result['completeness']['reasoning'][:80]}")
        print(f"  Average:      {result['avg_score']}/5")

        if result["faithfulness"].get("hallucinated_claims"):
            print(f"  HALLUCINATIONS DETECTED:")
            for claim in result["faithfulness"]["hallucinated_claims"]:
                print(f"    - {claim}")

    # Summary
    n = len(simulated_results)
    avg_faith = sum(r["faithfulness"]["score"] for r in simulated_results) / n
    avg_rel = sum(r["relevance"]["score"] for r in simulated_results) / n
    avg_comp = sum(r["completeness"]["score"] for r in simulated_results) / n

    print(f"\n  {'=' * 50}")
    print(f"  AGGREGATE SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Samples evaluated:    {n}")
    print(f"  Avg Faithfulness:     {avg_faith:.2f}/5")
    print(f"  Avg Relevance:        {avg_rel:.2f}/5")
    print(f"  Avg Completeness:     {avg_comp:.2f}/5")
    print(f"  Avg Overall:          {(avg_faith + avg_rel + avg_comp)/3:.2f}/5")

    # Flag critical issues
    low_faith = [r for r in simulated_results
                 if r["faithfulness"]["score"] <= 2]
    if low_faith:
        print(f"\n  CRITICAL: {len(low_faith)} sample(s) with "
              f"faithfulness <= 2 (hallucination risk)")
        for r in low_faith:
            print(f"    - {r['id']}: {r['question'][:60]}")

    return simulated_results


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LLM-as-Judge Evaluation Pipeline")
    print("=" * 60)

    # Part 1: Show the evaluation dataset
    print("\n  EVALUATION DATASET")
    print("  " + "-" * 50)
    for sample in EVAL_DATASET:
        print(f"\n  {sample['id']}:")
        print(f"    Q: {sample['question']}")
        print(f"    Context: {sample['retrieved_context'][:80]}...")
        print(f"    RAG Answer: {sample['rag_answer'][:80]}...")

    # Part 2: Run simulated evaluation
    simulated_results = run_simulated_evaluation()

    # Part 3: Try live evaluation if API is available
    print(f"\n{'=' * 60}")
    print("LIVE EVALUATION (requires OPENAI_API_KEY)")
    print(f"{'=' * 60}")

    if os.environ.get("OPENAI_API_KEY"):
        judge = LLMJudge(model="gpt-4o-mini")
        live_results = judge.evaluate_batch(EVAL_DATASET)

        for result in live_results:
            print(f"\n  {result['id']}:")
            print(f"    Faithfulness: {result['faithfulness'].get('score', 'N/A')}/5")
            print(f"    Relevance:    {result['relevance'].get('score', 'N/A')}/5")
            print(f"    Completeness: {result['completeness'].get('score', 'N/A')}/5")
            print(f"    Average:      {result['avg_score']}/5")

        summary = judge.get_summary()
        print(f"\n  Summary: {json.dumps(summary, indent=2)}")
    else:
        print("  Skipped (no OPENAI_API_KEY). See simulated results above.")

    # Part 4: Key takeaways
    print(f"\n{'=' * 60}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 60}")
    print("1. LLM-as-Judge automates evaluation at scale.")
    print("2. Three key dimensions: faithfulness, relevance, completeness.")
    print("3. Faithfulness is the MOST important - catches hallucinations.")
    print("4. eval_002 shows why faithfulness matters: the RAG system")
    print("   hallucinated $2B instead of the correct $600M.")
    print("5. eval_003 shows good behavior: the system correctly said")
    print("   'I don't know' rather than fabricating an answer.")
    print("6. Use structured rubrics (1-5 scale) for consistent scoring.")
    print("7. Always log evaluation costs - they add up at scale.")
    print("8. In production, evaluate a random sample of queries daily.")


if __name__ == "__main__":
    main()
