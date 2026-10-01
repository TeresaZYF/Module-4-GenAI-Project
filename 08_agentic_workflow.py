"""
Demo 08: Agentic Workflow - Cross-Year Canadian Bank Filing Comparison
======================================================================
Instructor demo - ReAct agent comparing risk factor disclosures
across fiscal years for Canadian banks.

Prerequisites:
    pip install openai
    Set OPENAI_API_KEY environment variable

Usage:
    python demos/08_agentic_workflow.py
"""

import os, json, time
from typing import Dict
from datetime import datetime

FILING_DB = {
    "Royal Bank of Canada": {
        "2023": {"risk_factors": [
            "Credit risk from consumer lending is the primary exposure. PCL was $2.9 billion.",
            "Interest rate risk increased as the Bank of Canada maintained elevated policy rates.",
            "Operational risk from legacy systems requires ongoing technology investment.",
            "Geopolitical uncertainty from Russia-Ukraine conflict affects European exposures.",
        ]},
        "2024": {"risk_factors": [
            "Credit risk from consumer lending remains primary. PCL rose to $3.4 billion with rising delinquencies.",
            "Interest rate risk stabilized as Bank of Canada began cutting the overnight rate in June 2024.",
            "Operational risk from legacy systems requires ongoing technology investment.",
            "Geopolitical uncertainty expanded to include Middle East conflict alongside Russia-Ukraine.",
            "NEW: Artificial intelligence risk including model governance, data privacy, and OSFI E-23 readiness.",
            "NEW: Commercial real estate concentration risk as office vacancy rates in Toronto and Vancouver rise.",
        ]},
    },
}

def get_risk_factors(bank, fiscal_year):
    data = FILING_DB.get(bank, {}).get(fiscal_year, {}).get("risk_factors", [])
    if not data: return json.dumps({"status": "not_found"})
    return json.dumps({"status": "success", "bank": bank, "fiscal_year": fiscal_year, "risk_factors": data, "count": len(data)})

def list_available_years(bank):
    years = list(FILING_DB.get(bank, {}).keys())
    return json.dumps({"bank": bank, "available_years": years})

TOOLS = {"get_risk_factors": get_risk_factors, "list_available_years": list_available_years}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_risk_factors", "description": "Retrieve risk factors for a Canadian bank and fiscal year.", "parameters": {"type": "object", "properties": {"bank": {"type": "string"}, "fiscal_year": {"type": "string"}}, "required": ["bank", "fiscal_year"]}}},
    {"type": "function", "function": {"name": "list_available_years", "description": "List available fiscal years for a Canadian bank.", "parameters": {"type": "object", "properties": {"bank": {"type": "string"}}, "required": ["bank"]}}},
]

class ReActAgent:
    def __init__(self, max_iterations=8):
        self.max_iterations = max_iterations
        self.execution_log = []
        self.total_tokens = 0

    def run(self, query):
        try:
            from openai import OpenAI
            self.client = OpenAI()
        except Exception as e:
            return {"status": "error", "message": str(e)}

        messages = [
            {"role": "system", "content": "You are a Canadian bank compliance analyst. Compare risk disclosures across fiscal years. Cite the bank and fiscal year for each finding."},
            {"role": "user", "content": query},
        ]
        self.execution_log = []
        self.total_tokens = 0
        msg = None

        for i in range(self.max_iterations):
            step = {"iteration": i + 1}
            response = self.client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto", temperature=0)
            self.total_tokens += response.usage.total_tokens
            msg = response.choices[0].message
            if msg.tool_calls:
                messages.append(msg)
                step["tools"] = []
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = TOOLS[tc.function.name](**args) if tc.function.name in TOOLS else json.dumps({"error": "unknown"})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    step["tools"].append({"function": tc.function.name, "args": args})
            else:
                step["type"] = "answer"
                self.execution_log.append(step)
                break
            self.execution_log.append(step)

        cost = self.total_tokens * 0.15 / 1_000_000
        return {"status": "success", "answer": msg.content if msg and msg.content else "No answer", "iterations": len(self.execution_log), "tokens": self.total_tokens, "cost": cost, "log": self.execution_log}

def main():
    print("=" * 60)
    print("Agentic Workflow: Canadian Bank Cross-Year Comparison")
    print("=" * 60)
    agent = ReActAgent()
    queries = [
        "What new risk factors did Royal Bank of Canada add in fiscal 2024 that were not in 2023?",
        "How did RBC's description of interest rate risk change between 2023 and 2024?",
    ]
    for q in queries:
        print(f"\n{'=' * 60}\nQuery: {q}\n{'=' * 60}")
        result = agent.run(q)
        if result["status"] == "success":
            print(f"  Iterations: {result['iterations']}, Tokens: {result['tokens']}, Cost: ${result['cost']:.6f}")
            for step in result["log"]:
                if "tools" in step:
                    for t in step["tools"]: print(f"  [{step['iteration']}] TOOL: {t['function']}({json.dumps(t['args'])})")
            print(f"\n  Answer: {result['answer'][:400]}")

    print(f"\n{'=' * 60}\nKEY TAKEAWAYS\n{'=' * 60}")
    print("1. ReAct agents reason about which tools to call step by step.")
    print("2. Cross-year comparison identified NEW risks (AI, CRE)")
    print("   and CHANGED risks (interest rate language evolution).")
    print("3. Track cost per agent run for budget planning.")
    print("4. Log execution traces for OSFI E-23 audit compliance.")

if __name__ == "__main__":
    main()
