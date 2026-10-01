"""
Demo 10: MLflow for GenAI Experiment Tracking
===============================================
Instructor demo script - demonstrates how to use MLflow to track
GenAI experiments including prompts, model configurations, evaluation
metrics, and cost.

This script demonstrates:
- Logging GenAI experiments with MLflow
- Tracking prompt versions and configurations
- Recording evaluation metrics (faithfulness, relevance, cost)
- Comparing RAG configurations side-by-side
- Creating a model registry entry for the best configuration

Prerequisites:
    pip install mlflow openai
    (MLflow UI: mlflow ui --port 5000)

Usage:
    python demos/10_mlflow_genai.py
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

# Try importing MLflow - demo works in simulation mode without it
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("  MLflow not installed. Running in simulation mode.")
    print("  Install with: pip install mlflow")


# ──────────────────────────────────────────────────────────
# SIMULATED RAG CONFIGURATIONS TO COMPARE
# ──────────────────────────────────────────────────────────

RAG_CONFIGS = [
    {
        "name": "baseline_rag",
        "description": "Basic RAG with top-k retrieval",
        "params": {
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "chunk_size": 500,
            "chunk_overlap": 50,
            "retrieval_strategy": "top_k",
            "top_k": 3,
            "llm_model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 500,
            "system_prompt_version": "v1_basic",
        },
        "metrics": {
            "faithfulness": 3.2,
            "relevance": 4.1,
            "completeness": 3.5,
            "avg_latency_ms": 1200,
            "avg_cost_per_query": 0.003,
            "total_eval_samples": 50,
        },
    },
    {
        "name": "improved_chunking",
        "description": "Larger chunks with more overlap for better context",
        "params": {
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_strategy": "top_k",
            "top_k": 5,
            "llm_model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 500,
            "system_prompt_version": "v1_basic",
        },
        "metrics": {
            "faithfulness": 3.8,
            "relevance": 4.3,
            "completeness": 4.1,
            "avg_latency_ms": 1450,
            "avg_cost_per_query": 0.004,
            "total_eval_samples": 50,
        },
    },
    {
        "name": "mmr_retrieval",
        "description": "MMR retrieval for diverse context + improved prompt",
        "params": {
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_strategy": "mmr",
            "top_k": 5,
            "mmr_lambda": 0.7,
            "llm_model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 500,
            "system_prompt_version": "v2_structured",
        },
        "metrics": {
            "faithfulness": 4.5,
            "relevance": 4.6,
            "completeness": 4.4,
            "avg_latency_ms": 1600,
            "avg_cost_per_query": 0.005,
            "total_eval_samples": 50,
        },
    },
    {
        "name": "large_embedding_model",
        "description": "Larger embedding model + MMR + structured prompt",
        "params": {
            "embedding_model": "text-embedding-3-large",
            "embedding_dim": 3072,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "retrieval_strategy": "mmr",
            "top_k": 5,
            "mmr_lambda": 0.7,
            "llm_model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 500,
            "system_prompt_version": "v2_structured",
        },
        "metrics": {
            "faithfulness": 4.6,
            "relevance": 4.7,
            "completeness": 4.5,
            "avg_latency_ms": 1800,
            "avg_cost_per_query": 0.008,
            "total_eval_samples": 50,
        },
    },
]

# Sample prompt versions for artifact logging
PROMPT_VERSIONS = {
    "v1_basic": (
        "You are a financial analyst. Answer the question based on "
        "the provided context. If the context does not contain the "
        "answer, say you don't know."
    ),
    "v2_structured": (
        "You are a senior financial compliance analyst at a major bank.\n\n"
        "INSTRUCTIONS:\n"
        "1. Answer ONLY based on the provided Canadian bank filing excerpts.\n"
        "2. Cite the specific filing and section for every claim.\n"
        "3. If the context does not contain the answer, explicitly state "
        "that the information is not available in the retrieved documents.\n"
        "4. Use precise financial terminology.\n"
        "5. Structure your response with clear headings when appropriate."
    ),
}


# ──────────────────────────────────────────────────────────
# MLflow TRACKING (live mode)
# ──────────────────────────────────────────────────────────

def log_experiment_to_mlflow(config: Dict):
    """Log a single RAG configuration experiment to MLflow."""
    if not HAS_MLFLOW:
        return None

    experiment_name = "canadian_bank_rag_evaluation"

    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "module4_capstone", "domain": "financial_compliance"},
        )
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=config["name"],
        description=config["description"],
    ) as run:
        # Log parameters
        for key, value in config["params"].items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in config["metrics"].items():
            mlflow.log_metric(key, value)

        # Log computed metrics
        metrics = config["metrics"]
        overall_quality = (
            metrics["faithfulness"] +
            metrics["relevance"] +
            metrics["completeness"]
        ) / 3
        mlflow.log_metric("overall_quality", round(overall_quality, 2))

        cost_per_quality_point = (
            metrics["avg_cost_per_query"] / overall_quality
            if overall_quality > 0 else 999
        )
        mlflow.log_metric("cost_per_quality_point",
                         round(cost_per_quality_point, 6))

        # Log the system prompt as an artifact
        prompt_version = config["params"].get("system_prompt_version", "v1_basic")
        prompt_text = PROMPT_VERSIONS.get(prompt_version, "Unknown")
        prompt_path = f"/tmp/prompt_{prompt_version}.txt"
        with open(prompt_path, "w") as f:
            f.write(prompt_text)
        mlflow.log_artifact(prompt_path, "prompts")

        # Log config as JSON artifact
        config_path = f"/tmp/config_{config['name']}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path, "configs")

        # Tags
        mlflow.set_tag("rag_strategy", config["params"]["retrieval_strategy"])
        mlflow.set_tag("prompt_version", prompt_version)

        print(f"  Logged run: {config['name']} (run_id: {run.info.run_id})")
        return run.info.run_id


def register_best_model(configs: List[Dict]):
    """Register the best RAG configuration in the MLflow model registry."""
    if not HAS_MLFLOW:
        return

    # Find best config by overall quality
    best = max(configs, key=lambda c: (
        c["metrics"]["faithfulness"] +
        c["metrics"]["relevance"] +
        c["metrics"]["completeness"]
    ) / 3)

    print(f"\n  Best configuration: {best['name']}")
    print(f"  Registering in MLflow Model Registry...")

    # In production, you would register the actual model artifacts
    # Here we demonstrate the pattern
    print(f"  (In production: mlflow.pyfunc.log_model(...))")
    print(f"  (Then: mlflow.register_model(model_uri, 'canadian_bank_rag'))")


# ──────────────────────────────────────────────────────────
# SIMULATION MODE (no MLflow needed)
# ──────────────────────────────────────────────────────────

def run_simulated_tracking():
    """Demonstrate the tracking workflow without MLflow installed."""
    print("\n  SIMULATED MLflow TRACKING")
    print("  " + "-" * 50)
    print("  (Install MLflow to see real tracking UI)")

    for config in RAG_CONFIGS:
        metrics = config["metrics"]
        overall = (
            metrics["faithfulness"] +
            metrics["relevance"] +
            metrics["completeness"]
        ) / 3

        print(f"\n  Run: {config['name']}")
        print(f"    Description: {config['description']}")
        print(f"    Parameters:")
        print(f"      Embedding:  {config['params']['embedding_model']}")
        print(f"      Chunk size: {config['params']['chunk_size']}")
        print(f"      Retrieval:  {config['params']['retrieval_strategy']}")
        print(f"      Prompt:     {config['params']['system_prompt_version']}")
        print(f"    Metrics:")
        print(f"      Faithfulness:  {metrics['faithfulness']:.1f}/5")
        print(f"      Relevance:     {metrics['relevance']:.1f}/5")
        print(f"      Completeness:  {metrics['completeness']:.1f}/5")
        print(f"      Overall:       {overall:.2f}/5")
        print(f"      Latency:       {metrics['avg_latency_ms']}ms")
        print(f"      Cost/query:    ${metrics['avg_cost_per_query']:.4f}")

    # Comparison table
    print(f"\n  {'=' * 70}")
    print(f"  EXPERIMENT COMPARISON TABLE")
    print(f"  {'=' * 70}")
    header = (
        f"  {'Config':<25} {'Faith':>6} {'Relev':>6} "
        f"{'Compl':>6} {'Overall':>8} {'Cost':>8}"
    )
    print(header)
    print(f"  {'-' * 65}")

    best_overall = 0
    best_name = ""

    for config in RAG_CONFIGS:
        m = config["metrics"]
        overall = (m["faithfulness"] + m["relevance"] + m["completeness"]) / 3
        row = (
            f"  {config['name']:<25} {m['faithfulness']:>6.1f} "
            f"{m['relevance']:>6.1f} {m['completeness']:>6.1f} "
            f"{overall:>8.2f} ${m['avg_cost_per_query']:>7.4f}"
        )
        print(row)

        if overall > best_overall:
            best_overall = overall
            best_name = config["name"]

    print(f"  {'-' * 65}")
    print(f"  Best: {best_name} (overall: {best_overall:.2f}/5)")

    # Cost vs quality analysis
    print(f"\n  COST-QUALITY ANALYSIS")
    print(f"  " + "-" * 50)
    baseline = RAG_CONFIGS[0]
    baseline_overall = (
        baseline["metrics"]["faithfulness"] +
        baseline["metrics"]["relevance"] +
        baseline["metrics"]["completeness"]
    ) / 3

    for config in RAG_CONFIGS[1:]:
        m = config["metrics"]
        overall = (m["faithfulness"] + m["relevance"] + m["completeness"]) / 3
        quality_gain = overall - baseline_overall
        cost_increase = (
            m["avg_cost_per_query"] -
            baseline["metrics"]["avg_cost_per_query"]
        )

        if cost_increase > 0:
            cost_per_point = cost_increase / quality_gain if quality_gain > 0 else float("inf")
            print(
                f"  {config['name']}: +{quality_gain:.2f} quality, "
                f"+${cost_increase:.4f}/query "
                f"(${cost_per_point:.4f} per quality point)"
            )


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MLflow for GenAI Experiment Tracking")
    print("=" * 60)

    # Part 1: Run simulated tracking (always works)
    run_simulated_tracking()

    # Part 2: Try live MLflow tracking
    if HAS_MLFLOW:
        print(f"\n{'=' * 60}")
        print("LIVE MLflow TRACKING")
        print(f"{'=' * 60}")

        for config in RAG_CONFIGS:
            log_experiment_to_mlflow(config)

        register_best_model(RAG_CONFIGS)

        print(f"\n  View results: mlflow ui --port 5000")
        print(f"  Then open: http://localhost:5000")
    else:
        print(f"\n  MLflow not installed. Install with:")
        print(f"  pip install mlflow")
        print(f"  Then re-run this script for live tracking.")

    # Part 3: Key takeaways
    print(f"\n{'=' * 60}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 60}")
    print("1. Track EVERY RAG configuration as an MLflow experiment.")
    print("2. Log parameters: embedding model, chunk size, retrieval")
    print("   strategy, prompt version, LLM model, temperature.")
    print("3. Log metrics: faithfulness, relevance, completeness,")
    print("   latency, cost per query.")
    print("4. Log artifacts: system prompts, evaluation datasets,")
    print("   configuration files.")
    print("5. The comparison table shows that MMR retrieval + structured")
    print("   prompt gave the best quality improvement per dollar.")
    print("6. The large embedding model added only +0.1 quality but")
    print("   doubled the cost - likely not worth it.")
    print("7. Use MLflow Model Registry to version your best RAG config.")
    print("8. In Databricks, MLflow is built-in with Unity Catalog")
    print("   integration for governed model management.")


if __name__ == "__main__":
    main()
