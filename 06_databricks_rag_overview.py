"""
Demo 06: Databricks RAG Architecture for Canadian Bank Filings
===============================================================
Instructor demo - conceptual reference for Databricks-native RAG.
This script is meant to be READ, not executed outside Databricks.

This script demonstrates (conceptually):
- Databricks Vector Search for filing embeddings
- Model Serving for LLM inference
- Unity Catalog for data governance (OSFI compliance)
- MLflow for experiment tracking

Prerequisites (Databricks workspace):
    Databricks Runtime 14.0+ ML
    Unity Catalog enabled
    Vector Search endpoint provisioned

Usage:
    Read and discuss in class. Execute in Databricks workspace.
"""

# ──────────────────────────────────────────────────────────
# NOTE: This code runs ONLY in a Databricks workspace.
# It is provided as a conceptual reference for classroom
# discussion and as a template for students deploying on
# Databricks during the capstone project.
# ──────────────────────────────────────────────────────────


# ── STEP 1: Load Canadian Bank Filings into Delta Table ──

# from pyspark.sql import functions as F
#
# # Load parsed filing chunks from the data pipeline
# filings_df = spark.read.json("/mnt/data/canadian_bank_filings/chunks/")
#
# # Save as a Delta table in Unity Catalog
# filings_df.write.format("delta").mode("overwrite").saveAsTable(
#     "compliance_catalog.filings.canadian_bank_chunks"
# )
#
# # Verify
# display(spark.table("compliance_catalog.filings.canadian_bank_chunks")
#         .groupBy("bank", "filing_type").count())


# ── STEP 2: Create Vector Search Index ──

# from databricks.vector_search.client import VectorSearchClient
#
# vsc = VectorSearchClient()
#
# # Create a Vector Search endpoint (one-time setup)
# # vsc.create_endpoint(name="compliance-vs-endpoint", endpoint_type="STANDARD")
#
# # Create a Delta Sync index - auto-syncs with the Delta table
# index = vsc.create_delta_sync_index(
#     endpoint_name="compliance-vs-endpoint",
#     source_table_name="compliance_catalog.filings.canadian_bank_chunks",
#     index_name="compliance_catalog.filings.chunk_index",
#     pipeline_type="TRIGGERED",  # or "CONTINUOUS" for real-time sync
#     primary_key="chunk_id",
#     embedding_source_column="text",
#     embedding_model_endpoint_name="databricks-bge-large-en",  # Databricks-hosted
# )
#
# # Query the index
# results = index.similarity_search(
#     query_text="What are RBC's credit risk factors?",
#     columns=["text", "bank", "section", "fiscal_year"],
#     num_results=5,
#     filters={"bank": "Royal Bank of Canada"},
# )


# ── STEP 3: Model Serving with AI Gateway ──

# # Databricks Model Serving provides access to foundation models
# # with AI Gateway for governance and rate limiting
#
# import mlflow.deployments
#
# client = mlflow.deployments.get_deploy_client("databricks")
#
# # External model endpoint (e.g., OpenAI via AI Gateway)
# response = client.predict(
#     endpoint="openai-gpt4o-mini",  # configured in Model Serving
#     inputs={
#         "messages": [
#             {"role": "system", "content": "You are a Canadian bank compliance analyst."},
#             {"role": "user", "content": "What are the key credit risk factors for RBC?"},
#         ],
#         "max_tokens": 500,
#         "temperature": 0,
#     },
# )


# ── STEP 4: Complete RAG Chain on Databricks ──

# from langchain_community.vectorstores import DatabricksVectorSearch
# from langchain_community.chat_models import ChatDatabricks
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
#
# # Vector store retriever
# vs_retriever = DatabricksVectorSearch(
#     index, text_column="text",
#     columns=["bank", "section", "fiscal_year"],
# ).as_retriever(search_kwargs={"k": 5})
#
# # LLM via Model Serving
# llm = ChatDatabricks(endpoint="openai-gpt4o-mini", temperature=0)
#
# # Canadian bank compliance prompt
# prompt = PromptTemplate(
#     template="""You are a compliance analyst reviewing Canadian bank filings.
# Answer based ONLY on the context. Cite bank name, document, and section.
# If the context does not contain the answer, say so.
#
# Context: {context}
# Question: {question}
# Answer:""",
#     input_variables=["context", "question"],
# )
#
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm, retriever=vs_retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt},
# )


# ── STEP 5: MLflow Tracking ──

# import mlflow
#
# with mlflow.start_run(run_name="canadian_bank_rag_v1"):
#     mlflow.log_param("embedding_model", "databricks-bge-large-en")
#     mlflow.log_param("llm_model", "gpt-4o-mini")
#     mlflow.log_param("retrieval_k", 5)
#     mlflow.log_param("chunk_size", 800)
#
#     # Run evaluation
#     mlflow.log_metric("faithfulness", 4.2)
#     mlflow.log_metric("relevance", 4.5)
#     mlflow.log_metric("avg_cost_per_query", 0.005)
#
#     # Log prompt as artifact
#     mlflow.log_text(prompt.template, "prompt_template.txt")


# ── COMPLIANCE NOTES FOR CANADIAN BANKS ──

def print_compliance_notes():
    print("=" * 60)
    print("Databricks RAG: Canadian Banking Compliance Notes")
    print("=" * 60)
    print()
    print("OSFI GUIDELINE CONSIDERATIONS:")
    print("-" * 50)
    print("E-23 (Model Risk Management, effective May 2027):")
    print("  - GenAI/RAG systems may be classified as models")
    print("  - Requires model inventory, validation, monitoring")
    print("  - Document the RAG system in your model inventory")
    print()
    print("B-10 (Outsourcing):")
    print("  - Cloud AI services (OpenAI API, Azure) = outsourcing")
    print("  - Requires risk assessment and due diligence")
    print("  - Databricks on Azure Canada Central/East provides")
    print("    Canadian data residency")
    print()
    print("B-15 (Climate Risk):")
    print("  - Your RAG system can help analysts review climate")
    print("    disclosures for B-15 compliance monitoring")
    print()
    print("DATABRICKS ADVANTAGES FOR CANADIAN BANKS:")
    print("-" * 50)
    print("  - Azure Canada Central/East regions for data residency")
    print("  - Unity Catalog for column-level access control")
    print("  - Audit logging for all data access and model calls")
    print("  - AI Gateway for rate limiting and cost control")
    print("  - Delta Sharing for secure cross-team data access")


if __name__ == "__main__":
    print_compliance_notes()
    print()
    print("NOTE: The code above runs only in a Databricks workspace.")
    print("It is provided as a reference for classroom discussion")
    print("and as a template for capstone deployment.")
