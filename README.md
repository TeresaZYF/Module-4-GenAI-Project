# Module 4: GenAI Application Development - Demo Scripts

## Overview

These demo scripts are instructor reference materials for Module 4: Introduction to GenAI Application Development. They demonstrate key concepts from each topic using Canadian bank filing data and serve as working examples that students can reference when building their capstone project (Canadian Bank Annual Report Research Agent).

**These are NOT starter code.** Students write all capstone code themselves.

## Environment Setup

### Step 1: Create a Python Environment

```bash
# Using conda
conda create -n module4 python=3.10 -y
conda activate module4

# OR using venv
python -m venv module4_env
source module4_env/bin/activate  # Linux/Mac
module4_env\Scripts\activate     # Windows
```

### Step 2: Install Required Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install openai langchain langchain-openai langchain-community
pip install chromadb sentence-transformers
pip install beautifulsoup4 requests lxml
pip install mlflow
pip install tiktoken
pip install jupyter
```

### Step 3: Set API Keys

```bash
# OpenAI API key (required for demos 04-10)
export OPENAI_API_KEY="your-key-here"       # Linux/Mac
set OPENAI_API_KEY=your-key-here            # Windows CMD
$env:OPENAI_API_KEY="your-key-here"         # Windows PowerShell
```

### Step 4: Verify Setup

```bash
python -c "import openai; print('OpenAI OK')"
python -c "import chromadb; print('ChromaDB OK')"
python -c "import langchain; print('LangChain OK')"
python -c "import mlflow; print('MLflow OK')"
```

## Demo Scripts

| Script | Topic | Description | API Required? |
|--------|-------|-------------|---------------|
| `01_download_filings.py` | Topic 1 | Canadian bank filing downloader (SEDAR+) | No |
| `02_embedding_comparison.py` | Topic 2 | OpenAI vs local embeddings on Canadian filings | Yes (OpenAI) |
| `03_retrieval_strategies.py` | Topic 2 | ChromaDB with Canadian bank metadata filtering | No (local) |
| `04_prompt_engineering.py` | Topic 3 | Prompting for Canadian bank filing Q&A | Yes (OpenAI) |
| `05_langchain_rag.py` | Topic 4 | LangChain RAG with Canadian bank data | Yes (OpenAI) |
| `06_databricks_rag_overview.py` | Topic 4 | Databricks RAG reference (conceptual) | No |
| `07_tool_calling.py` | Topic 5 | Function calling with Canadian filing tools | Yes (OpenAI) |
| `08_agentic_workflow.py` | Topic 5 | ReAct agent for cross-year comparison | Yes (OpenAI) |
| `09_llm_judge.py` | Topic 6 | LLM-as-Judge evaluation pipeline | Optional |
| `10_mlflow_genai.py` | Topic 6 | MLflow experiment tracking for GenAI | Optional |

## Canadian Banking Context

All demo scripts use data from Canada's Big Five banks:

| Bank | Ticker | Fiscal Year End |
|------|--------|-----------------|
| Royal Bank of Canada | RY | October 31 |
| Toronto-Dominion Bank | TD | October 31 |
| Bank of Montreal | BMO | October 31 |
| Bank of Nova Scotia | BNS | October 31 |
| Canadian Imperial Bank of Commerce | CM | October 31 |

**Data Sources:**
- SEDAR+ (www.sedarplus.ca) - Canada's electronic filing system
- Bank investor relations websites (direct PDF downloads)

**Key Regulatory References:**
- OSFI Guideline E-23: Model Risk Management (effective May 2027)
- OSFI Guideline B-15: Climate Risk Management
- OSFI Guideline B-10: Outsourcing of Business Activities

## Running the Demos

```bash
python demos/01_download_filings.py      # No API needed
python demos/02_embedding_comparison.py  # Needs OPENAI_API_KEY
python demos/03_retrieval_strategies.py  # No API needed
python demos/04_prompt_engineering.py    # Needs OPENAI_API_KEY
python demos/05_langchain_rag.py         # Needs OPENAI_API_KEY
python demos/06_databricks_rag_overview.py  # Conceptual - read only
python demos/07_tool_calling.py          # Needs OPENAI_API_KEY
python demos/08_agentic_workflow.py      # Needs OPENAI_API_KEY
python demos/09_llm_judge.py             # Optional API
python demos/10_mlflow_genai.py          # Optional API
```

## Notes

- Demos 09 and 10 include simulation modes that run without API keys.
- Demo 06 is a conceptual reference for Databricks-specific features.
- All scripts use gpt-4o-mini by default for cost efficiency.
- Canadian bank fiscal years end October 31 (not December 31 like US banks).
