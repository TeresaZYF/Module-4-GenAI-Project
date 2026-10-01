"""
Demo 05: LangChain RAG Pipeline for Canadian Bank Filings
==========================================================
Instructor demo script - builds a complete RAG pipeline using
LangChain with ChromaDB for Canadian bank annual report Q&A.

Prerequisites:
    pip install langchain langchain-openai langchain-community chromadb
    Set OPENAI_API_KEY environment variable

Usage:
    python demos/05_langchain_rag.py
"""

import os
import json
from typing import List

SAMPLE_DOCS = [
    "Royal Bank of Canada: The provision for credit losses was $3.4 billion for fiscal 2024, up from $2.9 billion. Gross impaired loans increased to $5.8 billion. The allowance for credit losses was 0.72% of total loans.",
    "Royal Bank of Canada: Average daily VaR was $42 million in fiscal 2024 vs $38 million in 2023. A 100bp rate increase would add $320 million to net interest income over 12 months.",
    "Toronto-Dominion Bank: Net interest income was $23.8 billion, up 12% from $21.3 billion. Net interest margin expanded 14 basis points to 1.72%. US retail banking revenue grew 15%.",
    "Bank of Montreal: PCL was $2.3 billion in fiscal 2024, up from $1.5 billion in 2023. The increase reflects Bank of the West integration and normalizing credit conditions.",
    "Bank of Nova Scotia: International Banking revenue grew 8% to $4.1 billion. Pacific Alliance countries (Mexico, Peru, Chile, Colombia) remain core growth markets.",
    "Canadian Imperial Bank of Commerce: Mortgage portfolio totalled $372 billion (45% of total loans). Uninsured mortgages were 62%. Variable-rate renewal risk is being closely monitored.",
    "Royal Bank of Canada: In accordance with OSFI Guideline B-15, the bank enhanced its climate risk framework. Committed to net-zero by 2050 with 2030 interim targets.",
    "Bank of Montreal: Deployed over 200 AI/ML models. Established AI Ethics Committee in anticipation of OSFI Guideline E-23 (Model Risk Management).",
]

def main():
    print("=" * 60)
    print("LangChain RAG Pipeline: Canadian Bank Filings")
    print("=" * 60)

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n  OPENAI_API_KEY not set. Showing pipeline structure only.")
        print("\n  Pipeline steps:")
        print("  1. Load documents (bank filing text)")
        print("  2. Split into chunks (RecursiveCharacterTextSplitter)")
        print("  3. Create embeddings (OpenAIEmbeddings)")
        print("  4. Store in ChromaDB vector store")
        print("  5. Create retriever (similarity search, k=4)")
        print("  6. Build RAG chain (RetrievalQA)")
        print("  7. Query and get cited answers")
        return

    try:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.vectorstores import Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
    except ImportError as e:
        print(f"  Missing: {e}")
        print("  Install: pip install langchain langchain-openai langchain-community chromadb")
        return

    # Setup
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_texts(SAMPLE_DOCS, embeddings, collection_name="canadian_banks_demo")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template="""You are a Canadian bank compliance analyst. Answer based ONLY on the provided context.
Cite the bank name and metric for every claim. If information is not in the context, say so.

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    queries = [
        "What is RBC's provision for credit losses?",
        "Which bank has the largest mortgage portfolio?",
        "What are the climate risk commitments across Canadian banks?",
        "How many AI models has BMO deployed?",
    ]

    for query in queries:
        print(f"\n  Q: {query}")
        result = qa_chain.invoke({"query": query})
        print(f"  A: {result['result'][:200]}...")
        print(f"  Sources: {len(result.get('source_documents', []))} chunks retrieved")

    vectorstore.delete_collection()

    print(f"\n{'=' * 60}")
    print("KEY TAKEAWAYS")
    print(f"{'=' * 60}")
    print("1. LangChain provides a clean abstraction for RAG pipelines.")
    print("2. The prompt template enforces citation and grounding.")
    print("3. RetrievalQA chains retriever + LLM in a single call.")
    print("4. Always return source documents for debugging.")
    print("5. ChromaDB runs in-memory for prototyping - use persistent")
    print("   storage for production.")

if __name__ == "__main__":
    main()
