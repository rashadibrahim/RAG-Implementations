# LangGraph Retrieval (Grading + Rewrite)

A lightweight retrieval pipeline using LangGraph and Groq LLM. It retrieves documents, filters them for relevance, rewrites the question when needed, and generates a concise answer. This is not a full self-RAG system — it only performs relevance grading on retrieved documents and query rewriting.

## Overview
- **Graph:** Four nodes — `retrieve`, `grade_documents`, `transform_query`, `generate` — orchestrated via a `StateGraph`.
- **Retrieval:** Gets `Document` objects via `policy_retriever(question)` from [compay_retriever.py](compay_retriever.py).
- **Relevance Grading:** Uses `ChatGroq` with a structured `GradeDocuments` schema to keep only relevant docs.
- **Query Rewriting:** If no relevant docs remain, rewrites the question and loops back to retrieval.
- **Generation:** Produces a short answer grounded in the filtered context.

## Generic Retriever
This project is generic: `policy_retriever` is a plug-in point. You can replace it with any retriever that returns `List[Document]` (e.g., Chroma, FAISS, API-based search, files on disk). Only the `retrieve` node depends on it.

## Requirements
- Python 3.10+
- `.env` with `GROQ_API_KEY=your_key_here`

## Install
```bash
pip install langgraph langchain-core langchain-groq python-dotenv pydantic typing_extensions
```

Optional (for vector stores like Chroma):
```bash
pip install chromadb
```

## Run
```bash
python AgenticRetrieval.py
```

## Configuration
- **Model:** Default `qwen/qwen3-32b` via Groq in [AgenticRetrieval.py](AgenticRetrieval.py).
- **Retriever:** Edit or replace `policy_retriever` in [compay_retriever.py](compay_retriever.py).

## Limitations
- Not a complete self-RAG: no answer grading, citation management, or iterative evidence gathering beyond query rewriting.

## Files
- [AgenticRetrieval.py](AgenticRetrieval.py): Main graph and pipeline.
- [compay_retriever.py](compay_retriever.py): Retriever function (replaceable).

## Notes
- If retrieval fails to find relevant context, the answer may be "I don't know".
- Keep your `.env` out of version control (add to `.gitignore`).
