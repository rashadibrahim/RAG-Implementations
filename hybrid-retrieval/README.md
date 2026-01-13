# RAG with Hybrid Retrieval

A Retrieval-Augmented Generation (RAG) code that answers questions about employee handbooks using hybrid document retrieval and LLM-powered synthesis.

## Overview

This code enables intelligent question-answering on PDF documents through:

- Recursive Document Chunking: Splits documents into 500-token chunks with 100-token overlap for better context preservation
- Hybrid Retrieval: Combines semantic (FAISS vector search) and lexical (BM25) ranking (70% semantic, 30% lexical)
- Vector Embeddings: Uses HuggingFace sentence transformers (all-MiniLM-L6-v2) for semantic representation
- LLM Synthesis: Leverages Groq's Qwen 3 model for fast, accurate answer generation

## Features

- Hybrid search combining semantic and lexical retrieval
- Fast LLM-powered answer synthesis via Groq
- Weighted ensemble retriever for balanced results
- FAISS-based vector storage for efficient similarity search

## How It Works

1. Load: Reads PDF documents
2. Split: Breaks documents into overlapping chunks
3. Embed: Converts chunks into vector embeddings and stores in FAISS
4. Retrieve: Combines semantic similarity search and keyword-based ranking
5. Generate: Passes retrieved context to LLM for answering

## Usage Example

```python
query = "What is the policy on gross misconduct?"
retrieved_docs = ensemble_retriever.invoke(query)
context_text = "\n\n".join(d.page_content for d in retrieved_docs)
response = qa_chain.invoke({"context": context_text, "question": query})
print(response)
```

## Requirements

- langchain-community
- langchain-groq
- langchain-core
- sentence-transformers
- faiss-cpu
- pymupdf
- python-dotenv

## Setup

1. Create a .env file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key
   ```

2. Update the PDF file path in the script:
   ```python
   loader = PyMuPDFLoader("your-document.pdf")
   ```

3. Run the script to query your document
