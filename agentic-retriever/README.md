# Agentic Retrieval

An agentic Retrieval-Augmented Generation (RAG) system that answers questions by intelligently retrieving and processing documents.

## What It Does

This system implements an agentic workflow that:
1. **Retrieves** relevant documents using a document retriever
2. **Grades** documents for relevance to the question
3. **Transforms** questions if no relevant documents are found, then retries
4. **Generates** concise answers based on the best documents

## Key Features

- **Intelligent Document Filtering**: Only uses relevant documents to answer questions
- **Query Optimization**: Automatically rewrites questions to improve retrieval
- **Flexible Retriever**: The document retriever can be swapped to work with any data source (PDFs, databases, APIs, etc.)
- **LLM-Powered**: Uses LangChain and Groq for fast inference

## Generic Design

This system is **not specific** to company policies or any particular document type. The `compay_retriever` module can be replaced with any retriever that returns documents, making it adaptable to:
- Knowledge bases
- Databases
- APIs
- Any document collection

## Usage

Modify the question in `AgenticRetrieval.py`:
```python
initial_state: State = {
    "question": "Your question here",
    "documents": [],
    "generation": ""
}
```

Then run:
```bash
python AgenticRetrieval.py
```

## How to Customize

**To use a different data source**, simply replace the `compay_retriever` import and modify the `retrieve()` function to use your own retriever that returns documents.
