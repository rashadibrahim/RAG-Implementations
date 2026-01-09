# Hierarchical PDF RAG

A Retrieval-Augmented Generation system for intelligent PDF querying using hierarchical document chunking and vector retrieval.

## Overview

This code enables semantic search and question-answering on PDF documents through:

- **Hierarchical Chunking**: Splits documents into multiple chunk sizes (1024, 512, 128 tokens) for multi-level semantic understanding
- **Vector Embeddings**: Uses HuggingFace embeddings (all-MiniLM-L6-v2) for semantic representation
- **Auto-Merging Retrieval**: Intelligently merges document hierarchies to provide context-aware results
- **LLM Integration**: Leverages Groq API for generating coherent answers from retrieved chunks

## Features

- 📄 Batch process multiple PDF files from a directory
- 🔍 Semantic similarity search with hierarchical context
- 🤖 LLM-powered answer synthesis
- 💾 Persistent vector storage with Qdrant
- ⚡ Smart caching to avoid re-indexing

## Requirements

- Python 3.8+
- Qdrant vector database (running at `http://localhost:6333`)
- GROQ_API_KEY environment variable


The system will:
1. Scan the current directory for PDF files
2. Parse and chunk documents hierarchically
3. Generate embeddings and store in Qdrant
4. Answer the query and display source citations

## Customization

### Configure PDF Source

Specify your PDF file path(s) at the top of the script:
```python
# Option 1: Single PDF file
path = r"C:\path\to\your\file.pdf"

# Option 2: Directory with multiple PDFs (default: Current Working Directory)
path = os.getcwd()
```

### Modify Chunk Sizes

Adjust chunk sizes in `node_parser`:
```python
chunk_sizes=[1024, 512, 128]  # Adjust hierarchy levels
```

### Change the Query

Customize your question at the bottom:
```python
question = "What is the main topic of the PDF?"
```
