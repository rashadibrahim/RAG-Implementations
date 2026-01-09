import os
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader 
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from llama_index.core.retrievers import AutoMergingRetriever
from dotenv import load_dotenv
from llama_index.core import get_response_synthesizer
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import RetrieverQueryEngine
load_dotenv() # for GROQ_API_KEY


path = os.getcwd() # Any directory with PDFs or a direct PDF file path

input_files = None
if path.endswith('.pdf'): # in case a single pdf file is given
  input_files = [path]
else:
  # look for all pdfs in the directory
  input_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pdf')]



documents = SimpleDirectoryReader(
    input_files=input_files,
    file_extractor={".pdf": PyMuPDFReader()}     
).load_data()


node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[1024, 512, 128],
    chunk_overlap=50,  
)


nodes = node_parser.get_nodes_from_documents(documents)

embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

collection_name = "pdf_chunks" 
qdrant_client = QdrantClient(url="http://localhost:6333")

# Create collection if it doesn't exist
if not qdrant_client.collection_exists(collection_name=collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"Created new collection: {collection_name}")
else:
    print(f"Collection '{collection_name}' already exists.")

# Check if it's already indexed
collection_info = qdrant_client.get_collection(collection_name)
if collection_info.points_count > 0:
    print(f"Found {collection_info.points_count} existing vectors. Loading index from Qdrant (skipping re-indexing).")
    
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(persist_dir="./llama_storage", vector_store=vector_store)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
        storage_context=storage_context
    )
else:
    print("Collection is empty. Performing fresh indexing...")
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes, allow_update=True)
    
    index = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True
    )
    # Persist the storage context to disk so we can load it faster next time without re-indexing
    storage_context.persist(persist_dir="./llama_storage")
    print("Fresh indexing complete!")

# Retriever: Auto-merges hierarchies (retrieves children, falls back to parents)
retriever = AutoMergingRetriever(
    index.as_retriever(similarity_top_k=10), 
    storage_context=storage_context,
    # verbose=True  # Logs merging
)


llm = Groq(
    model="qwen/qwen3-32b",      
    temperature=0.1,
    max_tokens=1024,
)

# we use a synthesizer to combine retrieved chunks into a final answer
response_synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode="compact",  
)


query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

question = "What is the main topic of the PDF?"

response = query_engine.query(question)
print(response)
print("\nSources:")
for source in response.source_nodes:
    print(f"- {source.node.get_content()[:200]}... (score: {source.score:.3f})")



# you can use this function for Direct retrieval (no synthesizer/query engine needed)
def direct_retrieve(query: str):
    results = retriever.retrieve(query)
    for i, res in enumerate(results):
        print(f"Result {i+1} (score: {res.score:.3f}):\n{res.node.get_content()}\n")