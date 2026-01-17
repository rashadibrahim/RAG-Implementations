from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
import os



# Initialize embedding model and vector store
embidding_model = SentenceTransformer("all-MiniLM-L6-v2")


db_location = "./chroma_db"

add_docs = not os.path.exists(db_location) # Bool Variable to check if we need to add documents

# print("Loaded", len(documents), "documents from PDF.")
vector_store = chromadb.PersistentClient(path=db_location)
collection = vector_store.get_or_create_collection(
  name="compay-policy",
  metadata={"description": "PDF document embeddings for RAG"}
)


def policy_retriever(question: str, n_results: int = 5) -> str:
  query_embedding = embidding_model.encode([question]).tolist()[0]
  results = collection.query(
    query_embeddings=[query_embedding],
    n_results=n_results,
  )
  relevant_docs = []
  for i in range(len(results['ids'][0])):
    doc = Document(
      page_content=results['documents'][0][i],
      metadata=results['metadatas'][0][i]
    )
    relevant_docs.append(doc)
  return relevant_docs
  


if add_docs:
  documents = PyMuPDFLoader("./company Policy.pdf").load()
  ids = [str(i) for i in range(len(documents))]
  collection.add(
    ids=ids,
    embeddings=embidding_model.encode(
      [doc.page_content for doc in documents]
    ).tolist(),
    metadatas=[doc.metadata for doc in documents],
    documents=[doc.page_content for doc in documents])

  print(policy_retriever("what is The Disciplinary Procedure", 1))
  









