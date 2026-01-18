from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


pdf_path = "company Policy.pdf" 

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} documents from PDF.")


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  
)


vector_store = FAISS.from_documents(documents, embeddings)


policy_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)


if __name__ == "__main__":
    query = "What is the policy for Harassment and Bullying?"
    relevant_docs = policy_retriever.invoke(query)

    print(f"Found {len(relevant_docs)} relevant documents for the query: '{query}'")