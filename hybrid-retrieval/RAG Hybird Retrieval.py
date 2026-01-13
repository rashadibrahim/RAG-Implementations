from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Load and parse

loader = PyMuPDFLoader("company Policy.pdf")
docs = loader.load()

# Hierarchical splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", ".", " "])
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Hybrid retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore.as_retriever(), bm25_retriever], weights=[0.7, 0.3])


# Example query
# query = "What is the company's policy on remote work?"
# results = ensemble_retriever.invoke(query)
# for i, res in enumerate(results):
#     print(f"Result {i+1}:\n{res.page_content}\n")


# Synthesizer with Groq LLM
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.0,
)

# Define a prompt template for RAG
prompt = ChatPromptTemplate.from_template(
    "You are an assistant for querying an employee handbook."
    "Use the following context to answer the question. If you don't know the answer, say so.\n"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)

query = "What is the policy on gross misconduct?"


retrieved_docs = ensemble_retriever.invoke(query)
context_text = "\n\n".join(d.page_content for d in retrieved_docs)

qa_chain = prompt | llm | StrOutputParser()

response = qa_chain.invoke({"context": context_text, "question": query})
print(response)