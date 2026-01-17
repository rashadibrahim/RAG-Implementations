from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing import List
from compay_retriever import policy_retriever # a separate file that handles document retrieval for a company policy PDF
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    question: str
    documents: List[Document] = []
    generation: str = ""


def retrieve(state):
    """
    Retrive Documents
    Args:
        state (dict): The current state containing the question.
    Returns:
        state (dict): The updated state with retrieved documents.
    """
    question = state["question"]
    documents = policy_retriever(question)

    return {"documents": documents, "question": question}



class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"  
    )

# Prompt: https://smith.langchain.com/hub/efriis/self-rag-retrieval-grader
grade_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
You are a grader assessing relevance of a retrieved document to a user question. It does not need to be a stringent test. 
The goal is to filter out erroneous retrievals. If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """
    ),
    HumanMessagePromptTemplate.from_template("""
Retrieved document: {document}
User question: {question}
    """
    ),
])

llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.
    Args:
        state (dict): The current graph state.
    Returns:
        state (dict): Updates documents key with only filtered relevant documents.
    """

    question = state["question"]
    documents = state["documents"]
    filtered_documents = []
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_documents.append(doc)
        else:
            print(f"Document filtered out: {doc.page_content[:50]}...")
            continue
        
    return {"documents": filtered_documents, "question": question}
    



# prompt: https://smith.langchain.com/hub/efriis/self-rag-question-rewriter
re_write_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. 
Look at the input and try to reason about the underlying sematic intent / meaning.
"""),
    HumanMessagePromptTemplate.from_template("""
Here is the initial question: {question}

Formulate an improved question.
""")])

question_re_writer = re_write_prompt | llm | StrOutputParser()


def transform_query(state):
  better_question = question_re_writer.invoke({"question": state["question"]})
  print("Transformed Question:", better_question)
  return {"question": better_question}




gen_prompt = ChatPromptTemplate.from_messages([
  HumanMessagePromptTemplate.from_template("""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
""")
])

generation_chain = gen_prompt | llm | StrOutputParser()

def generate(state):
  """
  Generate answer
  Args:
      state (dict): The current graph state.
  Returns:
      state (dict): New key added to state, generation, that contains LLM generation. 
  """
  question = state["question"]
  documents = state["documents"]
  generation = generation_chain.invoke({
      "question": question,
      "context": documents})
  return {"generation": generation, "question": question, "documents": documents}




workflow = StateGraph(State)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
  "grade_documents",
  lambda state: ("generate" if len(state["documents"]) > 0 else "transform_query"),
  {
  "generate": "generate",
  "transform_query": "transform_query"
  }
)

workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)
app = workflow.compile()


if __name__ == "__main__":
    initial_state: State = {
        "question": "What is the company's policy on Hygiene?",
        "documents": [],
        "generation": ""
    }
    final_state = app.invoke(initial_state)
    print("Final State:", final_state)








