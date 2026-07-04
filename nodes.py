import os
from typing import List, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from typing_extensions import TypedDict

# 🚀 THE FIX: We import the raw library directly at the top, completely bypassing LangChain
from duckduckgo_search import DDGS

# --- STATE DEFINITION ---
class GraphState(TypedDict):
    """Represents the state of our graph/agent."""
    question: str
    documents: List[Any]
    web_search: bool
    generation: str

# --- INITIALIZE MODELS ---
# Using Llama 3.1 hosted on Groq's blazing fast LPUs!
# Note: Streamlit handles the API key securely via secrets.
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Local embeddings (These are tiny, so they run fine on your CPU)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Connect to our Smart Chunked Database
vectorstore = Chroma(
    persist_directory="./data/chroma_db",
    embedding_function=embeddings
)

# --- GRAPH NODES ---
def retrieve(state: GraphState):
    """Retrieves documents from the vector database."""
    print("🔎 ---NODE: RETRIEVE---")
    question = state["question"]
    
    # Fetch top 8 chunks to ensure we don't miss large tables
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """Evaluates if the retrieved documents are relevant using a single batch API call."""
    print("⚖️ ---NODE: GRADE DOCUMENT RELEVANCE (BATCH MODE)---")
    question = state["question"]
    documents = state["documents"]
    
    class DocumentGrade(BaseModel):
        index: int = Field(description="The index of the document")
        is_relevant: bool = Field(description="True if relevant, False if irrelevant")

    class GradeBatch(BaseModel):
        grades: List[DocumentGrade] = Field(description="List of grades for all provided documents.")
        
    structured_llm = llm.with_structured_output(GradeBatch)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict grading agent. You will receive a question and a batch of documents numbered by index. "
                   "Evaluate each document individually. Return a list containing a grade object for every document index provided. "
                   "If the document contains keywords or meaning related to the question, mark it True. Otherwise, mark it False."),
        ("human", "User question: {question}\n\nRetrieved documents:\n{documents}")
    ])
    
    docs_with_indices = "\n\n".join(
        [f"--- Document Index: {i} ---\n{doc.page_content}" for i, doc in enumerate(documents)]
    )
    
    print(f"   🤖 Sending batch of {len(documents)} documents to Groq Llama 3.1...")
    
    filtered_docs = []
    web_search = False
    
    try:
        retrieval_grader = prompt | structured_llm
        result = retrieval_grader.invoke({"question": question, "documents": docs_with_indices})
        
        grade_dict = {g.index: g.is_relevant for g in result.grades}
        
        for i, d in enumerate(documents):
            if grade_dict.get(i, False):
                print(f"   ✅ Grade: Document {i} is relevant")
                filtered_docs.append(d)
            else:
                print(f"   ❌ Grade: Document {i} is irrelevant (Discarding)")
                
    except Exception as e:
        print(f"   ⚠️ Batch grading failed due to error: {e}. Triggering web search to be safe.")
        
    if not filtered_docs:
        print("⚠️ All local documents failed! Triggering Web Search Fallback...")
        web_search = True
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state: GraphState):
    """Searches the web if local documents are insufficient."""
    print("🌐 ---NODE: WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    
    try:
        print("   🔍 Pinging DuckDuckGo...")
        # Using the raw library directly to avoid LangChain bugs
        results = DDGS().text(question, max_results=3)
        
        if results:
            docs = "\n\n".join([f"Content: {r.get('body')}" for r in results])
        else:
            docs = "No web results found."
    except Exception as e:
        print(f"   ⚠️ DuckDuckGo Connection Error: {e}")
        docs = "Web search is currently unavailable due to a network connection error."
    
    from langchain_core.documents import Document
    web_results = Document(page_content=docs)
    documents.append(web_results)
    
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """Generates the final answer based on the context."""
    print("✍️ ---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a highly technical engineering AI. Use the provided context to answer the question. "
                   "CRITICAL INSTRUCTION: If the context contains a table, or if the user asks for a table, "
                   "YOU MUST output it strictly in proper Markdown table format. Do not summarize the rows. "
                   "If the context does not contain the answer, do NOT guess. Say 'I do not have enough information.'\n\n"
                   "Context: {context}"),
        ("human", "{question}")
    ])
    
    docs_content = "\n\n".join(doc.page_content for doc in documents)
    rag_chain = prompt | llm
    
    print("   🤖 Generating final response via Groq...")
    generation = rag_chain.invoke({"context": docs_content, "question": question})
    
    return {"generation": generation.content, "question": question}
