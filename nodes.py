import os
from typing import List, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from typing_extensions import TypedDict

# --- 1. STATE DEFINITION ---
class GraphState(TypedDict):
    """The memory of our agent. It passes this state between nodes."""
    question: str
    documents: List[Any]
    web_search: bool
    generation: str

# --- 2. INITIALIZE MODELS ---


# The LLM Engine (Fast & Smart)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# The Embedding Engine (Matches Step 1 exactly)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Connect to the Smart Database we built in Step 1
vectorstore = Chroma(
    persist_directory="./data/chroma_db",
    embedding_function=embeddings
)

# --- 3. THE GRAPH NODES (ACTIONS) ---

def retrieve(state: GraphState):
    """Fetches documents from the database."""
    print("🔎 ---NODE: RETRIEVE---")
    question = state["question"]
    
    # We fetch 6 chunks. Because we used smart chunking, these are highly relevant.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """The Quality Inspector: Grades chunks to filter out hallucinations."""
    print("⚖️ ---NODE: GRADE DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    # We use batch grading to save API calls and run super fast
    class DocumentGrade(BaseModel):
        index: int = Field(description="The index of the document")
        is_relevant: bool = Field(description="True if relevant to question, False if irrelevant")

    class GradeBatch(BaseModel):
        grades: List[DocumentGrade] = Field(description="List of grades for all documents.")
        
    structured_llm = llm.with_structured_output(GradeBatch)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict grading agent. Evaluate the relevance of each document to the user's question. "
                   "If it contains keywords or meaning related to the question, mark it True. Otherwise, False."),
        ("human", "Question: {question}\n\nDocuments:\n{documents}")
    ])
    
    docs_with_indices = "\n\n".join(
        [f"--- Index: {i} ---\n{doc.page_content}" for i, doc in enumerate(documents)]
    )
    
    filtered_docs = []
    web_search = False
    
    try:
        grader_chain = prompt | structured_llm
        result = grader_chain.invoke({"question": question, "documents": docs_with_indices})
        grade_dict = {g.index: g.is_relevant for g in result.grades}
        
        for i, d in enumerate(documents):
            if grade_dict.get(i, False):
                filtered_docs.append(d)
                print(f"   ✅ Doc {i} passed inspection.")
            else:
                print(f"   ❌ Doc {i} failed inspection (Discarding).")
    except Exception as e:
        print(f"   ⚠️ Grader error: {e}. Defaulting to web search.")
        
    # If all documents were garbage, trigger fallback!
    if not filtered_docs:
        print("⚠️ All local data failed! Triggering Web Search Fallback...")
        web_search = True
        
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state: GraphState):
    """Searches the web if local documents are insufficient."""
    print("🌐 ---NODE: WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    
    search_tool = DuckDuckGoSearchRun()
    
    try:
        print("   🔍 Pinging DuckDuckGo...")
        docs = search_tool.invoke(question)
    except Exception as e:
        print(f"   ⚠️ DuckDuckGo Connection Error: {e}")
        docs = "Web search is currently unavailable due to a network connection error."
    
    from langchain_core.documents import Document
    web_results = Document(page_content=docs)
    documents.append(web_results)
    
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    """Generates the final answer using ONLY the approved context."""
    print("✍️ ---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # 11/10 Feature: Forcing the LLM to use the metadata from Step 1!
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite technical AI. Use the provided context to answer the question. "
                   "If the context contains a table, render it perfectly in Markdown. "
                   "CRITICAL: You must cite your sources at the end of your answer using the metadata provided! "
                   "If the answer isn't in the context, say 'I do not have enough information.'\n\n"
                   "Context: {context}"),
        ("human", "{question}")
    ])
    
    # Format the documents to explicitly show the LLM the metadata tags
    formatted_docs = []
    for doc in documents:
        source = doc.metadata.get('Source_Paper', 'Unknown Source')
        header = doc.metadata.get('Header 2', '') or doc.metadata.get('Header 1', '')
        formatted_docs.append(f"[Source: {source} | Section: {header}]\n{doc.page_content}")
        
    docs_content = "\n\n---\n\n".join(formatted_docs)
    
    rag_chain = prompt | llm
    generation = rag_chain.invoke({"context": docs_content, "question": question})
    
    return {"generation": generation.content, "question": question}
