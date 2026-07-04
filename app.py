import os
import sys


try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # If running locally without pysqlite3, just ignore and continue

import streamlit as st
from graph import app as agent_app

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Agentic Self-RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
<style>
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .thought-process {
        background-color: #1e1e2e;
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Architecture")
    st.markdown("""
    **Self-Corrective RAG**
    
    ---> Please note that the model only retrieves data from:
    1. Attention is All You Need (Vaswani et al., 2017)
    2. Physics Informed Neural Networks (Raissi et al., 2019)
    3. Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)
    4. Kolmogorov Arnold networks (KST) for PDEs (Zhang et al., 2023)
    5. ResNet: Deep Residual Learning for Image Recognition (He et al., 2015)
    6. RoFormer: Enhanced Transformer for Document Understanding (Li et al., 2022)
    
    ---> And then if the query is not found, it will attempt a web search using DuckDuckGo
    
    ---> If the data is not found, it will not hallicinate and will instead return a message indicating that the data was not found.
                
    """)
    st.divider()
    st.caption("Dont ask more queries, I have limited usage for my API")

# --- MAIN CHAT INTERFACE ---
st.title("🤖 Autonomous Self-Corrective RAG Assistant")
st.caption("Ask highly technical questions based on your ingested research papers.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- HANDLE USER INPUT ---
if prompt := st.chat_input("E.g., What is the mathematical formula for the attention mechanism?"):
    
    # 1. Add user message to chat UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Add assistant response
    with st.chat_message("assistant"):
        # We use a status box to show the agent's thought process!
        with st.status("🧠 Agent is thinking...", expanded=True) as status:
            
            # Fetch the API key from Streamlit's secure secrets instead of hardcoding it
            try:
                os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
            except FileNotFoundError:
                st.error("⚠️ GROQ_API_KEY missing from Streamlit Secrets!")
                st.stop()
            
            inputs = {"question": prompt}
            final_generation = ""
            
            # Stream the nodes as they execute in LangGraph
            for output in agent_app.stream(inputs):
                for key, value in output.items():
                    if key == "retrieve":
                        st.write("🔎 **Action:** Retrieving context from ChromaDB...")
                    elif key == "grade_documents":
                        st.write("⚖️ **Action:** Grading retrieved documents for relevance...")
                        if value.get("web_search"):
                            st.write("⚠️ **Alert:** Local documents failed. Triggering web search...")
                    elif key == "web_search":
                        st.write("🌐 **Action:** Executing DuckDuckGo web search fallback...")
                    elif key == "generate":
                        st.write("✍️ **Action:** Generating final response...")
                        final_generation = value.get("generation", "Error: No generation produced.")
            
            status.update(label="✅ Answer Generated!", state="complete", expanded=False)
            
        # 3. Display the final generated text
        st.markdown(final_generation)
        
    # 4. Save the assistant's final response to history
    st.session_state.messages.append({"role": "assistant", "content": final_generation})