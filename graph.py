from langgraph.graph import StateGraph, END
from nodes import GraphState, retrieve, grade_documents, generate, web_search

def route_question(state: GraphState):
    """
    Routes the workflow to either web search or final generation 
    based on the grader's decision.
    """
    print("🚦 ---ROUTE: CHECKING GRADER DECISION---")
    if state.get("web_search"):
        print("   👉 Routing to: Web Search")
        return "web_search"
    else:
        print("   👉 Routing to: Generate")
        return "generate"

# 1. Initialize the Graph state
workflow = StateGraph(GraphState)

# 2. Add the Nodes (Imported directly from your nodes.py file)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# 3. Connect the Nodes with Edges (The Flowchart Logic)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional Edge: After grading, do we generate or search the web?
workflow.add_conditional_edges(
    "grade_documents",
    route_question,
    {
        "web_search": "web_search",
        "generate": "generate",
    }
)

# If we had to search the web, go to generate next.
workflow.add_edge("web_search", "generate")
# After generation, the loop ends.
workflow.add_edge("generate", END)

# 4. Compile the Graph into an executable application
app = workflow.compile()

# --- 5. TERMINAL TESTING INTERFACE ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 11/10 ADVANCED AGENTIC RAG INITIALIZED!")
    print("="*50)
    print("The system is ready. Try asking a highly technical question based on the PDFs you downloaded.")
    print("(Type 'quit' or 'exit' to stop)\n")
    
    while True:
        user_question = input("🧑 User Question: ")
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Shutting down...")
            break
            
        # The initial state is just the user's question
        inputs = {"question": user_question}
        
        print("\n--- SYSTEM TRACE ---")
        # Run the graph and stream the output to the console
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"\n[Finished Node: {key.upper()}]")
        
        print("\n" + "="*50)
        print("🤖 FINAL GENERATED ANSWER:")
        print("="*50)
        print(value.get("generation", "Error: No generation produced."))
        print("\n" + "-"*50 + "\n")