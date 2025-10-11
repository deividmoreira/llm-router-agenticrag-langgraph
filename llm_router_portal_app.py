# LLM Router Dev Portal - Streamlit app with Agentic RAG and routing

# Standard library helpers for filesystem operations
import os

# Streamlit powers the web UI
import streamlit as st

# Loads environment variables from a .env file when present
from dotenv import load_dotenv

# Type helpers for the LangGraph state
from typing import TypedDict, Literal

# Groq client used for routing and final answer generation
from langchain_groq import ChatGroq

# Vector store used for RAG retrieval
from langchain_community.vectorstores import FAISS

# Embedding model for turning chunks into vectors
from langchain_community.embeddings import FastEmbedEmbeddings

# DuckDuckGo search tool for web fallback
from langchain_community.tools import DuckDuckGoSearchRun

# LangGraph state machine utilities
from langgraph.graph import StateGraph, END

# Load environment variables defined in .env
load_dotenv()

# Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in the environment or .env file!")
    st.stop() 

# Location of the FAISS index
VECTORSTORE_PATH = "faiss_index"

# Cache the heavyweight LLM client
@st.cache_resource
def load_response_llm():
    
    print("Loading Groq LLM...") 
    
    try:
        llm = ChatGroq(api_key = groq_api_key, model = "meta-llama/llama-4-maverick-17b-128e-instruct", temperature = 0.1)
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        st.stop()

# Cache the RAG retriever
@st.cache_resource
def load_retriever():
    
    print("Loading RAG retriever...") 
    
    if not os.path.exists(VECTORSTORE_PATH):
        st.error(f"FAISS index not found in '{VECTORSTORE_PATH}'. Run 'prepare_rag_index.py'.")
        st.stop()
    
    try:
        embedding_model = FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")
        vector_store = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization = True)
        retriever = vector_store.as_retriever(search_kwargs = {'k': 5})
        return retriever
    except Exception as e:
        st.error(f"Failed to load RAG retriever: {e}")
        st.stop()

# Graph state shared across the agent
class GraphState(TypedDict):
    query: str
    source_decision: Literal["RAG", "WEB", ""]
    rag_context: str | None
    web_results: str | None
    final_answer: str | None

# Routing node
def route_query_node(state: GraphState) -> dict:
    
    """
    Analyse the query and decide the best source (RAG or WEB).
    Update 'source_decision' in the state.
    """
    
    print("--- Node: Query Routing ---")
    
    query = state["query"]

    # Prompt with instructions and examples
    prompt = f"""Classify the user query below and route it to the most suitable source:
    1. **RAG**: Internal knowledge base with support procedures, product configurations, and proprietary guides. Use RAG for queries such as "how do I configure X in our platform?", "what is the setup for Y?", or "where is the internal documentation about Z?".
    2. **WEB**: Open web search for questions about third-party tools (e.g. Anaconda, Python, Excel), technology news, undocumented errors, or topics that are not covered by the internal documents.

    Examples:
    - Query: "How do I configure the internal email server?" -> Answer: RAG
    - Query: "What is the latest version of Streamlit?" -> Answer: WEB
    - Query: "What is the process to reset the ABC system password?" -> Answer: RAG
    - Query: "How do I install the Python interpreter on Windows 11?" -> Answer: WEB
    - Query: "How to install Anaconda Python" -> Answer: WEB

    Classify the following query:
    User Query: '{query}'

    Return only the word 'RAG' or the word 'WEB'."""

    try:
        # Lightweight LLM dedicated to routing with a slightly higher temperature
        router_llm = ChatGroq(api_key = groq_api_key,
                              model = "llama3-8b-8192",
                              temperature = 0.4)

        # Run the router
        response = router_llm.invoke(prompt)

        # Extract the response
        raw_decision = response.content 

        print(f"DEBUG: Router decision raw output: '{raw_decision}'")

        # Sanitise the decision and keep only the target token
        decision = raw_decision.strip().upper().replace("'", "").replace('"', '') 

        # Default to WEB when the answer is unexpected
        if decision == "RAG":
            final_decision = "RAG"
        else:
            if decision != "WEB":
                 print(f"  Invalid/unexpected router decision: '{raw_decision}'. Falling back to WEB.") 
            final_decision = "WEB"

        print(f"  Final router decision: {final_decision}") 

        return {"source_decision": final_decision}

    except Exception as e:
        print(f"  Error in routing node: {e}") 
        print("  Using WEB as fallback due to the error.") 
        return {"source_decision": "WEB"}

# Node responsible for RAG retrieval
def retrieve_rag_node(state: GraphState) -> dict:
    
    query = state["query"]
    
    try:
        local_retriever = load_retriever() 
        results = local_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in results])
        if not context:
            print("  No RAG context retrieved.") 
            return {"rag_context": "No relevant internal documents found."}
        else:
            print(f"  RAG context retrieved ({len(context)} chars).") 
            return {"rag_context": context}
    except Exception as e:
        print(f"  Error in RAG node: {e}") 
        return {"rag_context": f"Error while querying internal documents: {e}"}

# Node responsible for web search
def search_web_node(state: GraphState) -> dict:
    
    query = state["query"]
    
    try:
        web_search_tool = DuckDuckGoSearchRun()
        results = web_search_tool.run(query)
        if not results:
            print("  No web search results.") 
            return {"web_results": "No relevant web results were found."}
        else:
            print(f"  Web search returned results ({len(results)} chars).") 
            return {"web_results": results}
    except Exception as e:
        print(f"  Error in web search node: {e}") 
        return {"web_results": f"Error while running web search: {e}"}

# Node responsible for the final answer
def generate_answer_node(state: GraphState) -> dict:
    
    print("--- Node: Final Answer Generation ---") 
    
    query = state["query"]
    rag_context = state.get("rag_context")
    web_results = state.get("web_results")
    context_provided = ""
    source_used = "None"

    if rag_context != "No relevant internal documents found.":
        context_provided = f"Internal knowledge base context:\n{rag_context}"
        source_used = "RAG"
        print("  Using RAG context to craft the answer.") 
    elif web_results != "No relevant web results were found.":
        context_provided = f"Web search results:\n{web_results}"
        source_used = "WEB"
        print("  Using web results to craft the answer.") 
    else:
        context_provided = "No additional information was found in the available sources."
        print("  No useful context retrieved to support the answer.") 

    prompt = f"""You are a technical support assistant. Answer the user's question clearly and concisely, using ONLY the information provided in the context below when available.
    User Query: {query}
    {context_provided}
    Answer:"""

    try:
        llm_resposta_final = load_response_llm()
        response = llm_resposta_final.invoke(prompt)
        final_answer = response.content
        print(f"  Answer generated using source: {source_used}") 
        return {"final_answer": final_answer}
    except Exception as e:
        print(f"  Error in answer generation node: {e}") 
        return {"final_answer": f"Apologies, an error occurred while generating the answer: {e}"}

# Conditional edge that routes between RAG and web
def decide_source_edge(state: GraphState) -> Literal["retrieve_rag_node", "search_web_node"]:
    
    decision = state["source_decision"]
    
    print(f"--- Conditional edge: decision received = '{decision}' ---") 
    
    if decision == "RAG":
        print("  Routing to RAG node.") 
        return "retrieve_rag_node"
    else:
        print("  Routing to web search node.") 
        return "search_web_node"

# Compile the LangGraph graph
@st.cache_resource
def compile_graph():
    
    print("Compiling LangGraph...") 
    
    # Register the nodes
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("route_query_node", route_query_node)
    graph_builder.add_node("retrieve_rag_node", retrieve_rag_node)
    graph_builder.add_node("search_web_node", search_web_node)
    graph_builder.add_node("generate_answer_node", generate_answer_node)
    graph_builder.set_entry_point("route_query_node")
    
    # Conditional edge based on the router output
    graph_builder.add_conditional_edges("route_query_node", decide_source_edge, {
        "retrieve_rag_node": "retrieve_rag_node",
        "search_web_node": "search_web_node",
    })
    
    # Define the remaining edges
    graph_builder.add_edge("retrieve_rag_node", "generate_answer_node")
    graph_builder.add_edge("search_web_node", "generate_answer_node")
    graph_builder.add_edge("generate_answer_node", END)
    
    # Compile the graph
    try:
        app = graph_builder.compile()
        print("Graph compiled successfully!") 
        return app
    except Exception as e:
        st.error(f"Failed to compile the graph: {e}")
        st.stop()

# Configure the Streamlit page
st.set_page_config(page_title="LLM Router Dev Portal", page_icon=":100:", layout="centered")
st.title("LLM Router Dev Portal")
st.subheader("🤖 Agentic RAG with LangGraph and intelligent source routing")
st.markdown("Ask about support workflows or software tooling. The portal will decide whether to use your internal knowledge base or the open web.")

# Warm up caches and compile the graph
load_response_llm()
load_retriever()
app = compile_graph()

# User input
user_query = st.text_input("Your question:", key = "query_input")

# Submit button
if st.button("Get Answer", key = "submit_button"):
    
    if user_query:
        
        with st.spinner("Processing your query..."):
            
            try:
                inputs = {"query": user_query}

                # Execute the LangGraph agent
                final_state = app.invoke(inputs) 

                st.subheader("Answer:")
                st.markdown(final_state.get("final_answer", "The agent could not produce an answer."))
                st.info(f"Source selected by the router: {final_state.get('source_decision', 'N/A')}")

            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")
    else:
        st.warning("Please enter a question.")

# Sidebar instructions
st.sidebar.title("Usage tips")
st.sidebar.write("""
    - Ask focused questions about your scenario.
    - The router chooses between the vector store (RAG) and the open web.
    - Update the PDFs in `knowledge_base/` to reflect your organisation's content.
    - Generative AI can be wrong. Validate answers before acting.
""")

# Sidebar support shortcut
if st.sidebar.button("Support"):
    st.sidebar.write("Need help? Open an issue in the repository or email contato@llmrouter.dev")




    
