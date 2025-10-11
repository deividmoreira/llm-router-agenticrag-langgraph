# LLM Router Dev Portal

Reference portal for developers who want to build Agentic RAG pipelines with automated source routing using LangGraph, Groq, and Streamlit. The repository ships with a demo-ready app, data preparation scripts, and guidance to adapt the workflow to your own knowledge bases.

## Why this matters
- Demonstrates how LangGraph orchestrates routing, RAG, and external tools in a single agent workflow.
- Provides a practical example of Groq's API for both routing (`llama3-8b-8192`) and final answers (`meta-llama/llama-4-maverick-17b-128e-instruct`).
- Offers a lightweight Streamlit interface tailor-made for workshops, live demos, or proofs of concept.

## Key components
- `prepare_rag_index.py`: builds the local FAISS index from the PDFs stored in `knowledge_base/`.
- `llm_router_portal_app.py`: Streamlit application that combines routing, RAG, and web search via LangGraph.
- `knowledge_base/`: add your own PDF documents here.
- `faiss_index/`: generated vector index after running the preparation script.

## Requirements
- Python 3.12
- Groq account and `GROQ_API_KEY` available as an environment variable (dotenv supported)
- Recent `pip` (`python -m pip install --upgrade pip`)

## Quick start
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Drop your PDFs into `knowledge_base/` and build the index:
   ```bash
   python prepare_rag_index.py
   ```
4. Launch the app:
   ```bash
   streamlit run llm_router_portal_app.py
   ```
5. Ask questions in the text field. The router switches between RAG and the web (DuckDuckGo) in real time.

See `GETTING_STARTED.md` for a step-by-step walkthrough.

## High-level architecture
```
Query → Routing node (Groq) ─┐
                             ├─> LangGraph agent → Final answer
RAG retriever (FAISS + FastEmbed) ─┘
Web search (DuckDuckGo) ───────────┘
```
- **Routing node:** classifies the query as `RAG` or `WEB`.
- **RAG node:** uses FAISS + FastEmbed (`BAAI/bge-small-en-v1.5`) to fetch relevant chunks.
- **Web node:** falls back to DuckDuckGo when broader context is needed.
- **Answer node:** generates the final answer constrained to the retrieved context.

## Project layout
```
.
├── faiss_index/             # Generated FAISS store
├── knowledge_base/          # PDFs for the internal knowledge base
├── llm_router_portal_app.py # Streamlit app
├── prepare_rag_index.py     # Script to build the FAISS index
├── requirements.txt         # Python dependencies
├── GETTING_STARTED.md       # Quickstart in Markdown
└── README.md
```

## Adapting to your domain
- Replace the PDFs in `knowledge_base/` with your organisation's documentation.
- Update the routing prompt inside `route_query_node` to reflect your source taxonomy.
- Swap the Groq models for another provider if preferred (OpenAI, Anthropic, etc.).
- Extend the LangGraph workflow with additional nodes (validation, summarisation, monitoring hooks).

## Suggested next steps
1. Deploy the app to Streamlit Community Cloud or Hugging Face Spaces.
2. Integrate tracing/observability (LangSmith, Weights & Biases).
3. Add automated tests for critical prompts and routing logic.
4. Share implementation lessons via blog posts, talks, or workshops.

Contributions and issues are more than welcome!
