# Getting Started

End-to-end guide to spin up the LLM Router Dev Portal on your machine.

## 1. Create a virtual environment
```bash
conda create --name llmrouter python=3.12
conda activate llmrouter
```
> Prefer `venv`? Use `python -m venv .venv && source .venv/bin/activate`.

## 2. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Prepare the knowledge base
1. Add your PDFs to the `knowledge_base/` directory.
2. Build the local FAISS index:
   ```bash
   python prepare_rag_index.py
   ```

## 4. Configure secrets
Create a `.env` file (or export variables manually) with:
```
GROQ_API_KEY=your_key_here
```

## 5. Run the Streamlit portal
```bash
streamlit run llm_router_portal_app.py
```

## 6. Explore the demo
- Submit support-style questions.
- Track in the sidebar which source (RAG or Web) was selected.
- Inspect the terminal logs to see routing decisions.

## 7. Clean up (optional)
```bash
conda deactivate
conda remove --name llmrouter --all
```

Happy building!
