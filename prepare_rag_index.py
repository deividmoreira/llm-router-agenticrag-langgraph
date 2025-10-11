# LLM Router Dev Portal - RAG index preparation script
# Builds the local FAISS index consumed by the Streamlit application

# Provides filesystem helpers such as directory checks
import os

# Loads PDFs from the knowledge base directory
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Splits documents into chunks suitable for RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Builds and persists FAISS vector stores
from langchain_community.vectorstores import FAISS

# Embedding model used to convert text chunks into vectors
from langchain_community.embeddings import FastEmbedEmbeddings

# Directory containing the PDFs for the knowledge base
DATA_DIR = "knowledge_base"

# Path where the FAISS index will be stored locally
VECTORSTORE_PATH = "faiss_index"

# Ensure the knowledge base directory exists; ask the user to add PDFs if it was missing
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Directory '{DATA_DIR}' created. Add your knowledge base PDFs here.")
    exit()

# Main entry point to build the vector store from the PDFs
def build_vectordb():
    
    # Start loading the documents
    print(f"Loading PDFs from: {DATA_DIR}")
    try:
        
        # Load every PDF in the directory (recursive)
        pdf_loader = PyPDFDirectoryLoader(DATA_DIR, recursive = True)
        documents = pdf_loader.load()
        
        # No documents available aborts the process
        if not documents:
            print(f"No PDF documents found in '{DATA_DIR}'. Please add source material.")
            return False
            
        print(f"Loaded {len(documents)} PDF pages/documents.")

    except Exception as e:
        
        # Loading issues are surfaced here
        print(f"Error while loading PDFs: {e}")
        return False

    # Chunking the documents before vectorisation
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 150)
    
    # Creates smaller overlapping chunks for the embedding model
    docs_split = text_splitter.split_documents(documents)
    print(f"Documents split into {len(docs_split)} chunks.")

    # Initialise the embedding model
    print("Initialising FastEmbed embeddings...")
    embedding_model = FastEmbedEmbeddings(model_name = "BAAI/bge-small-en-v1.5")

    # Create the FAISS index and persist it
    print("Building FAISS index...")
    try:
        vector_store = FAISS.from_documents(docs_split, embedding_model)
        print("FAISS index created in memory.")
        vector_store.save_local(VECTORSTORE_PATH)
        print(f"FAISS index saved to: {VECTORSTORE_PATH}")
        return True
    except Exception as e:
        # Handle vector store build and persistence issues
        print(f"Error while creating or saving the FAISS index: {e}")
        return False

# Script entry point
if __name__ == "__main__":
    
    # Kick off the process
    print("\nStarting the local RAG index preparation...")
    
    # Run the vector store build and display the outcome
    if build_vectordb():
        print("FAISS index created successfully!")
        print(f"The vector store is available in '{VECTORSTORE_PATH}'.")
        print(f"Keep your PDFs up to date in '{DATA_DIR}'.\n")
    else:
        print("\nIndex creation failed. Review the errors above.")
