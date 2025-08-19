"""
This script implements a basic RAG (Retrieval-Augmented Generation) system using LangChain.
It processes text documents, splits them into chunks, creates embeddings, and stores them
in a Chroma vector database for efficient retrieval.
"""

from pathlib import Path
from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_ollama import OllamaEmbeddings

# Set up paths
current_dir: Path = Path(__file__).parent.resolve()
file_path: Path = current_dir / "books" / "odyssey.txt"
persistent_directory: Path = current_dir / "db" / "chroma_db"

# Display current working directory, file path and persistent directory for debugging
print(
    "\n--- Display current working directory, file path and persistent directory for debugging ---"
)
print(f"Current working directory: {current_dir}")
print(f"File path: {file_path}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not Path.exists(self=persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not Path.exists(self=file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader: TextLoader = TextLoader(file_path=file_path, encoding="utf-8")
    documents: List[Document] = loader.load()

    # Split the documents into chunks
    text_splitter: CharacterTextSplitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, separator="\n"
    )
    docs: List[Document] = text_splitter.split_documents(documents=documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings: OllamaEmbeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    print("--- Embeddings created successfully ---")

    # # Create embeddings
    # print("\n--- Creating embeddings ---")
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    # )
    # print("--- Embeddings created successfully ---")

    # Initialize the Chroma vector store
    print("\n--- Chroma vector store initialization ---")
    db: Chroma = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(object=persistent_directory),
    )
    print("--- Chroma vector store initialized successfully ---")

else:
    print("Vector store already exists. No need to initialize.")
