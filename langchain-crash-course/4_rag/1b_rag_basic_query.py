# 1b_rag_basic_query.py
"""
This script implements a RAG (Retrieval-Augmented Generation) system using LangChain.
It queries a vector store for relevant documents based on a user's query and prints the results.
"""

# Import necessary libraries
from pathlib import Path

# Import langchain modules
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

# Set up paths for the persistent directory
current_dir: Path = Path(__file__).parent.resolve()
persistent_directory: Path = current_dir / "db" / "chroma_db"

# Define embeddings model
embeddings: OllamaEmbeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Check if the Chroma vector store already exists
if not Path.exists(self=persistent_directory):
    raise FileNotFoundError(
        f"The directory {persistent_directory} does not exist. Please check the path."
    )

# Load the Chroma vector store from the persistent directory with the embeddings model
db: Chroma = Chroma(
    persist_directory=str(object=persistent_directory), embedding_function=embeddings
)

# Define the user's question
query = "Who is Odysseus' wife?"

# Retrieve relevant documents based on the query
retriever: VectorStoreRetriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)
relevant_docs: list[Document] = retriever.invoke(input=query)

# Display the retrieved documents with MetaData
for i, doc in enumerate(iterable=relevant_docs, start=1):
    print("\n--- Relevant Documents ---")
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
