from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

# Define persistent directory
current_dir: Path = Path(__file__).parent.resolve()
persistent_directory: Path = current_dir / "db" / "chroma_db_with_metadata"

# Define embeddings model
embeddings: OllamaEmbeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

# Load the Chroma vector store from the persistent directory with the embeddings model
db: Chroma = Chroma(
    persist_directory=str(persistent_directory), embedding_function=embeddings
)

# Define the user's question
query = "How did Juliet die?"

# Retrieve relevant documents based on the query
retriever: VectorStoreRetriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)
relevant_docs: list[Document] = retriever.invoke(input=query)

# Display the retrieved documents with MetaData
for i, doc in enumerate(relevant_docs, start=1):
    print(f"\n--- Relevant Document {i} ---")
    print(f"Content:\n{doc.page_content}\n")
    print(f"Metadata:\n{doc.metadata}")
