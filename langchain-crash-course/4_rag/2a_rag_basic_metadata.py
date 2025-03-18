from pathlib import Path
from typing import Any, List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Define text files directory and persistent directory
current_dir: Path = Path(__file__).parent.resolve()
books_dir: Path = current_dir / "books"
db_dir: Path = current_dir / "db"
persistent_directory: Path = db_dir / "chroma_db_with_metadata"

print(f"Books Directory: {books_dir}")
print(f"Persistent Directory: {persistent_directory}")

# Documents loading, Text splitting, Embedding text and Create chroma vector store
# Check chroma vector store exists
if not Path.exists(self=persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not Path.exists(self=books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all the text files in the books directory
    books_files: List[Path] = [
        file for file in Path.glob(self=books_dir, pattern="*.txt")
    ]

    # Load text content from each file and add with its metadata
    documents: List[Any] = []
    for book_file in books_files:
        file_path: Path = books_dir / book_file
        loader: TextLoader = TextLoader(file_path=file_path, encoding="utf-8")
        docs: List[Document] = loader.load()
        for doc in docs:
            doc.metadata = {"source": str(object=book_file.name)}
            documents.append(doc)

    # Split documents into chunks with optimal overlap
    text_splitter: CharacterTextSplitter = CharacterTextSplitter(
        chunk_size=1000,  # Base chunk size
        chunk_overlap=200,  # 20% overlap for good context preservation
        separator="\n",  # Natural text boundary
    )
    chunk_doc: List[Document] = text_splitter.split_documents(documents=documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(chunk_doc)}")

    # Create text embeddings
    print("\n--- Creating embeddings ---")
    embeddings: OllamaEmbeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    print("\n--- Finished creating embeddings ---")

    # Initialize the Chroma vector store
    print("\n--- Chroma vector store initialization ---")
    db: Chroma = Chroma.from_documents(
        documents=chunk_doc,
        embedding=embeddings,
        persist_directory=str(object=persistent_directory),
    )
    print("--- Chroma vector store initialized successfully ---")

else:
    print("Vector store already exists. No need to initialize.")
