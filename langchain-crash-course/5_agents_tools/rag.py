"""
This script demonstrates a robust way to create a RAG (Retrieval-Augmented Generation)
system with metadata and persistent storage. It includes the following steps:
1.  Loading documents from a specified directory with error handling.
2.  Splitting the loaded documents into smaller chunks.
3.  Creating embeddings for the chunks using Ollama.
4.  Initializing and persisting a Chroma vector store with the chunks and their metadata.
The script is designed to be idempotent, checking if the vector store already exists
before attempting to create it.
"""

# Import necessary libraries
from logging import Logger
from pathlib import Path
from typing import List, Optional, Tuple

# Import langchain modules
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_ollama.embeddings import OllamaEmbeddings

# Import custom logger
from utils.logger import ReActAgentLogger

# Module path
module_path: Path = Path(__file__).resolve()

# Set up logger
logger: Logger = ReActAgentLogger.get_logger(module_name=module_path.name)

# Define books and database directories paths
current_dir: Path = Path(__file__).parent.resolve()
books_dir: Path = current_dir / "books"
db_dir: Path = current_dir / "db"
store_name: str = "chroma_db_with_metadata"
persistent_directory: Path = db_dir / store_name


# Load documents
def load_documents(books_dir: Path) -> List[Document]:
    """Loads documents from text files in a directory, adding metadata.

    This function iterates through all '.txt' files in the specified directory,
    loads them as documents, and adds the filename as metadata to each document.
    It includes error handling for file loading and directory access.

    Args:
        books_dir (Path): The path to the directory containing the text files.

    Returns:
        List[Document]: A list of loaded documents, each with 'source' metadata.
    """
    documents: List[Document] = []
    failed_files: List[Tuple[Path, str]] = []

    try:
        books_files: List[Path] = list(books_dir.glob(pattern="*.txt"))
        if not books_files:
            logger.warning(msg=f"No '.txt' files found in {books_dir}")
            return documents

        for file_path in books_files:
            try:
                text_loader: TextLoader = TextLoader(
                    file_path=str(object=file_path), encoding="utf-8"
                )
                docs: List[Document] = text_loader.load()
                for doc in docs:
                    doc.metadata = {"source": file_path.name}
                    documents.append(doc)
                logger.info(msg=f"Successfully loaded document: {file_path}")
            except Exception as e:
                error_msg: str = f"Error loading {file_path}: {str(object=e)}"
                failed_files.append((file_path, str(object=e)))
                logger.error(msg=error_msg)
                continue

    except Exception as e:
        logger.error(msg=f"Error accessing directory {books_dir}: {str(object=e)}")

    # Log summary of failed files
    if failed_files:
        logger.warning(msg="Failed to load the following files:")
        for file_path, error in failed_files:
            logger.warning(msg=f"- {file_path}: {error}")

    return documents


# Create text chunks
def create_text_chunks(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Splits a list of documents into smaller chunks.

    Uses CharacterTextSplitter to divide the documents based on the specified
    chunk size and overlap.

    Args:
        documents (List[Document]): The list of documents to be split.
        chunk_size (int): The maximum number of characters in each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Document]: A list of the resulting document chunks. Returns an
        empty list if an error occurs.
    """
    try:
        text_splitter: CharacterTextSplitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n"
        )
        chunk_doc: List[Document] = text_splitter.split_documents(documents=documents)
        logger.info(msg=f"Successfully created {len(chunk_doc)} document chunks")
        return chunk_doc
    except Exception as e:
        logger.error(msg=f"Error splitting documents: {str(e)}")
        return []


def initialize_vector_store(
    books_dir: Path,
    persistent_directory: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Optional[Chroma]:
    """Initializes and persists a Chroma vector store from documents.

    This function orchestrates the process of loading documents, splitting them
    into chunks, creating embeddings, and storing them in a Chroma vector store.
    It checks if a vector store already exists at the persistent directory and
    skips initialization if it does.

    Args:
        books_dir (Path): The directory containing the source text files.
        persistent_directory (Path): The directory where the Chroma database
            will be persisted.
        chunk_size (int, optional): The size of text chunks. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between text chunks.
            Defaults to 200.

    Returns:
        Optional[Chroma]: The created Chroma vector store instance if a new
        store was initialized, otherwise None.
    """

    # Log application startup
    logger.info(msg="======== Starting Create RAG With Metadata Application ========")
    logger.info(msg=f"Books Directory: {books_dir}")
    logger.info(msg=f"Persistent Directory: {persistent_directory}")

    try:
        # Check vector store existence
        if persistent_directory.exists():
            logger.info(msg="Vector store already exists. No need to initialize.")
            return None

        logger.info(msg="Initializing new vector store...")

        # Check books directory existence
        if not books_dir.exists():
            logger.error(
                msg=f"The directory '{books_dir}' does not exist. Please check the path."
            )
            return None

        # Load documents
        documents: List[Document] = load_documents(books_dir=books_dir)
        if not documents:
            logger.warning(
                msg=f"No documents were successfully loaded from directory '{books_dir}'"
            )
            return None

        # Create chunks
        chunk_doc: List[Document] = create_text_chunks(
            documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not chunk_doc:
            logger.warning(msg="No document chunks were created.")
            return None

        # Create embeddings
        logger.info(msg="Creating embeddings...")
        try:
            embeddings: OllamaEmbeddings = OllamaEmbeddings(
                model="nomic-embed-text:latest"
            )
            logger.info(msg="Embeddings created successfully")
        except Exception as e:
            logger.error(msg=f"Error creating embeddings: {str(object=e)}")
            return None

        # Initialize vector store
        logger.info(msg="Initializing Chroma vector store...")
        try:
            db: Chroma = Chroma.from_documents(
                documents=chunk_doc,
                embedding=embeddings,
                persist_directory=str(object=persistent_directory),
            )
            logger.info(msg="Vector store initialized successfully")
            return db
        except Exception as e:
            logger.error(msg=f"Error initializing vector store: {str(object=e)}")
            return None

    except Exception as e:
        logger.error(
            msg=f"Unexpected error during vector store initialization: {str(object=e)}"
        )
        return None


def main() -> None:
    """Main function to run the RAG vector store initialization process.

    This function calls the necessary functions to set up the vector store.
    It handles the main logic flow and logs the outcome of the initialization.
    """

    try:
        # Initialize vector store
        db: Optional[Chroma] = initialize_vector_store(
            books_dir=books_dir,
            persistent_directory=persistent_directory,
        )

        # Check initialization result
        if db is None and persistent_directory.exists():
            logger.info("Vector store already exists. Standalone execution complete.")
        elif db is None:
            logger.error(
                "Vector store initialization failed - System may not function correctly!"
            )

    except Exception as e:
        logger.error(f"An unexpected error occurred in the main execution block: {e}")


if __name__ == "__main__":
    main()
