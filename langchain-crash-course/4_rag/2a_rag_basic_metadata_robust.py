"""
Robust version of RAG implementation with metadata handling.
This version includes better error handling, type safety, and code organization.
"""

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
from utils.logger import RAGLogger

# Set up logger
logger: Logger = RAGLogger.get_logger(module_name=__name__)

# Log application startup
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG Metadata Application")
logger.info(msg="=" * 50)


def load_documents(books_dir: Path) -> List[Document]:
    """
    Load documents from text files with error handling for each file.

    Args:
        books_dir: Directory containing the text files

    Returns:
        List[Document]: List of successfully loaded documents
    """
    documents: List[Document] = []
    failed_files: List[Tuple[Path, str]] = []

    try:
        books_files: List[Path] = list(Path.glob(self=books_dir, pattern="*.txt"))
        if not books_files:
            logger.warning(msg=f"No .txt files found in {books_dir}")
            return documents

        for book_file in books_files:
            file_path: Path = books_dir / book_file
            try:
                loader: TextLoader = TextLoader(file_path=file_path, encoding="utf-8")
                docs: List[Document] = loader.load()
                for doc in docs:
                    doc.metadata = {"source": str(object=book_file.name)}
                    documents.append(doc)
                logger.info(msg=f"Successfully loaded: {file_path}")
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


def create_text_chunks(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Split documents into chunks with error handling.

    Args:
        documents: List of documents to split
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        List[Document]: List of document chunks
    """
    try:
        text_splitter: CharacterTextSplitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n"
        )
        chunk_doc: List[Document] = text_splitter.split_documents(documents=documents)
        logger.info(msg=f"Successfully created {len(chunk_doc)} document chunks")
        return chunk_doc
    except Exception as e:
        logger.error(msg=f"Error splitting documents: {str(object=e)}")
        return []


def initialize_vector_store(
    books_dir: Path,
    persistent_directory: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Optional[Chroma]:
    """
    Initialize the vector store with document chunks and embeddings.

    Args:
        books_dir: Directory containing the text files
        persistent_directory: Directory to store the vector database
        chunk_size: Size of text chunks (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)

    Returns:
        Optional[Chroma]: Initialized vector store or None if initialization fails
    """
    try:
        if Path.exists(self=persistent_directory):
            logger.info(msg="Vector store already exists. No need to initialize.")
            return None

        logger.info(msg="Initializing new vector store...")

        if not Path.exists(self=books_dir):
            logger.error(msg=f"The directory {books_dir} does not exist")
            return None

        # Load documents
        documents: List[Document] = load_documents(books_dir=books_dir)
        if not documents:
            logger.warning(msg="No documents were successfully loaded")
            return None

        # Create chunks
        chunk_doc: List[Document] = create_text_chunks(
            documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not chunk_doc:
            logger.warning(msg="No document chunks were created")
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
    """Main function to set up and initialize the vector store."""
    try:
        # Define directories
        current_dir: Path = Path(__file__).parent.resolve()
        books_dir: Path = current_dir / "books"
        db_dir: Path = current_dir / "db"
        persistent_directory: Path = db_dir / "chroma_db_with_metadata"

        logger.info(msg=f"Books Directory: {books_dir}")
        logger.info(msg=f"Persistent Directory: {persistent_directory}")

        # Create db directory if it doesn't exist
        try:
            db_dir.mkdir(exist_ok=True)
            logger.info(msg="Database directory created/verified successfully")
        except Exception as e:
            logger.error(msg=f"Error creating database directory: {str(object=e)}")
            return

        # Initialize vector store
        db: Chroma | None = initialize_vector_store(
            books_dir=books_dir,
            persistent_directory=persistent_directory,
        )

        # Check initialization result
        if db is None and Path.exists(self=persistent_directory):
            logger.info(msg="Using existing vector store - no initialization needed")
        elif db is not None:
            logger.info(msg="Vector store initialization completed successfully")
        else:
            logger.error(
                msg="Vector store initialization failed - system may not function correctly"
            )

    except Exception as e:
        logger.error(
            msg=f"Unexpected error while set up and initializing vector store: {str(object=e)}"
        )


if __name__ == "__main__":
    main()
