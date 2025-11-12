"""
This script demonstrates a robust way to create a RAG (Retrieval-Augmented Generation)
system with PDF documents, metadata and persistent storage. It includes the following steps:
1.  Loading PDF documents from a specified directory with error handling.
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_ollama.embeddings import OllamaEmbeddings

# Import custom logger
from util.logger import ReActAgentLogger

# Module path
module_path: Path = Path(__file__).resolve()

# Set up logger
logger: Logger = ReActAgentLogger.get_logger(module_name=module_path.name)

# Define PDFs and database directories paths
current_dir: Path = Path(__file__).parent.resolve()
pdfs_dir: Path = current_dir / "pdfs"
db_dir: Path = current_dir / "db"
store_name: str = "chroma_db_pdf_metadata"
persistent_directory: Path = db_dir / store_name


# Load PDF documents
def load_pdf_documents(pdfs_dir: Path) -> List[Document]:
    """Loads documents from PDF files in a directory, adding metadata.

    This function iterates through all '.pdf' files in the specified directory,
    loads them as documents, and adds the filename as metadata to each document.
    It includes error handling for file loading and directory access.

    Args:
        pdfs_dir (Path): The path to the directory containing the PDF files.

    Returns:
        List[Document]: A list of loaded documents with metadata.
    """
    documents: List[Document] = []
    failed_files: List[Tuple[Path, str]] = []

    try:
        pdf_files: List[Path] = list(pdfs_dir.glob(pattern="*.pdf"))
        if not pdf_files:
            logger.warning(msg=f"No '.pdf' files found in {pdfs_dir}")
            return documents

        for file_path in pdf_files:
            try:
                pdf_loader: PyPDFLoader = PyPDFLoader(file_path=str(file_path))
                docs: List[Document] = pdf_loader.load()
                for doc in docs:
                    doc.metadata = {"source": file_path.name}
                    documents.append(doc)
                logger.info(msg=f"Successfully loaded PDF document: {file_path}")
            except Exception as e:
                error_msg: str = f"Error loading {file_path}: {str(e)}"
                failed_files.append((file_path, str(e)))
                logger.error(msg=error_msg)
                continue

    except Exception as e:
        logger.error(msg=f"Error accessing directory {pdfs_dir}: {str(e)}")

    # Log summary of failed files
    if failed_files:
        logger.warning(msg="Failed to load the following PDF files:")
        for file_path, error in failed_files:
            logger.warning(msg=f"- {file_path}: {error}")

    return documents


# Create text chunks
def create_text_chunks(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Splits a list of documents into smaller chunks.

    Uses RecursiveCharacterTextSplitter to divide the documents based on the specified
    chunk size and overlap. This splitter is better for PDFs as it tries to split
    on natural boundaries.

    Args:
        documents (List[Document]): The list of documents to be split.
        chunk_size (int): The maximum number of characters in each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Document]: A list of document chunks.
    """
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: List[Document] = text_splitter.split_documents(documents=documents)
    logger.info(msg=f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


# Create embeddings
def create_embeddings() -> OllamaEmbeddings:
    """Creates and returns an OllamaEmbeddings instance.

    Returns:
        OllamaEmbeddings: An instance of OllamaEmbeddings for creating vector embeddings.
    """
    embeddings: OllamaEmbeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    logger.info(msg="Embeddings created successfully")
    return embeddings


# Create vector store
def create_vector_store(
    chunks: List[Document], embeddings: OllamaEmbeddings, persistent_directory: Path
) -> Optional[Chroma]:
    """Creates and persists a Chroma vector store from document chunks.

    Args:
        chunks (List[Document]): The list of document chunks to store.
        embeddings (OllamaEmbeddings): The embeddings model to use.
        persistent_directory (Path): The directory to persist the vector store.

    Returns:
        Optional[Chroma]: The created Chroma vector store, or None if creation failed.
    """
    try:
        logger.info(msg="Initializing Chroma vector store...")
        db: Chroma = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persistent_directory),
        )
        logger.info(msg="Vector store initialized successfully")
        return db
    except Exception as e:
        logger.error(msg=f"Error creating vector store: {str(e)}")
        return None


# Initialize vector store
def initialize_pdf_vector_store(
    pdfs_dir: Path, persistent_directory: Path
) -> Optional[Chroma]:
    """Initializes a vector store from PDF documents in a directory.

    This function orchestrates the entire process of loading PDF documents,
    splitting them into chunks, creating embeddings, and storing them in
    a persistent Chroma vector store.

    Args:
        pdfs_dir (Path): The path to the directory containing PDF files.
        persistent_directory (Path): The path where the vector store will be persisted.

    Returns:
        Optional[Chroma]: The initialized vector store, or None if initialization failed.
    """
    logger.info(
        msg="======== Starting Create PDF RAG With Metadata Application ========"
    )
    logger.info(msg=f"PDFs Directory: {pdfs_dir}")
    logger.info(msg=f"Persistent Directory: {persistent_directory}")

    # Check if vector store already exists
    if persistent_directory.exists():
        logger.info(msg="Vector store already exists. No need to initialize.")
        return None

    # Check if PDFs directory exists
    if not pdfs_dir.exists():
        logger.error(
            msg=f"The directory '{pdfs_dir}' does not exist. Please check the path."
        )
        return None

    logger.info(msg="Initializing new vector store...")

    # Load PDF documents
    documents: List[Document] = load_pdf_documents(pdfs_dir=pdfs_dir)
    if not documents:
        logger.error(msg="No PDF documents loaded. Cannot create vector store.")
        return None

    # Create text chunks
    chunks: List[Document] = create_text_chunks(
        documents=documents, chunk_size=1000, chunk_overlap=200
    )

    # Create embeddings
    embeddings: OllamaEmbeddings = create_embeddings()

    # Create and persist vector store
    db: Optional[Chroma] = create_vector_store(
        chunks=chunks, embeddings=embeddings, persistent_directory=persistent_directory
    )

    return db


# Main function for standalone execution
def main() -> None:
    """Main function for standalone execution of the PDF RAG initialization."""
    db: Optional[Chroma] = initialize_pdf_vector_store(
        pdfs_dir=pdfs_dir, persistent_directory=persistent_directory
    )

    if db:
        logger.info(msg="PDF Vector store initialization complete.")
    else:
        logger.info(
            msg="PDF Vector store already exists. Standalone execution complete."
        )


# Main entry point
if __name__ == "__main__":
    main()
