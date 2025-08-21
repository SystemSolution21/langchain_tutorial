# 3_rag_text_splitting.py
"""
This script implements a RAG (Retrieval-Augmented Generation) system using LangChain.
It processes text documents, splits them into chunks, creates embeddings, and stores them
in a Chroma vector database for efficient retrieval."""

# Import standard libraries
import sys
from logging import Logger
from pathlib import Path
from typing import List

# Import environment variables
from dotenv import load_dotenv

# Import langchain modules
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Import logger
from utils.logger import RAGLogger

# Load environment variables
load_dotenv()


# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)

# Log application startup
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG Multi Model Embeddings Application")
logger.info(msg="=" * 50)

# Define directories and paths
current_dir: Path = Path(__file__).parent.resolve()
books_dir: Path = current_dir / "books"
file_path: Path = books_dir / "odyssey.txt"
db_dir: Path = current_dir / "db"

# Log directories
logger.info(msg=f"Books Directory: {books_dir}")
logger.info(msg=f"Database Directory: {db_dir}")

# Check db directory exists
try:
    db_dir.mkdir(exist_ok=True)
    logger.info(msg="Database directory created/verified successfully")
except Exception as e:
    logger.error(msg=f"Error creating database directory: {str(object=e)}")
    sys.exit(1)

# Load documents and split into chunks
try:
    # Check file exists
    if not Path.exists(self=file_path):
        logger.error(msg=f"The file {file_path} does not exist")
        sys.exit(1)

    # log file path
    logger.info(msg=f"File Path: {file_path}")

    # Load text content from file
    text_loader: TextLoader = TextLoader(file_path=file_path, encoding="utf-8")
    documents: List[Document] = text_loader.load()

    # Character-based splitting
    char_txt_splitter: CharacterTextSplitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )
    char_docs: List[Document] = char_txt_splitter.split_documents(documents=documents)
except Exception as e:
    logger.error(msg=f"Failed to load documents: {str(object=e)}")
    sys.exit(1)

# Define embeddings models
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# Create vector store
def create_vector_store(
    documents: List[Document],
    embeddings: HuggingFaceEmbeddings | OpenAIEmbeddings,
    store_name: str,
) -> None:
    persistent_directory: Path = db_dir / store_name
    if Path.exists(self=persistent_directory):
        logger.info(
            msg=f"Vector store '{store_name}' already exists. No need to initialize."
        )
        return

    logger.info(msg=f"Creating vector store '{store_name}'...")
    try:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(object=persistent_directory),
        )
        logger.info(msg=f"Vector store '{store_name}' created successfully.")
    except Exception as e:
        logger.error(
            msg=f"Unexpected error initializing vector store!: {str(object=e)}"
        )


# Query vector store
def query_vector_store(
    store_name: str,
    query: str,
    embeddings_function: HuggingFaceEmbeddings | OpenAIEmbeddings,
) -> None:
    """Query the vector store for relevant documents.

    Args:
        store_name: Name of the vector store
        query: Query string
        embeddings_function: Embeddings function to use for querying
    Returns:
        None
    """
    persistent_directory: Path = db_dir / store_name
    if not Path.exists(self=persistent_directory):
        logger.info(
            msg=f"Vector store '{store_name}' does not exist. No need to query."
        )
        return

    try:
        logger.info(msg=f"Querying vector store '{store_name}'...")
        db: Chroma = Chroma(
            persist_directory=str(object=persistent_directory),
            embedding_function=embeddings_function,
        )

        retriever: VectorStoreRetriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )

        relevant_docs: list[Document] = retriever.invoke(input=query)
        if relevant_docs:
            logger.info(
                msg=f"For vector store '{store_name}', found {len(relevant_docs)} relevant documents"
            )
        else:
            logger.info(msg="No relevant documents found")
            return

        for i, doc in enumerate(relevant_docs, start=1):
            print(f"\n--- Relevant Document {i} ---")
            print(f"Document:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    except Exception as e:
        logger.error(msg=f"Unexpected error querying vector store: {str(object=e)}")


def main() -> None:
    # Create vector stores
    # OpenAI Embeddings model for general-purpose embeddings with high accuracy.
    create_vector_store(
        documents=char_docs,
        embeddings=openai_embeddings,
        store_name="chroma_db_openai",
    )

    # HuggingFace Embeddings model for leveraging a wide variety of different tasks.
    create_vector_store(
        documents=char_docs,
        embeddings=huggingface_embeddings,
        store_name="chroma_db_huggingface",
    )

    # User query
    query: str = "who is odysseus wife?"

    # Query vector store
    # OpenAI
    query_vector_store(
        store_name="chroma_db_openai",
        query=query,
        embeddings_function=openai_embeddings,
    )
    # HuggingFace
    query_vector_store(
        store_name="chroma_db_huggingface",
        query=query,
        embeddings_function=huggingface_embeddings,
    )


# Main entry point
if __name__ == "__main__":
    main()
