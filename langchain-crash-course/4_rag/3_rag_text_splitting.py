# 3_rag_text_splitting.py

import sys
from logging import Logger
from pathlib import Path
from typing import List

# Import langchain modules
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_ollama.embeddings import OllamaEmbeddings

# Import logger
from utils.logger import RAGLogger

# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)

# Start logging
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG Text Splitting Application")
logger.info(msg="=" * 50)


# Create vector store
def create_vector_store(
    documents: List[Document],
    embeddings: OllamaEmbeddings,
    db_dir: Path,
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


# Character-base splitting
def character_splitter(
    documents: list[Document], embeddings: OllamaEmbeddings, db_dir: Path
) -> None:
    """
    Splits text into chunks based on a specified number of characters.

    Args:
        documents: List of documents to split
        embeddings: Embeddings model
        db_dir: Directory to store the vector database
        store_name: Name of the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    Returns:
        None
    Usages:
        Useful for consistent chunk sizes regardless of content structure.
    """
    logger.info(msg="---- Using character-based splitting ----")
    char_txt_splitter: CharacterTextSplitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    char_docs: List[Document] = char_txt_splitter.split_documents(documents=documents)
    create_vector_store(
        documents=char_docs,
        embeddings=embeddings,
        db_dir=db_dir,
        store_name="chroma_db_character",
    )


# Sentence-based splitting
def sentence_splitter(
    documents: list[Document], embeddings: OllamaEmbeddings, db_dir: Path
) -> None:
    """
    Splits text into chunks based on sentence boundaries.

    Args:
        documents: List of documents to split
        embeddings: Embeddings model
        db_dir: Directory to store the vector database
        store_name: Name of the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    Returns:
        None
    Usages:
        Ideal for maintaining semantic coherence within chunks.
    """
    logger.info(msg="---- Using sentence-based splitting ----")
    sent_splitter: SentenceTransformersTokenTextSplitter = (
        SentenceTransformersTokenTextSplitter(chunk_size=1000)
    )
    sent_docs: List[Document] = sent_splitter.split_documents(documents=documents)
    create_vector_store(
        documents=sent_docs,
        embeddings=embeddings,
        db_dir=db_dir,
        store_name="chroma_db_sentence",
    )


# Token-based splitting
def token_splitter(
    documents: list[Document], embeddings: OllamaEmbeddings, db_dir: Path
) -> None:
    """Splits text into chunks based on tokens (words or sub-words), using tokenizers like GPT-2.

    Args:
        documents: List of documents to split
        embeddings: Embeddings model
        db_dir: Directory to store the vector database
        store_name: Name of the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    Returns:
        None
    Usages:
        Useful for transformer models with strict token limits.
    """
    logger.info(msg="---- Using token-based splitting ----")
    token_splitter: TokenTextSplitter = TokenTextSplitter(
        chunk_size=512, chunk_overlap=0
    )
    token_docs: List[Document] = token_splitter.split_documents(documents=documents)
    create_vector_store(
        documents=token_docs,
        embeddings=embeddings,
        db_dir=db_dir,
        store_name="chroma_db_token",
    )


# Recursive character-based splitting
def recursive_character_splitter(
    documents: list[Document], embeddings: OllamaEmbeddings, db_dir: Path
) -> None:
    """Split text at natural boundaries (sentences, paragraphs) within character limit.

    Args:
        documents: List of documents to split
        embeddings: Embeddings model
        db_dir: Directory to store the vector database
        store_name: Name of the vector store
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    Returns:
        None
    Usages:
        Balances between maintaining coherence and adhering to character limits.
    """
    logger.info(msg="---- Using recursive character-based splitting ----")
    recursive_char_splitter: RecursiveCharacterTextSplitter = (
        RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    )
    recursive_char_docs: List[Document] = recursive_char_splitter.split_documents(
        documents=documents
    )
    create_vector_store(
        documents=recursive_char_docs,
        embeddings=embeddings,
        db_dir=db_dir,
        store_name="chroma_db_recursive_character",
    )


# Custom text splitting
class CustomTextSplitter(TextSplitter):
    """Custom splitting logic based on specific requirements."""

    def split_text(self, text: str) -> List[str]:
        return text.split(sep="\n\n")  # split by paragraphs


# Custom-based splitter
def custom_splitter(
    documents: list[Document], embeddings: OllamaEmbeddings, db_dir: Path
) -> None:
    """Custom text splitter based on specific requirements.

    Args:
        documents: List of documents to split
        embeddings: Embeddings model
        db_dir: Directory to store the vector database
        store_name: Name of the vector store
    Returns:
        None
    Usages:
        Useful when default splitters don't meet specific requirements.
    """
    logger.info(msg="---- Using custom-based splitting ----")
    custom_splitter: CustomTextSplitter = CustomTextSplitter()
    custom_docs: List[Document] = custom_splitter.split_documents(documents=documents)
    create_vector_store(
        documents=custom_docs,
        embeddings=embeddings,
        db_dir=db_dir,
        store_name="chroma_db_custom",
    )


def main() -> None:
    try:
        # Define directories
        current_dir: Path = Path(__file__).parent.resolve()
        books_dir: Path = current_dir / "books"
        file_path: Path = books_dir / "romeo_and_juliet.txt"
        db_dir: Path = current_dir / "db"

        # logging file path
        logger.info(msg=f"File Path: {file_path}")

        # Check file exists
        if not Path.exists(self=file_path):
            logger.error(msg=f"The file {file_path} does not exist")
            sys.exit(1)

        # Load text content from file
        text_loader: TextLoader = TextLoader(file_path=file_path, encoding="utf-8")
        documents: List[Document] = text_loader.load()

        # Define embeddings model
        embeddings: OllamaEmbeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

    except Exception as e:
        logger.error(msg=f"Unexpected error: {str(object=e)}")

    # # Character-based splitting
    # character_splitter(documents=documents, embeddings=embeddings, db_dir=db_dir)

    # Sentence-based splitting
    sentence_splitter(documents=documents, embeddings=embeddings, db_dir=db_dir)

    # # Token-based splitting
    # token_splitter(documents=documents, embeddings=embeddings, db_dir=db_dir)

    # # Recursive character-based splitting
    # recursive_character_splitter(
    #     documents=documents, embeddings=embeddings, db_dir=db_dir
    # )

    # # Custom-based splitting
    # custom_splitter(documents=documents, embeddings=embeddings, db_dir=db_dir)


# Main entry point
if __name__ == "__main__":
    main()
