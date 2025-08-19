# 3_rag_text_splitting.py

import sys
from logging import Logger
from pathlib import Path
from typing import List, Optional

# Import langchain modules
from langchain.text_splitter import CharacterTextSplitter
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


# Create and persist vector store
def create_and_persist_vector_store(
    documents: List[Document],
    embeddings: OllamaEmbeddings,
    db_dir: Path,
    store_name: str,
) -> Chroma:
    persistent_directory: Path = db_dir / store_name
    if not Path.exists(self=persistent_directory):
        logger.info(msg=f"Creating vector store '{store_name}'...")
        try:
            db: Chroma = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=str(object=persistent_directory),
            )
            logger.info(msg=f"Vector store '{store_name}' created successfully.")
        except Exception as e:
            logger.error(
                msg=f"Unexpected error initializing vector store!: {str(object=e)}"
            )
    else:
        logger.info(
            msg=f"Vector store '{store_name}' already exists. No need to initialize."
        )
    return db


# Character text splitter
def character_text_splitter(
    documents: list[Document], embeddings: OllamaEmbeddings, db_dir: Path
) -> Optional[Chroma]:
    logger.info(msg="\n--- Character Text Splitter ---")
    char_txt_splitter: CharacterTextSplitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    char_docs: List[Document] = char_txt_splitter.split_documents(documents=documents)
    create_and_persist_vector_store(
        documents=char_docs,
        embeddings=embeddings,
        db_dir=db_dir,
        store_name="chroma_db_character",
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

    # Character text splitter
    character_text_splitter(documents=documents, embeddings=embeddings, db_dir=db_dir)


# Main entry point
if __name__ == "__main__":
    main()
