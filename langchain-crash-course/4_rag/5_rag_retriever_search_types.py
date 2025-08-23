# 5_rag_retriever_search_types.py
"""
This script implements a RAG (Retrieval-Augmented Generation) system using LangChain.
It queries a vector store for relevant documents using different search types and prints the results.
    1. Similarity Search
    2. Max Marginal Relevance (MMR)
    3. Similarity Score Threshold
"""

# Import standard libraries
from logging import Logger
from pathlib import Path

# Import environment variables
from dotenv import load_dotenv

# Import langchain modules
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# Import custom logger
from utils.logger import RAGLogger

# Load environment variables
load_dotenv()


# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)

# Log application startup
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG Retriever Search Types Embeddings Application")
logger.info(msg="=" * 50)

# Define directories and paths
current_dir: Path = Path(__file__).parent.resolve()
db_dir: Path = current_dir / "db"
persistent_directory: Path = db_dir / "chroma_db_with_metadata"

# Define embeddings models
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# Query vector store
def query_vector_store(
    store_name: str,
    query: str,
    embeddings_function: HuggingFaceEmbeddings,
    search_type: str,
    search_kwargs: dict,
) -> None:
    """Query the vector store for relevant documents.

    Args:
        store_name: Name of the vector store
        query: Query string
        embeddings_function: Embeddings function to use for querying
    Returns:
        None
    """
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
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

        relevant_docs: list[Document] = retriever.invoke(input=query)
        if relevant_docs:
            logger.info(
                msg=f"Relevant documents retrieved using search type '{search_type}' and search kwargs '{search_kwargs}' from vector store '{store_name}'"
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
    # User query
    query: str = "How did Juliet die?"

    # 1. Similarity Search
    # This method retrieves documents based on vector similarity.
    # It finds the most similar documents to the query vector based on cosine similarity.
    # Use to retrieve the top k most similar documents.
    print("\n--- Using Similarity Search ---")
    query_vector_store(
        store_name="chroma_db_with_metadata",
        query=query,
        embeddings_function=huggingface_embeddings,
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # 2. Max Marginal Relevance (MMR)
    # This method balances between selecting documents that are relevant to the query and diverse among themselves.
    # 'fetch_k' specifies the number of documents to initially fetch based on similarity.
    # 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
    # Use to avoid redundancy and retrieve diverse yet relevant documents.
    # Note: Relevance measures how closely documents match the query.
    # Note: Diversity ensures that the retrieved documents are not too similar to each other,
    #       providing a broader range of information.
    print("\n--- Using Max Marginal Relevance (MMR) ---")
    query_vector_store(
        store_name="chroma_db_with_metadata",
        query=query,
        embeddings_function=huggingface_embeddings,
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
    )

    # 3. Similarity Score Threshold
    # This method retrieves documents that exceed a certain similarity score threshold.
    # 'score_threshold' sets the minimum similarity score a document must have to be considered relevant.
    # Use to ensure that only highly relevant documents are retrieved, filtering out less relevant ones.
    print("\n--- Using Similarity Score Threshold ---")
    query_vector_store(
        store_name="chroma_db_with_metadata",
        query=query,
        embeddings_function=huggingface_embeddings,
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.8},
    )


# Main entry point
if __name__ == "__main__":
    main()
