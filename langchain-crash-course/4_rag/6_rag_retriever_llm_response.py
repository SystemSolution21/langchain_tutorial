# 6_rag_retriever_llm_response.py
"""
This script implements a RAG (Retrieval-Augmented Generation) system using LangChain.
It queries a vector store for relevant documents, generates an LLMs response based on the query and relevant documents,
and prints the response.
"""

# Import standard libraries
import os
from logging import Logger
from pathlib import Path

# Import environment variables
from dotenv import load_dotenv

# Import langchain modules
from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

# Import custom logger
from utils.logger import RAGLogger

# Load environment variables
load_dotenv()


# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=__name__)

# Log application startup
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG Retriever LLM Response Application")
logger.info(msg="=" * 50)

# Define directories and paths
current_dir: Path = Path(__file__).parent.resolve()
db_dir: Path = current_dir / "db"
store_name: str = "chroma_db_with_metadata"
persistent_directory: Path = db_dir / store_name

# Define embeddings models
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)

# Define LLM
model: str = os.getenv(key="OLLAMA_LLM", default="llama3.2:latest")
llm = ChatOllama(model=model)


# Query vector store
def query_vector_store(
    query: str,
    search_type: str,
    search_kwargs: dict,
) -> list[Document] | None:
    """Query the vector store for relevant documents."""
    if not persistent_directory.exists():
        logger.info(
            msg=f"Vector store '{store_name}' does not exist. Please check the store name and try again."
        )
        return

    try:
        logger.info(msg=f"Querying vector store '{store_name}'...")
        # Load the Chroma vector store
        db: Chroma = Chroma(
            persist_directory=str(object=persistent_directory),
            embedding_function=ollama_embeddings,
        )

        # Debug: Check all unique sources in the database
        all_docs = db.get()
        unique_sources = set(
            doc.get("source", "Unknown") for doc in all_docs["metadatas"]
        )
        logger.info(msg=f"All sources in vector store: {unique_sources}")
        logger.info(msg=f"Total documents in store: {len(all_docs['ids'])}")

        # Debug: Check similarity scores
        results_with_scores = db.similarity_search_with_score(query, k=10)
        logger.info(msg="Top 10 results with scores:")
        for doc, score in results_with_scores:
            logger.info(
                msg=f"Score: {score:.4f}, Source: {doc.metadata.get('source', 'Unknown')}"
            )

        # Debug: Search with metadata filter
        results_langchain = db.similarity_search_with_score(
            query, k=5, filter={"source": "langchain_demo.txt"}
        )
        logger.info(msg="LangChain-specific results:")
        for doc, score in results_langchain:
            logger.info(
                msg=f"Score: {score:.4f}, Content preview: {doc.page_content[:100]}"
            )

        retriever: VectorStoreRetriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs: list[Document] = retriever.invoke(input=query)
        if relevant_docs:
            logger.info(
                msg=f"Retrieved vector store '{store_name}', found {len(relevant_docs)} relevant documents."
            )
        else:
            logger.info(msg="No relevant documents found")
            return None

        # Display the retrieved documents with MetaData
        for i, doc in enumerate(relevant_docs, start=1):
            print(f"\n--- Relevant Document {i} ---")
            print(f"Document:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

        # Return relevant documents
        return relevant_docs

    except Exception as e:
        logger.error(
            msg=f"Unexpected error querying vector store '{store_name}': {str(object=e)}"
        )
        return None


# Define LLMs response
def llm_response(relevant_docs: list[Document], query: str) -> BaseMessage | None:
    """Generate LLMs response based on relevant documents.

    Args:
        relevant_docs: List of relevant documents
    Returns:
        LLMs response
    """
    # Combine the query and relevant documents
    joined_docs: str = "\n\n".join([doc.page_content for doc in relevant_docs])
    combined_input: str = f"""
        Here are some documents that might help answer the question:
        Question: {query}
        Relevant Documents:
        {joined_docs}
        Please provide an answer based only on the provided documents.
        If the answer is not found in the documents, respond with 'I'm not sure'.
    """

    # Define prompt template
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=combined_input),
    ]

    try:
        logger.info(msg="Generating LLMs response on retrieved relevant documents...")
        # Invoke LLMs
        response: BaseMessage = llm.invoke(input=messages)
        return response

    except Exception as e:
        logger.error(msg=f"Unexpected error generating LLMs response: {str(object=e)}")
        return None


def main() -> None:
    # User query - more specific to get better matches
    query: str = "What is LangChain framework?"

    # Query vector store using similarity search
    relevant_docs: list[Document] | None = query_vector_store(
        query=query,
        search_type="similarity",
        search_kwargs={"k": 10},  # Get more results to include LangChain docs
    )

    # Generate LLMs response
    if relevant_docs:
        result: BaseMessage | None = llm_response(
            relevant_docs=relevant_docs, query=query
        )

        # Print the response
        if result:
            logger.info(msg="AI response generated successfully")
            print(f"AI: {result.content}")


# Main entry point
if __name__ == "__main__":
    main()
