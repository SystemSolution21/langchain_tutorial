# 7_rag_llm_conversation.py
"""
A Retrieval-Augmented Generation (RAG) application for conversational AI.

This script sets up a conversational AI that can answer questions based on a
pre-existing knowledge base stored in a Chroma vector store. It maintains a
chat history to provide context-aware responses.

The pipeline consists of the following steps:
1.  **History-Aware Retrieval**: Reformulates the user's question to be a
    standalone query by considering the chat history.
2.  **Document Retrieval**: Fetches relevant documents from the Chroma vector
    store based on the reformulated question.
3.  **Question Answering**: Generates a response using the retrieved documents,
    the original question, and the chat history.

The application uses Ollama for both embeddings ('nomic-embed-text') and the
language model ('gemma3:4b'). The user can interact with the AI in a loop via
the command line.
"""

# Import standard libraries
import sys
from logging import Logger
from pathlib import Path
from typing import Any

# Import environment variables
from dotenv import load_dotenv

# Import langchain modules
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import Runnable
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
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)

# Log application startup
logger.info(msg="=" * 50)
logger.info(msg="Starting RAG LLM Conversation Application")
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
llm = ChatOllama(model="gemma3:4b")

# Check vector store existence
if not persistent_directory.exists():
    logger.error(
        msg=f"Vector store '{store_name}' does not exist. Please check the path."
    )
    sys.exit(1)

# Load vector store and create retriever
try:
    logger.info(msg=f"Loading vector store '{store_name}'...")
    # Load the Chroma vector store
    db: Chroma = Chroma(
        persist_directory=str(persistent_directory),
        embedding_function=ollama_embeddings,
    )
    # Create a retriever
    retriever: VectorStoreRetriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    logger.info(msg=f"Created retriever from vector store '{store_name}' successfully.")

except Exception as e:
    logger.error(msg=f"Unexpected error querying vector store '{store_name}': {e}")
    sys.exit(1)

# Contextualize question prompt
# System prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = """
Given a chat history and the latest user question, this prompt helps the AI reformulate 
the question to be standalone. The reformulated question should be understandable 
without relying on prior chat context. The AI should not answer the question—only 
rephrase it if necessary, or return it unchanged if already standalone.
"""

# Create contextualize question prompt template
contextualize_q_prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=[
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# this users the LLM to help reformulate the question based on chat history
history_aware_retriever: VectorStoreRetriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt_template,
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Limit your response to a maximum of ten sentences and keep the answer concise.
\n\n
{context}
"""

# Create answer question prompt template
qa_prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=[
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answering_chain: Runnable[dict[str, Any], Any] = create_stuff_documents_chain(
    llm=llm, prompt=qa_prompt_template
)

# Create a RAG chain that combines the history-aware retriever and the question answering chain
rag_chain: Runnable[dict[str, Any], Any] = create_retrieval_chain(
    history_aware_retriever, question_answering_chain
)


# Run RAG LLM conversation
def main() -> None:
    """
    Runs the main conversational loop for the RAG-based chat application.

    This function initializes the chat history and enters an infinite loop to
    continuously accept user input. It processes the user's query through the
    RAG chain, prints the AI's response, and updates the chat history.
    The loop can be exited by typing 'exit', or by sending a
    KeyboardInterrupt (Ctrl+C) or EOFError (Ctrl+D).
    """
    print("\nStart chatting with AI! Type 'exit' to end the conversation.")

    # Initialize chat history
    chat_history: list[BaseMessage] = []

    while True:
        try:
            # User query
            query: str = input("You: ").strip()

            if not query:
                continue

            if query.lower() == "exit":
                logger.info(msg="User exited conversation")
                print("Exiting...")
                break

            # Process user query through RAG chain
            logger.info(msg="Processing user query through RAG chain...")
            result: Any = rag_chain.invoke(
                input={"input": query, "chat_history": chat_history}
            )

            # Display AI response
            if result:
                logger.info(msg="AI response generated successfully")
                print(f"AI: {result['answer']}")

                # Update chat history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=result["answer"]))
                logger.info(msg="Chat history updated successfully")

        except (KeyboardInterrupt, EOFError):
            logger.info(msg="Keyboard interrupt or EOF error")
            print("Exiting...")
            break

        except Exception as e:
            logger.error(msg=f"Unexpected error: {e}")
            print("Exiting...")
            break


# Main entry point
if __name__ == "__main__":
    main()
