# agent_react_rag_context.py

# Import standard libraries
import sys
from logging import Logger
from pathlib import Path
from typing import Any

# Import environment variables
from dotenv import load_dotenv

# Import langchain modules
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
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
from langchain_core.tools import Tool
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
logger.info(msg="Starting Agent ReAct RAG Context Application")
logger.info(msg="=" * 50)

# ===== Setup RAG =====
# Define directories and paths
rag_dir: Path = Path(__file__).parents[1] / "4_rag"
db_dir: Path = rag_dir / "db"
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
# this helps LLM to reformulate the question based on chat history
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

# ===== Setup ReAct Agent with RAG =====
# load ReAct prompt template from hub
react_prompt_template: Any = hub.pull(owner_repo_commit="hwchase17/react")

# create a tool that uses the RAG chain
tools: list[Tool] = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            input={"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="Useful for answering questions based on the provided context.",
    ),
]

# Create a ReAct agent with the RAG tool
agent: Runnable[Any, Any] = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=react_prompt_template,
)

# Create agent executor
agent_executor: AgentExecutor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)


# ===== Run ReAct RAG conversation =====
def main() -> None:
    """
    Runs the main conversational loop for the RAG-based ReAct chat application.

    This function initializes the chat history and enters an infinite loop to
    continuously accept user input. It processes the user's query through the
    RAG chain, prints the AI's response, and updates the chat history.
    The loop can be exited by typing 'exit', or by sending a
    KeyboardInterrupt (Ctrl+C) or EOFError (Ctrl+D).
    """
    print("\nStart RAG-based ReAct chatting! Type 'exit' to end the conversation.")

    # Initialize chat history
    chat_history: list[BaseMessage] = []

    while True:
        try:
            # User query
            query: str = input("You: ").strip()

            if not query:
                print("Please ask a question!.")
                continue

            if query.lower() == "exit":
                logger.info(msg="User exited conversation")
                print("Exiting...")
                break

            # Process user query through agent executor
            logger.info(
                msg="Processing user query through ReAct Agent with RAG chain..."
            )
            response: Any = agent_executor.invoke(
                input={"input": query, "chat_history": chat_history}
            )

            # Display AI response
            if response:
                logger.info(msg="AI response generated successfully")
                print(f"AI: {response['output']}")

                # Update chat history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=response["output"]))
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
