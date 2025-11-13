# agent_react_rag_pdf_advanced.py
"""
ReAct Agent with RAG PDF Advanced Application
(Asynchronous Version)

This module demonstrates a conversational LangChain agent with PDF RAG context.
It creates a structured chat agent that can use predefined tools to answer user
questions based on PDF documents in an interactive conversation loop, maintaining
the context of the conversation.
"""

# Import standard libraries
import asyncio
import os
import sys
from logging import Logger
from pathlib import Path
from typing import Any, List

# Import necessary library
from aioconsole import ainput
from dotenv import load_dotenv

# Import langchain modules
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.base import Runnable
from langchain_core.tools import Tool
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Import custom modules
from rag_pdf_advanced import (
    generate_sample_questions,
    initialize_advanced_pdf_vector_store,
    update_vector_store,
)
from rich.console import Console
from rich.markdown import Markdown
from util.logger import ReActAgentLogger

# ==================== Setup PDFs and database directories===================
current_dir: Path = Path(__file__).parent.resolve()
pdfs_dir: Path = current_dir / "pdfs"
db_dir: Path = current_dir / "db"
store_name: str = "chroma_db_pdf_advanced"
persistent_directory: Path = db_dir / store_name

# =================== Setup Logger ====================
module_path: Path = Path(__file__).resolve()
logger: Logger = ReActAgentLogger.get_logger(module_name=module_path.name)
# Log application startup
logger.info(
    msg="========= Starting ReAct Agent with RAG PDF Advanced Application =========="
)

# ==================== Setup RAG ====================
# Load environment variables
load_dotenv()

# Get environment variables
ollama_llm: str | None = os.getenv(key="OLLAMA_LLM", default="gemma3:4b")

# Define embeddings models - MUST match the embeddings used in rag_pdf_advanced.py
# Using BAAI/bge-large-en-v1.5 for better semantic understanding (1024 dimensions)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Define LLM
llm = ChatOllama(model=ollama_llm)

# Load vector store and create retriever
try:
    # Check vector store
    if not persistent_directory.exists():
        # Initialize vector store
        db_instance = initialize_advanced_pdf_vector_store(
            pdfs_dir=pdfs_dir, persistent_directory=persistent_directory
        )
        # initialization failed, the directory might not exist.
        if db_instance is None and not persistent_directory.exists():
            logger.error(
                "Failed to initialize the PDF vector store. Please check the logs for details."
            )
            sys.exit(1)
    logger.info(msg=f"Loading PDF vector store '{store_name}'...")

    # Load the Chroma vector store
    db: Chroma = Chroma(
        persist_directory=str(persistent_directory),
        embedding_function=embeddings,
    )

    # Check for PDF changes and update vector store if needed
    logger.info(msg="Checking for PDF changes...")
    try:
        updated_db = update_vector_store(
            pdfs_dir=pdfs_dir,
            persistent_directory=persistent_directory,
            embeddings=embeddings,
        )
        if updated_db is not None:
            db = updated_db
            logger.info(msg="Vector store updated with new/modified PDFs")
    except Exception as update_error:
        logger.warning(f"Could not check for PDF updates: {update_error}")

except Exception as e:
    logger.error(msg=f"Error loading PDF vector store '{store_name}': {e}")
    sys.exit(1)

# Create a retriever
retriever: VectorStoreRetriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
logger.info(msg=f"Created retriever from PDF vector store '{store_name}' successfully.")

# Generate and display sample questions from PDF content
try:
    sample_questions = generate_sample_questions(db=db, num_questions=5)
    if sample_questions:
        print("\n" + "=" * 70)
        print("📚 SAMPLE QUESTIONS FROM YOUR PDF DOCUMENTS:")
        print("=" * 70)
        for i, question in enumerate(sample_questions, 1):
            print(f"{i}. {question}")
        print("=" * 70 + "\n")
except Exception as e:
    logger.warning(f"Could not generate sample questions: {e}")

# Contextualize question prompt
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
history_aware_retriever: VectorStoreRetriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt_template,
)

# Answer question prompt
qa_system_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Limit your response to less than 100 sentences and keep the answer concise.

When answering, provide specific information from the context and reference the sources.
The sources will be automatically appended to your answer.
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
question_answering_chain: Runnable[dict[str, Any], Any] = create_stuff_documents_chain(
    llm=llm, prompt=qa_prompt_template
)

# Create a RAG chain that combines the history-aware retriever and the question answering chain
rag_chain: Runnable[dict[str, Any], Any] = create_retrieval_chain(
    history_aware_retriever, question_answering_chain
)

# ==================== Setup ReAct Agent with RAG ====================
react_prompt_template: str = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: When the tool returns an answer with "📚 Sources:" section, 
you MUST include the entire sources section in your Final Answer exactly as provided. 
Do not summarize or remove the source citations.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt_template: PromptTemplate = PromptTemplate.from_template(
    template=react_prompt_template
)


# ==================== Pydantic Models for Structured Output ====================
class SourceMetadata(BaseModel):
    """Metadata for a single source document."""

    file: str = Field(description="Name of the source PDF file")
    page: str = Field(description="Page number or 'N/A' if not available")
    type: str = Field(
        description="Content type: text, table, ocr_images, chart_graph, or structured"
    )


class RAGResponse(BaseModel):
    """Structured response from RAG system with answer and sources."""

    answer: str = Field(description="The answer to the user's question")
    sources: List[SourceMetadata] = Field(
        description="List of source documents used to generate the answer"
    )

    def format_response(self) -> str:
        """Format the response with sources for display."""
        if not self.sources:
            return self.answer

        # Build formatted response with proper Markdown list
        response_text = self.answer + "\n\n**📚 Sources:**\n\n"
        for i, src in enumerate(self.sources, 1):
            page_info = f"Page {src.page}" if src.page != "N/A" else "N/A"
            # Use Markdown list format (- or *) for proper indentation
            response_text += (
                f"- **[{i}]** `{src.file}` ({page_info}) - *Type: {src.type}*\n"
            )

        return response_text


# RAG responses with source metadata
def rag_with_sources(input: str, **kwargs) -> str:
    """
    Invoke RAG chain and format response with source metadata.

    Args:
        input: User query
        **kwargs: Additional arguments including chat_history

    Returns:
        Dictionary with answer and source information
    """
    # Invoke the RAG chain
    result = rag_chain.invoke(
        input={"input": input, "chat_history": kwargs.get("chat_history", [])}
    )

    # Extract answer and context documents
    answer = result.get("answer", "")
    context_docs = result.get("context", [])

    # Build source information using Pydantic models
    sources = []
    seen_sources = set()

    for doc in context_docs:
        source = doc.metadata.get("source", "Unknown")
        page = str(doc.metadata.get("page", "N/A"))
        content_type = doc.metadata.get("content_type", "text")

        # Create unique identifier for this source
        source_id = f"{source}::{page}"

        if source_id not in seen_sources:
            seen_sources.add(source_id)
            sources.append(SourceMetadata(file=source, page=page, type=content_type))

    # Create structured response
    rag_response = RAGResponse(answer=answer, sources=sources)

    # Return formatted response
    return rag_response.format_response()


# create a tool that uses the RAG chain with source metadata
tools: list[Tool] = [
    Tool(
        name="Answer Question",
        func=rag_with_sources,
        description="Useful for answering questions based on PDF documents. Returns answer with source citations.",
    ),
]

# Create a ReAct agent with the RAG tool
agent: Runnable[Any, Any] = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt_template,
)

# Create agent executor
agent_executor: AgentExecutor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)


# ==================== Run ReAct Agent with PDF RAG context conversation ====================
async def main() -> None:
    """
    Runs the main conversational loop for the PDF RAG-based ReAct chat application.

    This function initializes the chat history and enters an infinite loop to
    continuously accept user input. It processes the user's query through the
    RAG chain, prints the AI's response, and updates the chat history.
    The loop can be exited by typing 'exit', or by sending a
    KeyboardInterrupt (Ctrl+C) or EOFError (Ctrl+D).
    """
    print(
        "\nStart ReAct Agent with PDF RAG context chatting! Type 'exit' to end the conversation."
    )

    # Initialize chat history
    chat_history: list[BaseMessage] = []

    while True:
        try:
            # User query
            query: str = (await ainput("You: ")).strip()

            if not query:
                print("Please ask a question!.")
                continue

            if query.lower() == "exit":
                logger.info(msg="User exited conversation")
                print("Exiting...")
                break

            # Process user query through agent executor
            logger.info(
                msg="Processing user query through ReAct Agent with PDF RAG context..."
            )
            response: Any = await agent_executor.ainvoke(
                input={"input": query, "chat_history": chat_history}
            )

            # Display AI response
            if response:
                logger.info(msg="AI response generated successfully")

                # Create Rich console for Markdown rendering
                console = Console()

                # Render response with Markdown formatting
                print("\nAI: ", end="")
                console.print(Markdown(response["output"]))

                # Update chat history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=response["output"]))
                logger.info(msg="Chat history updated successfully")

        except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
            logger.info(msg="Keyboard interrupt or EOF error")
            print("Exiting...")
            break

        except Exception as e:
            logger.error(msg=f"Unexpected error: {e}")
            print("Exiting...")
            break


# Main entry point
if __name__ == "__main__":
    asyncio.run(main())
