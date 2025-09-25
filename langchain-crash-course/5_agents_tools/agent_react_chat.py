# agent_react_chat.py
"""
ReAct Chat Agent Application
(Asynchronous Version)

This module demonstrates a conversational LangChain agent with memory.
It creates a structured chat agent that can use predefined tools to
answer user questions in an interactive conversation loop, maintaining
the context of the conversation.

The agent uses either OpenAI or Ollama as the language model backend and includes
tools for getting the current time and fetching summaries from Wikipedia. It
leverages `RunnableWithMessageHistory` to manage chat history for each session,
allowing for contextual conversations.

Features:
- Configurable LLM backend (OpenAI or Ollama)
- Structured chat agent with ReAct (Reason and Action) logic
- Conversational memory management
- Custom tool integration (Time, Wikipedia)
- Interactive asynchronous conversation loop
- Comprehensive logging
"""

# Import standard libraries
import asyncio
import os
import sys
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

# Import necessary libraries
from aioconsole import ainput
from dotenv import load_dotenv

# Import langchain modules
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Import custom modules
from utils.logger import RAGLogger

# Import Wikipedia library
from wikipedia import summary

# Load environment variables from .env
load_dotenv()

# Check API key and LLM models
api_key: str | None = os.getenv(key="OPENAI_API_KEY")
openai_llm: str | None = os.getenv(key="OPENAI_LLM")
ollama_llm: str | None = os.getenv(key="OLLAMA_LLM")

openai_configured: str | None = api_key and openai_llm
ollama_configured: str | None = ollama_llm

if not openai_configured and not ollama_configured:
    raise ValueError(
        "Neither OpenAI (OPENAI_API_KEY, OPENAI_LLM) nor Ollama (OLLAMA_LLM) is configured. "
        "Please check your .env file. Note: Ollama llm model should be locally installed."
    )
    sys.exit(1)

# Set llm
if openai_configured:
    llm = ChatOpenAI(model=str(object=openai_llm), temperature=0)
elif ollama_configured:
    llm = ChatOllama(model=ollama_llm)


# Module path
module_path: Path = Path(__file__).resolve().parent

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)


# Define current time tool
def get_current_time(*args: Any, **kwargs: Any) -> str:
    """Get current time."""
    current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(msg=f"Getting current time: {current_time}")
    return current_time


def get_wikipedia_summary(query: str) -> str:
    """Get a summary from Wikipedia."""
    try:
        summary_result: str = summary(title=query, sentences=10)
        logger.info(msg=f"Getting Wikipedia summary: {summary_result[:100]}.....")
        return summary_result
    except Exception as e:
        logger.error(msg=f"Error getting Wikipedia summary: {e}")
        return f"Error: {e}"


# Set tools list to Agent
tools: list = [
    Tool(
        name="Current Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=get_wikipedia_summary,
        description="Useful for when you need to answer questions about the topic.",
    ),
]

# Pull the prompt template from the hub
# https://smith.langchain.com/hub/hwchase17/structured-chat-agent
prompt_template: Any = hub.pull(owner_repo_commit="hwchase17/structured-chat-agent")

# Set up chat history
store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get chat history for a session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Create structured chat agent
agent: Runnable[Any, Any] = create_structured_chat_agent(
    tools=tools, llm=llm, prompt=prompt_template
)

# Create agent executor
agent_executor: AgentExecutor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

# Add chat history to agent executor
agent_with_chat_history: RunnableWithMessageHistory = RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# Run agent executor with chat history
async def main() -> None:
    logger.info(msg="Start Agent React Chat Application...")
    print("Type 'exit' to end the conversation.")

    session_id: str = "chat_session"

    while True:
        try:
            # User Query
            query: str = (await ainput("You: ")).strip()

            if not query:
                continue

            if query.lower() == "exit":
                logger.info(msg="User exited conversation")
                print("Exiting...")
                break

            # Run agent executor with chat history
            response: Any = await agent_with_chat_history.ainvoke(
                input={"input": query},
                config={"configurable": {"session_id": session_id}},
            )
            print(f"Agent: {response['output']}")

        except (KeyboardInterrupt, EOFError):
            logger.info(msg="Keyboard interrupt or EOF error")
            print("Exiting...")
            break

        except Exception as e:
            logger.error(msg=f"Unexpected error: {e}")


# Main entry point
if __name__ == "__main__":
    asyncio.run(main=main())
