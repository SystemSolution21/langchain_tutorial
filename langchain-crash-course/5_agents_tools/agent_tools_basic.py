# agent_tools_basic.py

# Import standard libraries
import os
import sys
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

# Import necessary libraries
from dotenv import load_dotenv

# Import langchain modules
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Import custom logger
from utils.logger import RAGLogger

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
        "Please check your .env file."
    )
    sys.exit(1)

# Set llm
if openai_configured:
    llm = ChatOpenAI(model=str(openai_llm), temperature=0)
elif ollama_configured:
    llm = ChatOllama(model=ollama_llm, temperature=0.8, num_predict=256)


# Module path
module_path: Path = Path(__file__).resolve().parent

# Set logger
logger: Logger = RAGLogger.get_logger(module_name=module_path.name)


# Define current time tool
def get_current_time(*args: Any, **kwargs: Any) -> str:
    """Get current time."""
    now: datetime = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


# Set tool list to Agent
tools: list = [
    Tool(
        name="Current Time",
        func=get_current_time,
        description="Useful to know the current time.",
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
prompt: Any = hub.pull(owner_repo_commit="hwchase17/react")

# Create an agent
agent: Runnable[Any, Any] = create_react_agent(
    tools=tools, llm=llm, prompt=prompt, stop_sequence=True
)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Run agent executor
def main() -> None:
    logger.info(msg="Start Agent Tools Basic Application...")
    print("Type 'exit' to end the conversation.")

    while True:
        try:
            # User Query
            query: str = input("You: ").strip()

            if not query:
                continue

            if query.lower() == "exit":
                logger.info("User exited conversation")
                print("Exiting...")
                break

            # Run agent executor
            response: Any = agent_executor.invoke(input={"input": query})
            print(f"Agent: {response['output']}")

        except (KeyboardInterrupt, EOFError):
            logger.info(msg="Keyboard interrupt or EOF error")
            print("Exiting...")
            break

        except Exception as e:
            logger.error(f"Unexpected error: {e}")


# Main entry point
if __name__ == "__main__":
    main()
