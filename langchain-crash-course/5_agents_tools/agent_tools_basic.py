# agent_tools_basic.py
"""
Agent Tools Basic Application
(Asynchronous Version)

This module demonstrates a basic implementation of a LangChain agent with tools.
It creates a ReAct (Reason and Action) agent that can use predefined tools to
answer user questions in an interactive conversation loop.

The agent uses either OpenAI or Ollama as the language model backend and includes
a simple tool for getting the current time. The implementation follows the ReAct
pattern where the agent reasons about what action to take, executes the action,
observes the result, and continues until it can provide a final answer.

Features:
- Configurable LLM backend (OpenAI or Ollama)
- ReAct agent pattern implementation
- Interactive conversation loop
- Custom tool integration
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

# Import async console library
from aioconsole import ainput

# Import necessary libraries
from dotenv import load_dotenv

# Import langchain modules
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Import custom logger
from utils.logger import ReActAgentLogger

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
    llm = ChatOpenAI(model=str(object=openai_llm), temperature=0)
elif ollama_configured:
    llm = ChatOllama(model=ollama_llm)


# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = ReActAgentLogger.get_logger(module_name=module_path.name)


# Define current time tool
def get_current_time(*args: Any, **kwargs: Any) -> str:
    """Get current time."""
    current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time


# Set tools list to Agent
tools: list = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful to know the current time.",
    ),
]

# Pull the prompt template from the hub
# ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react
# prompt: Any = hub.pull(owner_repo_commit="hwchase17/react")

# Prompt template as same as hwchase17/react
prompt_template: str = """ 
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

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

prompt: PromptTemplate = PromptTemplate.from_template(template=prompt_template)

# Create an agent
agent: Runnable[Any, Any] = create_react_agent(
    tools=tools, llm=llm, prompt=prompt, stop_sequence=True
)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Run agent executor
async def main() -> None:
    """
    Main function to run the agent executor.

    This function continuously prompts the user for a query, runs the agent
    executor, and displays the response. The loop can be exited by typing
    'exit', or by sending a KeyboardInterrupt (Ctrl+C) or EOFError (Ctrl+D).
    """
    logger.info(msg="========= Start Agent Tools Basic Application ==========")
    print("Type 'exit' to end the conversation.")

    while True:
        try:
            # User Query
            query: str = (await ainput("You: ")).strip()

            if not query:
                continue

            if query.lower() == "exit":
                logger.info("User exited conversation")
                print("Exiting...")
                break

            # Run agent executor
            response: Any = await agent_executor.ainvoke(input={"input": query})
            logger.info(msg=f"Agent: {response['output']}")
            print(f"Agent: {response['output']}")

        except (KeyboardInterrupt, EOFError):
            logger.info(msg="Keyboard interrupt or EOF error")
            print("Exiting...")
            break

        except Exception as e:
            logger.error(f"Unexpected error: {e}")


# Main entry point
if __name__ == "__main__":
    asyncio.run(main())
