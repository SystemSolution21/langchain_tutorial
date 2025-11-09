"""
Demonstrates how to create and use LangChain tools by subclassing BaseTool.

This script defines two custom tools:
1. SimpleWebSearchTool: A tool that uses the Tavily API to perform a web search.
2. MultiplyNumbersTool: A simple tool to multiply two numbers.

It then creates a LangChain agent that can use these tools and runs an interactive
chat session where the user can interact with the agent. This approach provides
fine-grained control over the tool's implementation.
"""

# Import standard libraries
import asyncio
import os
import sys
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Type

# Import necessary libraries
from aioconsole import ainput
from dotenv import load_dotenv

# Import langchain modules
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from tavily import TavilyClient

# Import custom logger
from util.logger import ReActAgentLogger

# Load environment variables
load_dotenv()

# Module path
module_path: Path = Path(__file__).resolve()

# Set logger
logger: Logger = ReActAgentLogger.get_logger(module_name=module_path.name)

# ==================== Define tools====================


class SimpleWebSearch(BaseModel):
    """Input model for the SimpleWebSearchTool, specifying the search query."""

    query: str = Field(description="Search query")


class MultiplyNumbers(BaseModel):
    """Input model for the MultiplyNumbersTool, specifying the two numbers to multiply."""

    num1: float = Field(description="First number")
    num2: float = Field(description="Second number")


class SimpleWebSearchTool(BaseTool):
    """Tool for searching the web."""

    name: str = "Simple_Web_Search"
    description: str = "Useful for searching the web."
    args_schema: Type[BaseModel] = SimpleWebSearch

    def _run(self, query: str) -> str:
        """Executes the web search synchronously."""
        try:
            client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            results: Dict[str, Any] = client.search(query=query)
            return f"Search results for: {query}\n\n{results}\n"
        except Exception as e:
            return f"An error occurred during the Tavily search: {e}"

    async def _arun(self, query: str) -> str:
        """Executes the web search asynchronously."""
        return await asyncio.to_thread(self._run, query)


class MultiplyNumbersTool(BaseTool):
    """Tool for multiplying two numbers."""

    name: str = "Multiply_Numbers"
    description: str = "Useful for multiplying two numbers."
    args_schema: Type[BaseModel] = MultiplyNumbers

    def _run(self, num1: float, num2: float) -> str:
        """Executes the multiplication synchronously."""
        result: float = num1 * num2
        return f"The product of {num1} and {num2} is {result}\n"


tools: list = [
    SimpleWebSearchTool(),  # Simple web search tool
    MultiplyNumbersTool(),  # Multiply numbers tool
]

# ==================== Create LMM and Agent ====================
# Create Chat Model
llm = ChatOllama(model="llama3.2:3b")

# pull prompt template from hub
prompt_template: Any = hub.pull(owner_repo_commit="hwchase17/openai-tools-agent")

# Create an agent
agent: Runnable[Any, Any] = create_tool_calling_agent(
    llm=llm,  # llm to use
    tools=tools,  # tools to use
    prompt=prompt_template,  # prompt to use
)

# Create agent executor
agent_executor: AgentExecutor = AgentExecutor.from_agent_and_tools(
    agent=agent,  # agent to use
    tools=tools,  # tools to use
    verbose=True,  # prints out the agent's thought process
    handle_parsing_errors=True,  # gracefully handles errors in parsing the agent output
)


# ==================== Run tools calling agent ====================
async def main() -> None:
    """
    Runs the main asynchronous loop for the chat application.

    This function initializes the agent, handles API key checks, and manages
    the interactive chat session with the user, including history management
    and graceful exit.
    """
    # Check TAVILY_API_KEY
    if not os.getenv("TAVILY_API_KEY"):
        print(
            "\n[ERROR] TAVILY_API_KEY not found in environment variables."
            "\nPlease set the key in your .env file to use the web search tool."
        )
        sys.exit(1)  # Exit the application with a non-zero status code

    logger.info(msg="========= Start BaseTool Calling AI Agent Application ==========")
    print("Type 'exit' to end the conversation.")

    # Initialize chat history
    chat_history: list[BaseMessage] = []

    while True:
        try:
            query: str = (await ainput(prompt="You: ")).strip()

            if not query:
                print("Please ask a question!.")
                continue

            if query.lower() == "exit":
                logger.info(msg="User exited conversation")
                print("Exiting...")
                break

            # Process user query through agent executor
            response: Any = await agent_executor.ainvoke(
                input={"input": query, "chat_history": chat_history}
            )

            # Display AI response
            if response:
                logger.info(msg=f"AI: {response['output']:100}.....")
                print(f"AI: {response['output']}")

                # Update chat history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=response["output"]))

        except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
            logger.info(msg="Keyboard interrupt or EOF error")
            print("Exiting...")
            break

        except Exception as e:
            logger.error(msg=f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main=main())
