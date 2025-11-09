# tool_basetool.py

# Import standard libraries
import asyncio
import os
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

# Load environment variables
load_dotenv()


# ==================== Define tools====================


class SimpleWebSearch(BaseModel):
    """Input for simple_web_search."""

    query: str = Field(description="Search query")


class MultiplyNumbers(BaseModel):
    """Input for multiply_numbers."""

    num1: float = Field(description="First number")
    num2: float = Field(description="Second number")


class SimpleWebSearchTool(BaseTool):
    """Tool for searching the web."""

    name: str = "Simple_Web_Search"
    description: str = "Useful for searching the web."
    args_schema: Type[BaseModel] = SimpleWebSearch

    def _run(self, query: str) -> str:
        """Use the tool."""
        api_key: str | None = os.getenv(key="TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results: Dict[str, Any] = client.search(query=query)
        return f"Search results for: {query}\n\n{results}\n"


class MultiplyNumbersTool(BaseTool):
    """Tool for multiplying two numbers."""

    name: str = "Multiply_Numbers"
    description: str = "Useful for multiplying two numbers."
    args_schema: Type[BaseModel] = MultiplyNumbers

    def _run(self, num1: float, num2: float) -> str:
        """Use the tool."""
        result: float = num1 * num2
        return f"The product of {num1} and {num2} is {result}\n"


# ==================== Create tools using BaseTool====================
tools: list = [
    SimpleWebSearchTool(),  # Simple web search tool
    MultiplyNumbersTool(),  # Multiply numbers tool
]

# ==================== Create LLM====================
# Create Chat Model
llm = ChatOllama(model="llama3.2:3b")

# pull prompt template from hub
prompt_template: Any = hub.pull(owner_repo_commit="hwchase17/openai-tools-agent")


# ==================== Create agent====================
agent: Runnable[Any, Any] = create_tool_calling_agent(
    llm=llm,  # llm to use
    tools=tools,  # tools to use
    prompt=prompt_template,  # prompt to use
)

# ==================== Create agent executor====================
agent_executor: AgentExecutor = AgentExecutor.from_agent_and_tools(
    agent=agent,  # agent to use
    tools=tools,  # tools to use
    verbose=True,  # prints out the agent's thought process
    handle_parsing_errors=True,  # gracefully handles errors in parsing the agent output
)


# ==================== Run tools calling agent ====================
async def main() -> None:
    print(
        "\nStart chatting with BaseTool Calling Agent AI! Type 'exit' to end the conversation."
    )

    # Initialize chat history
    chat_history: list[BaseMessage] = []

    while True:
        try:
            query: str = (await ainput(prompt="You: ")).strip()

            if not query:
                print("Please ask a question!.")
                continue

            if query.lower() == "exit":
                print("Exiting...")
                break

            # Process user query through agent executor
            response: Any = await agent_executor.ainvoke(
                input={"input": query, "chat_history": chat_history}
            )

            # Display AI response
            if response:
                print(f"AI: {response['output']}")

                # Update chat history
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=response["output"]))

        except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
            print("Exiting...")
            break

        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main=main())
