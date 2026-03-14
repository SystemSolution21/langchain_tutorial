# tool_constructor.py
"""
Demonstrates how to create and use LangChain tools using Tool and StructuredTool constructors.

This script defines three custom tools:
1. Greet User: A simple tool to greet the user.
2. Reverse String: A tool to reverse a string.
3. Concatenate Strings: A tool to concatenate two strings.

It then creates a LangChain agent that can use these tools and runs an interactive
chat session where the user can interact with the agent. This approach provides
fine-grained control over the tool's implementation.
"""

# Import standard libraries
import asyncio
import os
from typing import Any

# Import necessary libraries
from aioconsole import ainput
from dotenv import load_dotenv

# Import langchain modules
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool, Tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()  # LangSmith API key required for tracing


# ==================== Define tools====================
def greet_user(name: str) -> str:
    """Greet the user."""
    return f"Hello, {name}! Welcome to LangChain."


def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


def concatenate_strings(text1: str, text2: str) -> str:
    """Concatenate two strings."""
    return text1 + text2


class ConcatenateStringsArgs(BaseModel):
    """Input for concatenate_strings."""

    text1: str = Field(description="First string")
    text2: str = Field(description="Second string")


# ==================== Create tools using Tool and StructuredTools constructors====================
tools: list = [
    Tool.from_function(
        func=greet_user,
        name="Greet User",
        description="Useful for greeting the user.",
    ),
    Tool.from_function(
        func=reverse_string,
        name="Reverse String",
        description="Useful for reversing a string.",
    ),
    StructuredTool.from_function(
        func=concatenate_strings,  # function to call
        name="Concatenate Strings",  # name of the tool
        description="Useful for concatenating two strings.",  # description of the tool
        args_schema=ConcatenateStringsArgs,  # args schema for the tool
    ),
]

# ==================== Create LLM====================
# Create Chat Model
model: str = os.getenv(key="OLLAMA_LLM", default="llama3.2:latest")
llm = ChatOllama(model=model)

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
    print("\nStart chatting with Constructor Tool Calling Agent AI!")

    # Initialize chat history
    chat_history: list[BaseMessage] = []

    while True:
        try:
            query: str = (
                await ainput(prompt="Type 'exit' to end the conversation!\nYou: ")
            ).strip()

            if not query:
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
