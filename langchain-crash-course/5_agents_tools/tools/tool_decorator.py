# tool_constructor.py

# Import standard libraries
import asyncio
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
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()  # LangSmith API key required for tracing


# ==================== Define tools====================
@tool
def greet_user(name: str) -> str:
    """Greet the user."""
    return f"Hello, {name}! Welcome to LangChain."


@tool
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


class ConcatenateStringsArgs(BaseModel):
    """Input for concatenate_strings."""

    text1: str = Field(description="First string")
    text2: str = Field(description="Second string")


# args_schema is for structured tools
@tool(args_schema=ConcatenateStringsArgs)
def concatenate_strings(text1: str, text2: str) -> str:
    """Concatenate two strings."""
    return text1 + text2


# ==================== Create tools ====================
tools: list = [
    greet_user,
    reverse_string,
    concatenate_strings,
]

# ==================== Create LLM====================
# Create Chat Model
llm = ChatOllama(model="llama3.2:3b")

# pull prompt template from hub
prompt_template: Any = hub.pull(owner_repo_commit="hwchase17/openai-tools-agent")


# ==================== Create agent====================
agent: Runnable[Any, Any] = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template,
)

# ==================== Create agent executor====================
agent_executor: AgentExecutor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)


# ==================== Run tools calling agent ====================
async def main() -> None:
    print(
        "\nStart chatting with Decorator Tool Calling Agent AI! Type 'exit' to end the conversation."
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
