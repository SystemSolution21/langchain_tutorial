from typing import Any, List, Dict, Union
from langchain_core.messages.base import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# # Create chat model
# # model: str = "gemini-1.5-flash"
# llm = ChatGoogleGenerativeAI(model=model)

# Create chat model
model: str = "llama3.2:3b"
llm = ChatOllama(model=model)


# Store chat history
chat_history: List[BaseMessage] = []

# Set system message
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # Add system message to chat history

# Chat looping
while True:
    query: str = input("You: ")
    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))  # Add user message
    #
    # Invoke model using chat model: str = "llama3.2:3b"history
    result: BaseMessage = llm.invoke(input=model)
    response: Union[str, list[str | Dict[str, Any]]] = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message
    print(f"AI: {response}")  # Print the response with a newline

print("\n-----chat history-----")
for message in chat_history:
    print(f"{message.type}: {message.content}")
