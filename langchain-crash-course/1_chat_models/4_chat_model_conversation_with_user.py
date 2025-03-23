from typing import Any, List, Dict, Union
from langchain_core.messages.base import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Create gemini chat model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

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

    # Invoke model using chat history
    result: BaseMessage = gemini_model.invoke(input=chat_history)
    response: Union[str, list[str | Dict[str, Any]]] = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message
    print(f"AI: {response}")  # Print the response with a newline

print("\n-----chat history-----")
for message in chat_history:
    print(f"{message.type}: {message.content}")
