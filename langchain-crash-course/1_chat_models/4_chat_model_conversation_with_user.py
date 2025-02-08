from dotenv import load_dotenv
from langchain_core.messages.base import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


# Load environment variables
load_dotenv()

# Create gemini chat model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Store chat history
chat_history: list = []

# Set system message
system_message = SystemMessage(content="Your are helpful AI assistant.")
chat_history.append(system_message)  # Add system message to chat history

# Chat looping
while True:
    query: str = input("You: ")
    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))  # Add user message

    # Invoke model using chat history
    result: BaseMessage = gemini_model.invoke(input=chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message
    print(f"AI: {response}")

print("-----chat history-----")
print(chat_history)
