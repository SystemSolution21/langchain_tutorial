from typing import Any, List, Dict, Union
from google.cloud.firestore_v1.client import Client
from langchain_core.messages.base import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Firebase
file_path: Path = Path(__file__).parent.resolve()
cred = credentials.Certificate(cert=file_path / "firebase_config.json")
firebase_admin.initialize_app(credential=cred)
db: Client = firestore.client()


def save_message_to_firebase(message: BaseMessage, conversation_id: str) -> None:
    """Save a message to Firebase Firestore."""
    messages_ref: Any = (
        db.collection("conversations")
        .document(document_id=conversation_id)
        .collection("messages")
    )

    message_data: Dict[str, Any] = {
        "type": message.type,
        "content": message.content,
        "timestamp": datetime.now(),
    }

    messages_ref.add(message_data)


# Create chat model
model_name: str = "gemini-1.5-flash"
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model=model_name)

# Store chat history
chat_history: List[BaseMessage] = []

# Generate a unique conversation ID (you could also use user ID if available)
conversation_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set system message
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)
save_message_to_firebase(message=system_message, conversation_id=conversation_id)

# Chat looping
while True:
    # Get user input
    query: str = input("You: ")
    # Exit if user enters "exit"
    if query.lower() == "exit":
        break

    # Add and save user message
    human_message = HumanMessage(content=query)
    chat_history.append(human_message)
    save_message_to_firebase(message=human_message, conversation_id=conversation_id)

    # Invoke model using chat history
    result: BaseMessage = model.invoke(input=chat_history)
    response: Union[str, list[str | Dict[str, Any]]] = result.content

    # Add and save AI message
    ai_message = AIMessage(content=response)
    chat_history.append(ai_message)
    save_message_to_firebase(message=ai_message, conversation_id=conversation_id)

    print(f"AI: {response}")

print("\n-----chat history-----")
for message in chat_history:
    print(f"{message.type}: {message.content}")
print("-----------------------")
