"""Retrieve and print chat history from Firebase Firestore."""

from typing import Any, List, Dict
import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path
from datetime import datetime
from google.cloud.firestore_v1.client import Client

# Initialize Firebase
file_path: Path = Path(__file__).parent.resolve()
cred = credentials.Certificate(cert=file_path / "firebase_config.json")
firebase_admin.initialize_app(credential=cred)
db: Client = firestore.client()


def get_chat_history(conversation_id: str) -> List[Dict]:
    """Retrieve chat history from Firestore."""
    messages: Any = (
        db.collection("conversations")
        .document(document_id=conversation_id)
        .collection("messages")
        .order_by("timestamp")
        .stream()
    )
    return [message.to_dict() for message in messages]


def format_timestamp(timestamp: datetime) -> str:
    """Convert Firestore timestamp to readable format."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def print_chat_history(chat_history: List[Dict]) -> None:
    """Print chat history in a readable format."""
    print("\n=== Chat History ===")
    print(f"Total Messages: {len(chat_history)}\n")

    for i, message in enumerate(iterable=chat_history, start=1):
        print(f"Message {i}:")
        print(f"Type: {message['type']}")
        print(f"Time: {format_timestamp(timestamp=message['timestamp'])}")
        print(f"Content: {message['content']}")
        print("-" * 50)


def main() -> None:
    """Main function to retrieve and print chat history."""
    conversation_id = "20250323_142913"  # conversation_id from firebase console
    conversation_id = "20250323_143136"  # conversation_id from firebase console
    conversation_id = "20250324_184412"  # conversation_id from firebase console
    chat_history: List[Dict] = get_chat_history(conversation_id=conversation_id)
    print_chat_history(chat_history=chat_history)


if __name__ == "__main__":
    # Run the main function
    main()
