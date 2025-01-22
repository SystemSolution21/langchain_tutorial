from typing import Any, List
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage
from langchain_openai import ChatOpenAI

# Load Environment Variables
load_dotenv()

# Create ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

"""
Set Messages
    SystemMessage:
        {"role": "system", "content": "message for priming AI behavior"},
    HumanMessage:
        {"role": "user", "content": "message from human to AI"},
    AIMessage:
        {"role": "system", "content": "message for priming AI behavior"}
"""
messages: List[Any] = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="Divide 81 by 9?"),
    AIMessage(content="81 divide by 9 equal 9"),
    HumanMessage(content="10 times 5?"),
]

# Invoke the model with messages
result: BaseMessage = model.invoke(input=messages)
print(result.content)
