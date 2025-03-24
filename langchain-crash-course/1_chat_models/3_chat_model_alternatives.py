from typing import Any
from langchain_core.messages.base import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv


# Setup environment variables and messages
load_dotenv()

messages: list[Any] = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="divided 81 by 9?"),
]

# ---- LangChain OpenAI Chat Model Example ----

# Create a ChatOpenAI model
openai_model = ChatOpenAI(model="gpt-4o-mini")
# Invoke the model with messages
result = openai_model.invoke(input=messages)
print(f"Answer from OpenAI: {result.content}")

# ---- Anthropic Chat Model Example ----

# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
# model = ChatAnthropic(model="claude-3-opus-20240229")
anthropic_model = ChatAnthropic(
    model_name="Claude-3-5-Sonnet-20241022", timeout=None, stop=None
)
result: BaseMessage = anthropic_model.invoke(input=messages)
print(f"Answer from Anthropic: {result.content}")

# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
result: BaseMessage = gemini_model.invoke(input=messages)
print(f"Answer from Google: {result.content}")
