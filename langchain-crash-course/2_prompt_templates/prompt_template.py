from dotenv import load_dotenv
from langchain_core.messages.base import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Load Environment Variables
load_dotenv()

# Create OpenAI Chat Model
openai_model = ChatOpenAI(model="gpt-4.1-nano")

# 1. String Prompt Template
prompt_template: PromptTemplate = PromptTemplate.from_template(
    template="Tell me a joke about {topic}"
)
print("\n-----String Prompt Template-----")
print(prompt_template.format(topic="cats"))
prompt: PromptValue = prompt_template.invoke(input={"topic": "cats"})
print(prompt.to_string())

# 2. Chat Prompt Template
template: str = "Tell me a joke about {topic}."
chat_prom_temp: ChatPromptTemplate = ChatPromptTemplate.from_template(template=template)
prompt = chat_prom_temp.invoke(input={"topic": "cats"})
print("\n-----Chat Prompt Template-----")
print(prompt.to_string())

# 3. Chat Prompt Template with placeholders
template: str = """You are a helpful Assistant.
Human: Tell me a {adjective} joke about {animal}.
Assistant:"""
chat_prom_temp_plac: ChatPromptTemplate = ChatPromptTemplate.from_template(
    template=template
)
prompt = chat_prom_temp_plac.invoke(input={"adjective": "funny", "animal": "cats"})
print("\n-----Chat Prompt template placeholders-----")
print(prompt.to_string())

# 4. Chat Prompt Template with System and Human messages
messages: list[tuple[str, str]] = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
chat_prom_temp_mess: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=messages
)
prompt = chat_prom_temp_mess.invoke(input={"topic": "cats", "joke_count": 1})
print("\n-----Chat Prompt template with System and Human messages-----")
print(prompt.to_string())

# # Call OPenAI Chat Model
# openai_response: BaseMessage = openai_model.invoke(input=prompt)
# print(openai_response.content)

# 5. Create Llama Chat Model
llm = ChatOllama(model="llama3.2:3b", temperature=0.8, num_predict=256)
messages = [
    (
        "system",
        "Your are helpful translator. Translate the user sentence to {language}.",
    ),
    ("human", "{text}"),
]
chat_prom_temp_mess: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    messages=messages
)
language: str = "Japanese"
text: str = "I like programming."
prompt = chat_prom_temp_mess.invoke(input={"language": language, "text": text})
print(
    "\n-----Chat prompt template with system and human messages on Langchain_Ollama model-----"
)
print(prompt.to_string())

# Call Llama Chat Model
response: BaseMessage = llm.invoke(input=prompt)
print(f"AI: {response.content}")
