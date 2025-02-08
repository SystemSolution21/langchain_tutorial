from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue

# 1. Basic String Prompt Template
template: str = "Tell me a joke about {topic}."
prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(
    template=template
)
prompt: PromptValue = prompt_template.invoke(input={"topic": "cats"})
print("-----Basic string prompt template-----")
print(prompt)

# 2. Prompt with placeholders
template: str = """You are a helpful Assistant.
Human: Tell me a {adjective} joke about {animal}.
Assistant:"""
prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(
    template=template
)
prompt = prompt_template.invoke(input={"adjective": "funny", "animal": "cats"})
print("-----Prompt template placeholders")
print(prompt)
