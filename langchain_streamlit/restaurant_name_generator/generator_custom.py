# generator_custom.py
"""
This script generates a restaurant name and menu items based on the given cuisine.
It uses LangChain to create a chain of thought process to generate the restaurant name and menu items.
"""

# Import standard libraries
import os

# Import langchain libraries
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable
from langchain_ollama import ChatOllama

# Create Chat Model
model: str = os.getenv(key="OLLAMA_LLM", default="llama3.2:latest")
llm = ChatOllama(model=model)


def generate_restaurant_name_menu_item(cuisine: str) -> dict[str, str]:
    """
    Generates a restaurant name and menu items based on the given cuisine.
    """
    # Chain 1: Restaurant Name Generation
    restaurant_name_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                "You are a creative restaurant naming expert. Provide ONLY ONE restaurant name without any explanation or additional text.",
            ),
            (
                "user",
                "Suggest a facny restaurant name for {cuisine} food.",
            ),
        ]
    )

    restaurant_name_chain: RunnableSerializable[dict[str, str], str] = (
        restaurant_name_prompt | llm | StrOutputParser()
    )

    # Chain 2: Menu Items Generation
    menu_item_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                "You are an expert chef. list ONLY 5 signature dishes as a comma-separated list without any additional text or explanation.",
            ),
            (
                "user",
                "Suggest 10 menu items for {restaurant_name}, a {cuisine} restaurant.",
            ),
        ]
    )

    menu_item_chain: RunnableSerializable[dict[str, str], str] = (
        menu_item_prompt | llm | StrOutputParser()
    )

    # Define the chain steps
    def generate_restaurant_name(user_input: dict) -> dict:
        restaurant_name: str = restaurant_name_chain.invoke(
            input={"cuisine": user_input["cuisine"]}
        )
        return {"cuisine": user_input["cuisine"], "restaurant_name": restaurant_name}

    def generate_menu_item(user_input: dict) -> dict:
        menu_items = menu_item_chain.invoke(
            {
                "restaurant_name": user_input["restaurant_name"],
                "cuisine": user_input["cuisine"],
            }
        )
        return {
            "restaurant_name": user_input["restaurant_name"],
            "menu_items": menu_items,
        }

    # Create the final chain
    chain: RunnableSerializable[str, dict[str, str]] = (
        RunnableLambda(func=lambda x: {"cuisine": x})
        | RunnableLambda(func=generate_restaurant_name)
        | RunnableLambda(func=generate_menu_item)
    )

    return chain.invoke(input=cuisine)


if __name__ == "__main__":
    result: dict[str, str] = generate_restaurant_name_menu_item(cuisine="Japanese")
    print("\nRestaurant Name:")
    print("-" * 50)
    print(result["restaurant_name"].strip())
    print("\nMenu Items:")
    print("-" * 50)
    items: list[str] = [item.strip() for item in result["menu_items"].split(sep=",")]
    for item in items:
        print(f"• {item}")
