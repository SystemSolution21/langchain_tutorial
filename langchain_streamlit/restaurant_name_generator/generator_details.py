from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable, RunnableLambda

# Create Chat Model
model: list[str] = [
    "llama3.2:3b",
    "gemma3:4b",
    "openthinker:7b",
    "deepseek-r1:14b",
]
llm = ChatOllama(model=model[0])


def generate_restaurant_name_menu_item(cuisine: str) -> Dict[str, str]:

    # Chain 1: Restaurant Name Generation
    restaurant_name_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                "You are a creative restaurant naming expert. Provide List of restaurant name with it's precise explanation text.",
            ),
            (
                "user",
                "Suggest a fancy restaurant name for {cuisine} food.",
            ),
        ]
    )

    restaurant_name_chain: RunnableSerializable[Dict[str, str], str] = (
        restaurant_name_prompt | llm | StrOutputParser()
    )

    # Chain 2: Menu Items Generation
    menu_item_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        messages=[
            (
                "system",
                "You are an expert chef. List signature dishes with it's precise explanation text.",
            ),
            (
                "user",
                "Suggest signature menu items for the {cuisine} restaurant.",
            ),
        ]
    )

    menu_item_chain: RunnableSerializable[Dict[str, str], str] = (
        menu_item_prompt | llm | StrOutputParser()
    )

    # Define the chain steps
    def generate_restaurant_name(input_dict: Dict) -> Dict:
        restaurant_name: str = restaurant_name_chain.invoke(
            input={"cuisine": input_dict["cuisine"]}
        )
        return {"cuisine": input_dict["cuisine"], "restaurant_name": restaurant_name}

    def generate_menu_item(input_dict: Dict) -> Dict:
        menu_items: str = menu_item_chain.invoke(
            input={
                "restaurant_name": input_dict["restaurant_name"],
                "cuisine": input_dict["cuisine"],
            }
        )
        return {
            "restaurant_name": input_dict["restaurant_name"],
            "menu_items": menu_items,
        }

    # Create the final chain
    chain: RunnableSerializable[str, Dict[str, str]] = (
        RunnableLambda(func=lambda x: {"cuisine": x})
        | RunnableLambda(func=generate_restaurant_name)
        | RunnableLambda(func=generate_menu_item)
    )

    return chain.invoke(input=cuisine)


if __name__ == "__main__":
    result: Dict[str, str] = generate_restaurant_name_menu_item(cuisine="Italian")
    print("\n\nRestaurant Name:")
    print("-" * 50)
    print(result["restaurant_name"].strip())
    print("\n\nSignature Menu Items:")
    print("-" * 50)
    print(result["menu_items"].strip())
