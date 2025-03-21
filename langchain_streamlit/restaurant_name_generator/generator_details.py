from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda

# Create Chat Model
model: list[str] = [
    "llama3.2:3b",
    "gemma3:4b",
    "openthinker:7b",
    "deepseek-r1:14b",
]
llm = ChatOllama(model=model[0])


def generate_restaurant_name_menu_item(cuisine: str) -> Dict[str, Any]:
    # Chain 1: Restaurant Name Generation
    name_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a creative restaurant naming expert. Provide List of restaurant name with it's precise explanation text.",
            ),
            (
                "user",
                "I want to open a restaurant for {cuisine} food. Suggest a fancy name for it.",
            ),
        ]
    )

    name_chain = name_prompt | llm | StrOutputParser()

    # Chain 2: Menu Items Generation
    menu_prompt = ChatPromptTemplate.from_messages(
        [
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

    menu_chain = menu_prompt | llm | StrOutputParser()

    # Define the chain steps
    def generate_restaurant_name(input_dict: Dict) -> Dict:
        restaurant_name = name_chain.invoke({"cuisine": input_dict["cuisine"]})
        return {"cuisine": input_dict["cuisine"], "restaurant_name": restaurant_name}

    def generate_menu_item(input_dict: Dict) -> Dict:
        menu_items = menu_chain.invoke(
            {
                "restaurant_name": input_dict["restaurant_name"],
                "cuisine": input_dict["cuisine"],
            }
        )
        return {
            "restaurant_name": input_dict["restaurant_name"],
            "menu_items": menu_items,
        }

    # Create the final chain
    chain = (
        RunnableLambda(lambda x: {"cuisine": x})
        | RunnableLambda(generate_restaurant_name)
        | RunnableLambda(generate_menu_item)
    )

    return chain.invoke(cuisine)


if __name__ == "__main__":
    result = generate_restaurant_name_menu_item("Italian")
    print("\n\nRestaurant Name:")
    print("-" * 50)
    print(result["restaurant_name"].strip())
    print("\n\nSignature Menu Items:")
    print("-" * 50)
    print(result["menu_items"].strip())
