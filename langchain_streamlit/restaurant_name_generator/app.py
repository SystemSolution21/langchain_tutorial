import time
from typing import Any, Dict, Generator, LiteralString

# import generator_custom as generator
import generator_details as generator
import streamlit as st

st.set_page_config(
    page_title="Restaurant Name Menu Item Generator", page_icon=":fork_and_knife:"
)
st.title(body="Restaurant Name Menu Item Generator")

cuisine: LiteralString = st.sidebar.selectbox(
    label="Pick a Cuisine",
    options=("Japanese", "Italian", "Mexican", "Chinese", "American", "Indian"),
)

if cuisine:
    response: Dict[str, str] = generator.generate_restaurant_name_menu_item(
        cuisine=cuisine
    )
    # generator_details
    # Restaurant Name Header
    st.subheader(body="*** Restaurant Name ***")

    # StreamRestaurant Name
    def restaurant_name() -> Generator[str, Any, None]:
        for word in response["restaurant_name"].strip().split(sep=" "):
            yield word + " "
            time.sleep(0.02)

    st.write_stream(stream=restaurant_name())

    # Menu Items Header
    st.subheader(body="*** Menu Items ***")

    # Stream Menu Items
    def menu_items() -> Generator[str, Any, None]:
        for word in response["menu_items"].strip().split(sep=" "):
            yield word + " "
            time.sleep(0.02)

    st.write_stream(stream=menu_items())

    # # generator_custom
    # st.subheader(body="*** Restaurant Name ***")
    # st.markdown(body=response["restaurant_name"].strip())
    # st.subheader(body="*** Menu Items ***")
    # items: List[str] = [item.strip() for item in response["menu_items"].split(sep=",")]
    # for item in items:
    #     st.markdown(body=f"- {item}")
