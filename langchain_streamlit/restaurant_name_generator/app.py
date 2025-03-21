from typing import List, LiteralString, Dict
import streamlit as st

# import generator
import generator_details as generator

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
    st.subheader(body="*** Restaurant Name ***")
    st.markdown(body=response["restaurant_name"].strip())
    st.subheader(body="*** Menu Items ***")
    st.markdown(body=response["menu_items"].strip())

    # items: List[str] = [item.strip() for item in response["menu_items"].split(sep=",")]
    # for item in items:
    #     st.markdown(body=f"- {item}")
