import streamlit as st

# Set the browser-tab title and page layout.
st.set_page_config(page_title="IND320 - Reservoir Data", layout="wide")

# Register each page from its separate Python file.
pages = [
    st.Page("views/home.py", title="Home", default=True),
    st.Page("views/data.py", title="Data"),
    st.Page("views/plots.py", title="Plots"),
]

# Show sidebar navigation and run the selected page.
navigation = st.navigation(pages, position="sidebar")
navigation.run()