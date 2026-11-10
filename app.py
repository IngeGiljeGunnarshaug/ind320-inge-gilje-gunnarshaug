import streamlit as st

# Set the browser-tab title and page layout.
st.set_page_config(page_title="IND320 - Energy Data", layout="wide")

# Register each page from its separate Python file.
pages = [
    st.Page("views/home.py", title="Home", default=True),
    st.Page("views/data.py", title="Reservoir Data"),
    st.Page("views/four.py", title="ENTSO-E Data"),
]

# Global geographical area selection.
st.sidebar.subheader("Geographical Area")

area_type = st.sidebar.radio(
    "Area type",
    options=["EL", "NO", "VASS"],
    key="global_area_type",
    horizontal=True
)

# Show sidebar navigation and run the selected page.
navigation = st.navigation(pages, position="sidebar")
navigation.run()