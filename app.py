import streamlit as st

st.set_page_config(
    page_title="Face Authentication System",
    page_icon="🎭",
    layout="wide"
)

login_page = st.Page("pages/login.py", title="Login", icon="👍")
register_page = st.Page("pages/register.py", title="Register", icon="🎁")
landing_page = st.Page("pages/landing.py", title="Home", icon="💖")

pg = st.navigation([landing_page, login_page, register_page])
pg.run()

st.sidebar.write("Hey!")