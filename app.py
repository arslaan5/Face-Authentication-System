import streamlit as st

st.set_page_config(
    page_title="Face Authentication System",
    page_icon="🎭",
    layout="wide"
)

login_page = st.Page("pages/login.py", title="Login", icon="👍")
register_page = st.Page("pages/register.py", title="Register", icon="🎁")
landing_page = st.Page("pages/landing.py", title="Home", icon="💖")
logout_page = st.Page("pages/logout.py", title="Logout", icon="🔒")

# Show logout option only if user is authenticated
if st.session_state.authenticated:
    pages = [landing_page, logout_page]
else:
    pages = [landing_page, login_page, register_page]

# Display navigation
pg = st.navigation(pages)

# Show username in sidebar if authenticated
if st.session_state.authenticated:
    st.sidebar.write(f"Logged in as {st.session_state.username}")

pg.run()