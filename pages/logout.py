import streamlit as st

if "authenticated" in st.session_state:
    st.session_state.authenticated = False
    st.switch_page("pages/landing.py")