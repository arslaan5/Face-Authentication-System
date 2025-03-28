import streamlit as st
import numpy as np
import face_recognition
from src.components.utils import retrieve_all_users, convert_to_image, detect_and_embed
import time

# Initialize session state for login status
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def authenticate_user(embedding):
    try:
        users = retrieve_all_users()
        known_names = [user['name'] for user in users]
        known_embeddings = [np.array(user['face_encoding']).flatten() for user in users]

        if len(known_embeddings) == 0 or len(known_names) == 0:
            st.error("No registered users found. Please register first.")
            return False
        
        distances = face_recognition.face_distance(known_embeddings, embedding)
        min_distance = np.min(distances)
        threshold = 0.7

        if min_distance < threshold:
            best_match_index = np.argmin(distances)
            st.session_state.authenticated = True
            st.session_state.username = known_names[best_match_index]
            # Display success message
            st.toast(f"🎉 Signed in as {known_names[best_match_index]}!")
            return True
        else:
            st.error("Face not recognized. Please try again or register.")
            return False
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return False

def login_user(file):
    image = convert_to_image(file)
    if image is None:
        return False

    embedding = detect_and_embed(image)
    if embedding is None:
        return False

    return authenticate_user(embedding)


st.title("Login")

st.write("Ensure your face is clearly visible.")
# Camera input for capturing an image
col1, col2 = st.columns([1, 1])

with col1:
    camera_file = st.camera_input("*Capture Image")
submit_btn = st.button("Login", icon="🔑", type="primary", use_container_width=True)
if submit_btn:
    if camera_file:
        with col2:
            st.write("")
            st.write("")
            status_placeholder = st.empty()
            with status_placeholder:
                st.write("🔄 Authenticating...")

            success = login_user(camera_file)

            if success:
                status_placeholder.success("✅ Authentication Successful!")
                st.write("")
                countdown_placeholder = st.empty()
                st.write("")
                for i in range(3, 0, -1):
                    countdown_placeholder.write(f"⏳ Redirecting in {i} seconds...")
                    time.sleep(1)

                st.switch_page("pages/landing.py")

            else:
                status_placeholder.error("❌ Authentication Failed. Please try again.")
    else:
        st.warning("⚠️ Please capture an image before logging in.")
