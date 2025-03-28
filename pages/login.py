import streamlit as st
import os
import numpy as np
import face_recognition
from src.components.utils import retrieve_all_users, convert_to_image, detect_and_embed
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DB_PATH")

def authenticate_user(embedding):
    try:
        users = retrieve_all_users()
        known_names = [user['name'] for user in users]
        known_embeddings = [np.array(user['face_encoding']).flatten() for user in users]

        distances = face_recognition.face_distance(known_embeddings, embedding)
        min_distance = np.min(distances)
        threshold = 0.7

        if min_distance < threshold:
            best_match_index = np.argmin(distances)
            return st.success(f"🎉 Signed in as {known_names[best_match_index]}!")
        else:
            st.error("Face not recognized. Please try again or register.")
    except Exception as e:
        st.error(f"Authentication failed: {e}")

def login_user(file):
    image = convert_to_image(file)
    if image is None:
        return

    embedding = detect_and_embed(image)
    if embedding is None:
        return

    authenticate_user(embedding)


st.title("Login")

st.write("Ensure your face is clearly visible.")
# Camera input for capturing an image
camera_file = st.camera_input("*Capture Image")
submit_btn = st.button("Login", icon="🔑", type="primary", use_container_width=True)
if submit_btn:
    if camera_file:
        with st.spinner("Authenticating...", show_time=True):
            login_user(camera_file)
    else:
        st.warning("⚠️ Please capture an image before logging in.")