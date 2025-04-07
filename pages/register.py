import streamlit as st
import cv2
import pickle
import sqlite3
from src.components.utils import insert_user, check_user_exists, validate_name, convert_to_image, detect_and_embed
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Path to the SQLite database
DB_PATH = os.getenv("DB_PATH", "face_recognition.db")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def register_user(username, file):
    """
    Register a new user by capturing their face and storing their details in the database.
    
    Args:
        file: The uploaded file containing the user's face image.
    """
    col1, col2 = st.columns([1, 1], gap="small", vertical_alignment="center")
    # Convert the uploaded file to an image
    image = convert_to_image(file)
    if image is None:
        return False
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=240)

    # Get and validate the user's name
    name = username.strip().title()
    if not name:
        st.warning("Please enter your name.")
        return False
    if not validate_name(name):
        return False
    if not file:
        st.warning("Please capture your face.")
        return False
    
    # Connect to the SQLite database
    with sqlite3.connect(DB_PATH) as conn:
        # Check if the user already exists
        if check_user_exists(conn, name):
            st.error(f"🤦‍♀️ User '{name}' already exists. Please use a different name or contact the developer.")
            return False

    with col2:
        # Detect the face and generate its embedding
        embedding = detect_and_embed(image)
    if embedding is None:
        st.error("Embedding generation failed.")
        return False

    try:
        # Serialize the embedding and insert the new user into the database
        embedding_blob = pickle.dumps(embedding)
        insert_user(conn, name, embedding_blob)
        st.session_state.authenticated = True
        st.session_state.username = name
        return True
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return False

# Set the header for the registration page
st.title("Register")

st.write("Ensure your face is clearly visible.")

# Sidebar options for image upload
page = st.sidebar.radio("Select registration option", ["Choose image from file", "Upload from camera"])

col1, col2 = st.columns([1, 1], vertical_alignment="top", gap="medium")

if page == "Choose image from file":
    with col1:
        # File uploader for selecting an image file
        image_file = st.file_uploader("*Choose an image", type=["jpg", "png"], key="image_file")
        name_input = st.text_input("*Enter your name:", max_chars=50, key="name_image")
    submit_btn = st.button("Register", icon="📝", type="primary", use_container_width=True)
    if submit_btn:
        if image_file:
            if name_input:
                with col2:
                    status_placeholder = st.empty()
                    
                    with status_placeholder:
                        st.write("🔄 Registering...")
                    
                    success = register_user(name_input, image_file)

                    if success:
                        status_placeholder.success("✅ Registration Successful!")
                        countdown_placeholder = st.empty()
                        st.write("")
                        for i in range(5, 0, -1):
                            countdown_placeholder.write(f"⏳ Redirecting in {i} seconds...")
                            time.sleep(1)

                        st.switch_page("pages/landing.py")
            else:
                st.warning("⚠️ Please enter your name before registering.")
        else:
            st.warning("⚠️ Please upload an image before registering.")

if page == "Upload from camera":
    with col1:
        # Camera input for capturing an image
        camera_file = st.camera_input("*Capture Image", key="camera_file")
        name_input = st.text_input("*Enter your name:", max_chars=50, key="name_camera")
    submit_btn = st.button("Register", icon="📝", type="primary", use_container_width=True)
    if submit_btn:
        if camera_file:
            if name_input:
                with col2:
                    st.write("")
                    st.write("")
                    
                    status_placeholder = st.empty()
                    with status_placeholder:
                        st.write("🔄 Registering...")
                    
                    success = register_user(name_input, camera_file)

                    if success:
                        status_placeholder.success("✅ Registration Successful!")
                        st.write("")
                        countdown_placeholder = st.empty()
                        st.write("")
                        for i in range(5, 0, -1):
                            countdown_placeholder.write(f"⏳ Redirecting in {i} seconds...")
                            time.sleep(1)

                        st.switch_page("pages/landing.py")
            else:
                st.warning("⚠️ Please enter your name before registering.")
        else:
            st.warning("⚠️ Please capture an image before registering.")
