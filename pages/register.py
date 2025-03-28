import streamlit as st
import cv2
import pickle
import sqlite3
from src.components.utils import insert_user, check_user_exists, validate_name, convert_to_image, detect_and_embed
import os
from dotenv import load_dotenv

load_dotenv()

# Path to the SQLite database
DB_PATH = os.getenv("DB_PATH")

def register_user(username, file):
    """
    Register a new user by capturing their face and storing their details in the database.
    
    Args:
        file: The uploaded file containing the user's face image.
    """
    # Convert the uploaded file to an image
    image = convert_to_image(file)
    if image is None:
        return
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=240)

    # Get and validate the user's name
    name = username.strip().title()
    if not name:
        st.warning("Please enter your name.")
        return
    if not validate_name(name):
        return
    if not file:
        st.warning("Please capture your face.")
        return
    
    # Connect to the SQLite database
    with sqlite3.connect(DB_PATH) as conn:
        # Check if the user already exists
        if check_user_exists(conn, name):
            st.error(f"🤦‍♀️ User '{name}' already exists. Please use a different name or contact the developer.")
            return

    # Detect the face and generate its embedding
    embedding = detect_and_embed(image)
    if embedding is None:
        st.error("Embedding generation failed.")
        return

    try:
        # Serialize the embedding and insert the new user into the database
        embedding_blob = pickle.dumps(embedding)
        insert_user(conn, name, embedding_blob)
        st.success("✅ Registration successful! You can now log in.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Set the header for the registration page
st.header("Register")

# Sidebar options for image upload
page = st.sidebar.radio("Select registration option", ["Choose image from file", "Upload from camera"])

if page == "Choose image from file":
    st.write("Upload your image with clearly visible face.")
    # File uploader for selecting an image file
    image_file = st.file_uploader("*Choose an image", type=["jpg", "png"])
    name_input = st.text_input("*Enter your name:", max_chars=50)
    submit_btn = st.button("Register", icon="📝", type="primary", use_container_width=True)
    if submit_btn:
        if image_file:
            if name_input:
                with st.spinner("Registering... Please wait.", show_time=True):
                    register_user(name_input, image_file)
            else:
                st.warning("⚠️ Please enter your name before registering.")
        else:
            st.warning("⚠️ Please upload an image before registering.")

if page == "Upload from camera":
    st.write("Ensure your face is clearly visible.")
    # Camera input for capturing an image
    camera_file = st.camera_input("*Capture Image")
    name_input = st.text_input("*Enter your name:", max_chars=50)
    submit_btn = st.button("Register", icon="📝", type="primary", use_container_width=True)
    if submit_btn:
        if camera_file:
            if name_input:
                with st.spinner("Registering... Please wait.", show_time=True):
                    register_user(name_input, camera_file)
            else:
                st.warning("⚠️ Please enter your name before registering.")
        else:
            st.warning("⚠️ Please capture an image before registering.")