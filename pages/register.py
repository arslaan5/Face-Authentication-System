import streamlit as st
import cv2
import pickle
import sqlite3
from src.components.utils import insert_user, check_user_exists, validate_name, convert_to_image, detect_and_embed
import os

# Path to the SQLite database
DB_PATH = os.path.abspath(r"E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db")

# Set the header for the registration page
st.header("Register")

def register_user(file):
    """
    Register a new user by capturing their face and storing their details in the database.
    
    Args:
        file: The uploaded file containing the user's face image.
    """
    # Get and validate the user's name
    name = st.text_input("Enter your name: *").strip().title()
    if not validate_name(name):
        st.warning("Please enter your name.")
        return
    if not file:
        st.warning("Please capture your face.")
        return

    # Convert the uploaded file to an image
    image = convert_to_image(file)
    if image is None:
        return
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=240)

    # Detect the face and generate its embedding
    embedding = detect_and_embed(image)
    if embedding is None:
        st.error("Embedding generation failed.")
        return

    try:
        # Connect to the SQLite database
        with sqlite3.connect(DB_PATH) as conn:
            # Check if the user already exists
            if check_user_exists(conn, name):
                st.error(f"🤦‍♀️ User '{name}' already exists. Please use a different name or contact the developer.")
                return
            
            # Serialize the embedding and insert the new user into the database
            embedding_blob = pickle.dumps(embedding)
            insert_user(conn, name, embedding_blob)
            st.success("✅ Registration successful!")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sidebar options for image upload
page = st.sidebar.radio("Select registration option", ["Choose image from file", "Upload from camera"])

if page == "Choose image from file":
    st.write("Upload your image with clearly visible face.")
    # File uploader for selecting an image file
    image = st.file_uploader("Choose an image", type=["jpg", "png"])
    if image:
        register_user(image)

if page == "Upload from camera":
    st.write("Ensure your face is clearly visible.")
    # Camera input for capturing an image
    camera_file = st.camera_input("Capture Image *")
    if camera_file:
        register_user(camera_file)
