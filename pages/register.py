import streamlit as st
import cv2
import numpy as np
import pickle
import sqlite3
import re
from src.components.main import detect_face
from src.components.utils import generate_embedding, insert_user, check_user_exists
import os

# Path to the SQLite database
DB_PATH = os.path.abspath(r"E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db")

# Set the header for the registration page
st.header("Register")

def validate_name(name):
    """
    Validate the user's name to ensure it contains only letters and spaces.
    
    Args:
        name (str): The name entered by the user.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    if not name or not re.match(r"^[a-zA-Z\s]+$", name):
        st.warning("Please enter a valid name using only letters.")
        return False
    return True

def convert_to_image(file):
    """
    Convert the uploaded file to an OpenCV image.
    
    Args:
        file: The uploaded file.

    Returns:
        np.ndarray: The converted image, or None if the conversion fails.
    """
    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Invalid image. Please try again.")
            return None
        return image
    except Exception as e:
        st.error(f"Image processing failed: {e}")
        return None

def detect_and_embed(image):
    """
    Detect a face in the image and generate its embedding.
    
    Args:
        image (np.ndarray): The image in which to detect the face.

    Returns:
        np.ndarray: The face embedding, or None if detection or embedding generation fails.
    """
    face, _ = detect_face(image)
    if face is None:
        st.error("No face detected. Please try again.")
        return None
    
    try:
        embedding = generate_embedding(face)
        st.image(face, caption="Detected Face")
        return embedding
    except (FileNotFoundError, ValueError) as e:
        st.error(f"❗ {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

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
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Captured Image", width=240)

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
