import streamlit as st
import re
import os
import cv2
import numpy as np
import face_recognition
from src.components.main import detect_face
from src.components.utils import generate_embedding, retrieve_all_users

st.title("Login")

DB_PATH = os.path.abspath(r"E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db")

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

def authenticate_user(name, embedding):
    try:
        users = retrieve_all_users()
        known_names = [user['name'] for user in users]
        known_embeddings = [np.array(user['face_encoding']).flatten() for user in users]

        distances = face_recognition.face_distance(known_embeddings, embedding)
        min_distance = np.min(distances)
        threshold = 0.7

        if min_distance < threshold:
            best_match_index = np.argmin(distances)
            return st.success(f"Logged in as {known_names[best_match_index]}!")
        else:
            st.error("Face not recognized. Please try again or register.")
    except Exception as e:
        st.error(f"Authentication failed: {e}")

def login_user(file):
    name = st.text_input("Enter your name:").strip().title()
    if not validate_name(name):
        st.warning("Please enter your name.")
        return
    if not file:
        st.warning("Please capture your face.")
        return

    image = convert_to_image(file)
    if image is None:
        return

    embedding = detect_and_embed(image)
    if embedding is None:
        return

    authenticate_user(name, embedding)


st.write("Ensure your face is clearly visible.")
# Camera input for capturing an image
camera_file = st.camera_input("Capture Image")
if camera_file:
    login_user(camera_file)