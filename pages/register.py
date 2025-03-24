import streamlit as st
import cv2
import numpy as np
import pickle
import sqlite3
import re
from src.components.main import detect_face
from src.components.utils import generate_embedding, insert_user, check_user_exists

# Set the header for the registration page
st.header("Register")

def register_user(file):
    # Get the user's name from input and format it
    name = st.text_input("Enter your name:").strip().title()

    # Validate the name input
    if not name or not re.match(r"^[a-zA-Z\s]+$", name):
        st.warning("Please enter a valid name using only letters.")
        return
    if not file:
        st.warning("Please capture your face.")
        return

    # Convert the uploaded file to an image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Invalid image. Please try again.")
        return
    
    # Display the captured image
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Captured Image", width=240)

    try:
        # Connect to the SQLite database
        with sqlite3.connect(r"E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db") as conn:
            # Check if the user already exists
            if check_user_exists(conn, name):
                st.error(f"User '{name}' already exists. Please use a different name or contact the developer.")
                return
            
            # Detect the face in the image
            face, _ = detect_face(image)
            if face is not None:
                st.image(face, caption="Detected Face")
            else:
                st.error("No face detected. Please try again.")
                return

            # Generate the face embedding
            embedding = generate_embedding(face)
            if embedding is None:
                st.error("Failed to generate embedding.")
                return

            # Serialize the embedding to store in the database
            embedding_blob = pickle.dumps(embedding)

            # Insert the new user into the database
            insert_user(conn, name, embedding_blob)
            st.success("Registration successful!")
    except sqlite3.IntegrityError:
        st.error("User already exists. Please use a different name.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sidebar options for image upload
page = st.sidebar.radio("Choose upload option", ["Choose from file", "Upload from camera"])

if page == "Choose from file":
    st.write("Upload your image with clearly visible face.")
    # File uploader for selecting an image file
    image = st.file_uploader("Choose an image", type=["jpg", "png"])
    if image:
        register_user(image)

if page == "Upload from camera":
    st.write("Ensure your face is clearly visible.")
    # Camera input for capturing an image
    camera_file = st.camera_input("Capture Image")
    if camera_file:
        register_user(camera_file)
