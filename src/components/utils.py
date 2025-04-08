import sqlite3
import os
import pickle
import face_recognition
import cv2 as cv
import numpy as np
from datetime import datetime
import streamlit as st
import re
from src.components.create_db import DB_PATH


def insert_user(conn, name, face_encoding):
    """Insert a new user into the database with encodings

    Args:
        conn (SQLite3 conn): Connection to the SQLite database
        name (str): Name of the user
        face_encoding (List): Face encoding of the user
    """
    
    try:
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO users (name, face_encoding, registration_date)
        VALUES (?, ?, ?)
        ''', (name.strip().title(), face_encoding, datetime.now().strftime("%d-%m-%Y %H:%M:%S")))

        conn.commit()
        print(f"User '{name.strip().title()}' added successfully!")

    except sqlite3.Error as e:
        print("SQLite error:", e)

def retrieve_all_users():
    """Retrieve all users from the database.

    Returns:
        List: List of dictionaries of users with name and face encoding
    """

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT name, face_encoding FROM users')
    rows = cursor.fetchall()

    users = []
    for row in rows:
        name, face_encoding_blob = row
        face_encoding = pickle.loads(face_encoding_blob)
        users.append({'name': name, 'face_encoding': face_encoding})

    conn.close()

    return users

def generate_embedding(img):
    """Generate face embeddings

    Args:
        img (str or np.ndarray): Path to the image file or Numpy array of the image

    Returns:
        List: Face embedding of the image
    """

    if isinstance(img, str):
        if not os.path.isfile(img):
            raise FileNotFoundError(f"Image file '{img}' not found!")
        image = cv.imread(img)
    elif isinstance(img, np.ndarray):
        image = img
    else:
        raise ValueError("Invalid input type. Please provide a filepath or a NumPy array.")

    if image is None or len(image.shape) < 2:
        raise ValueError("Invalid image. Please provide a valid image file or array.")
    
    if image.shape[-1] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[-1] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    if image.dtype != np.uint8:
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    embedding = face_recognition.face_encodings(image, num_jitters=2)
    
    if not embedding:
        raise ValueError("No face detected in the image.")
    
    return embedding[0]

def add_user(name, image_path):
    """Add a new user to the database using image file.

    Args:
        name (str): Name of the user
        image_path (str or filepath): Path to the image file
    """

    face_encoding = generate_embedding(image_path)

    face_encoding_blob = pickle.dumps(face_encoding)

    conn = sqlite3.connect(DB_PATH)
    
    insert_user(conn, name, face_encoding_blob)

    conn.close()

def check_user_exists(conn, name):
    """Check if a user with the given name already exists in the database."""
    name = name.strip().title()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users WHERE name = ?", (name,))
    return cursor.fetchone()[0] > 0

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
        image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        if image is None:
            st.error("Invalid image. Please try again.")
            return None
        return image
    except Exception as e:
        st.error("Please ensure you are uploading/capturing an image file.")
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

def detect_face(image):
    """
    Detect face in an image using the face_recognition library.
    
    Args:
        image (str or np.ndarray): Path to the image file or a NumPy array.

    Returns:
        tuple: (Face region as NumPy array, Bounding box coordinates as (x, y, width, height))
               Returns (None, None) if no face is detected.
    """
    if isinstance(image, str):  # If input is a file path
        if not cv.os.path.isfile(image):
            raise FileNotFoundError(f"Image file '{image}' not found.")
        image = face_recognition.load_image_file(image)
    elif not isinstance(image, np.ndarray):
        raise ValueError("Input must be a file path or a NumPy array.")
    
    if image is None or len(image.shape) < 2:
        raise ValueError("Invalid image. Please provide a valid image file or NumPy array.")
    
    if image.ndim == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    if image.dtype != np.uint8:
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2)

    if not face_locations:
        print("No face detected.")
        return None, None

    top, right, bottom, left = face_locations[0]
    face = image[top:bottom, left:right]

    cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    face = cv.resize(face, (160, 160))
    face = face.astype(np.float32) / 255.0
    # face = np.expand_dims(face, axis=0)  # Add batch dimension

    return face, (left, top, right - left, bottom - top)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    # create_database(conn)
    # # Example: Add a user
    # add_user("arslaan siddiqui", r"E:\Face-Recognition-for-Login-Authentication-System\assets\photo.jpg")
    # add_user("Shah Rukh Khan", r"E:\Face-Recognition-for-Login-Authentication-System\assets\shahrukh.jpg")
    # add_user("Hrithik Roshan", r"E:\Face-Recognition-for-Login-Authentication-System\assets\hrithik.jpg")
    # add_user("Alia Bhatt", r"E:\Face-Recognition-for-Login-Authentication-System\assets\alia.png")

    # # Example: Retrieve all users
    # users = retrieve_all_users()
    # for user in users:
    #     print(f"User: {user['name']}, Encoding: {user['face_encoding']}")

    # embedding1 = generate_embedding(r"E:\Face-Recognition-for-Login-Authentication-System\assets\alia.png")
    # print(embedding1)

    # embedding2 = generate_embedding(r"E:\Face-Recognition-for-Login-Authentication-System\assets\photo.jpg")
    # embedding2 = generate_embedding(0.3487)
    # print(embedding2)

    # check_user_exists = check_user_exists(conn, "Arslaan siddiqui")
    # print(check_user_exists)