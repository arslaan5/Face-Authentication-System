import sqlite3
import os
import pickle
import face_recognition
import cv2 as cv
import numpy as np


def create_database(conn):
    """Create the SQLite database and users table if they do not exist

    Args:
        conn (SQLite3 conn): Connection to the SQLite database
    """
    
    db_path = os.path.abspath(r"E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db")
    print("Database path:", db_path)  # Debugging step

    try:
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            face_encoding BLOB NOT NULL
        )
        ''')

        conn.commit()
        print("Database and table created successfully or already exists!")

    except sqlite3.Error as e:
        print("SQLite error:", e)


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
        INSERT INTO users (name, face_encoding)
        VALUES (?, ?)
        ''', (name, face_encoding))

        conn.commit()
        print(f"User '{name}' added successfully!")

    except sqlite3.Error as e:
        print("SQLite error:", e)


def retrieve_all_users():
    """Retrieve all users from the database.

    Returns:
        List: List of dictionaries of users with name and face encoding
    """

    # Connect to the database
    conn = sqlite3.connect(r"E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db")
    cursor = conn.cursor()

    # Retrieve all users
    cursor.execute('SELECT name, face_encoding FROM users')
    rows = cursor.fetchall()

    # Convert face encodings back to numpy arrays
    users = []
    for row in rows:
        name, face_encoding_blob = row
        face_encoding = pickle.loads(face_encoding_blob)
        users.append({'name': name, 'face_encoding': face_encoding})
    
    # Close the connection
    conn.close()

    return users


def generate_embedding(img):
    """Generate face embeddings

    Args:
        img (str or NumPy array): Path to the image file or Numpy array of the image

    Returns:
        List: Face embedding of the image
    """

    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file '{img}' not found!")
        image = cv.imread(img)
    elif isinstance(img, np.ndarray):
        image = img
    else:
        raise ValueError("Invalid input type. Please provide a filepath or a NumPy array.")
    
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    embedding = face_recognition.face_encodings(image, num_jitters=3)
    
    return embedding[0]


def add_user(name, image_path):
    """Add a new user to the database using image file.

    Args:
        name (str): Name of the user
        image_path (str or filepath): Path to the image file
    """

    # Encode the face
    face_encoding = generate_embedding(image_path)

    # Convert the encoding to a binary format for storage
    face_encoding_blob = pickle.dumps(face_encoding)

    # Connect to the database
    conn = sqlite3.connect(r"E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db")
    
    # Insert the user data and face encoding into the database
    insert_user(conn, name, face_encoding_blob)

    conn.close()


if __name__ == "__main__":
    # Example: Add a user
    # add_user("Arslaan Siddiqui", r"E:\Face-Recognition-for-Login-Authentication-System\assets\photo.jpg")
    # add_user("Shah Rukh Khan", r"E:\Face-Recognition-for-Login-Authentication-System\assets\shahrukh.jpg")
    # add_user("Hrithik Roshan", r"E:\Face-Recognition-for-Login-Authentication-System\assets\hrithik.jpg")
    # add_user("Alia Bhatt", r"E:\Face-Recognition-for-Login-Authentication-System\assets\alia.png")

    # Example: Retrieve all users
    # retrieve_all_users()

    # embedding1 = generate_embedding(r"E:\Face-Recognition-for-Login-Authentication-System\assets\alia.png")
    # print(embedding1)
    # embedding2 = generate_embedding(r"E:\Face-Recognition-for-Login-Authentication-System\assets\photo.jpg")
    embedding2 = generate_embedding(0.3487)
    print(embedding2)
