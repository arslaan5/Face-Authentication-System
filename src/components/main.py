import face_recognition
import numpy as np
import cv2 as cv
import sqlite3
from .utils import create_database, insert_user, retrieve_all_users, generate_embedding, detect_face
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

# Path to the SQLite database
DB_PATH = os.getenv("DB_PATH")

def capture_faces():
    """
    Capture and save faces from webcam in real-time using face_recognition.
    Prompts the user to enter their name and captures their face when 's' is pressed.
    Saves the face embedding to the database.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)
    create_database(conn)

    # Open a connection to the webcam
    video_capture = cv.VideoCapture(0)

    print("Press 's' to capture your face and save.")
    print("Press 'q' to quit.")
    
    # Prompt the user to enter their name
    name = input("Enter your name: ").strip().title()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Detect face in the current frame
        face, bbox = detect_face(frame)

        if face is not None and bbox is not None:
            x, y, w, h = bbox
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv.imshow("Register Face", frame)

        print("Press 's' to save or 'q' to quit.")

        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            if face is not None and bbox is not None:
                x, y, w, h = bbox
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_image = frame[y:y + h, x:x + w]

                # Generate face embedding
                embedding = generate_embedding(face_image)

                if embedding is not None:
                    embedding_blob = pickle.dumps(embedding)

                    # Insert the user into the database
                    insert_user(conn, name, embedding_blob)
                    break
                else:
                    print("Failed to generate embedding. Please try again.")
            else:
                print("No face detected. Please try again.")

        elif key == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()

    conn.close()


def recognize_faces():
    """
    Recognize faces in real-time using face_recognition.
    Loads known faces from the database and compares them to faces detected in the webcam feed.
    """
    # Retrieve all users from the database
    users = retrieve_all_users()

    known_names = [user['name'] for user in users]
    known_embeddings = [np.array(user['face_encoding']).flatten() for user in users]

    # Open a connection to the webcam
    video_capture = cv.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        cv.resize(frame, (720, 480))
    
        if not ret:
            print("Failed to capture video.")
            break

        # Detect face locations in the current frame
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            face_image = frame[top:bottom, left:right]

            # Generate face embedding
            embedding = generate_embedding(face_image)

            if embedding is not None:
                distances = face_recognition.face_distance(known_embeddings, embedding)
                best_match_index = np.argmin(distances)
                min_distance = distances[best_match_index]

                recognition_threshold = 0.7
                if min_distance < recognition_threshold:
                    name = known_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(frame, name, (left, top - 20), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

        cv.imshow("Recognize Face", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # Uncomment the function you want to run
    # capture_faces()
    recognize_faces()
