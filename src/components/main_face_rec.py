import face_recognition
import numpy as np
import cv2 as cv
import sqlite3
from utils import create_database, insert_user, retrieve_all_users
import pickle

def detect_face(image):
    """
    Detect face in an image using the face_recognition library.
    """
    if isinstance(image, str):  # If input is a file path
        image = face_recognition.load_image_file(image)
    elif isinstance(image, np.ndarray):  # If input is a NumPy array
        image = image
    else:
        raise ValueError("Input must be a file path or a NumPy image array")

    # Detect face locations in the image
    face_locations = face_recognition.face_locations(image)

    if face_locations:
        # Get the first detected face
        top, right, bottom, left = face_locations[0]

        # Extract the face region
        face = image[top:bottom, left:right]

        # Convert BGR to RGB (for displaying with matplotlib)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Draw a rectangle around the detected face
        cv.rectangle(image_rgb, (left, top), (right, bottom), (0, 255, 0), 4)

        # Resize and normalize the face for model input
        face = cv.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = face.astype('float32') / 255  # Normalize to [0, 1]

        return face, (left, top, right - left, bottom - top)  # Return face and bounding box

    print("No face detected")
    return None, None


def generate_embedding(image):
    """
    Generate face embedding using the face_recognition library.
    """
    # Convert the image to RGB (required by face_recognition)
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Generate face encodings
    face_encodings = face_recognition.face_encodings(rgb_image)

    if face_encodings:
        return face_encodings[0]  # Return the first face encoding
    else:
        return None
    

def capture_faces():
    """Capture and save faces from webcam in real-time using face_recognition."""
    
    # Connect to SQLite database
    conn = sqlite3.connect(r'E:\Face-Recognition-for-Login-Authentication-System\face_recognition.db')
    create_database(conn)

    # Initialize webcam
    video_capture = cv.VideoCapture(0)

    print("Press 's' to capture your face and save.")
    print("Press 'q' to quit.")
    
    # Prompt the user to enter their name
    name = input("Enter your name: ")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Detect faces in the frame
        face, bbox = detect_face(frame)

        if face is not None and bbox is not None:
            # Extract bounding box coordinates
            x, y, w, h = bbox

            # Draw bounding box around detected face
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv.imshow("Register Face", frame)

        print("Face detected! Press 's' to save or 'q' to quit.")

        # Press 's' to capture and save the face
        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            if face is not None and bbox is not None:
                # Extract bounding box coordinates
                x, y, w, h = bbox

                # Draw bounding box around detected face
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_image = frame[y:y + h, x:x + w]

                # Generate face embedding using face_recognition
                embedding = generate_embedding(face_image)

                if embedding is not None:
                    # Convert the embedding to a binary format for storage
                    embedding_blob = pickle.dumps(embedding)

                    # Save the user data and face embedding to the database
                    insert_user(conn, name, embedding_blob)
                    break
                else:
                    print("Failed to generate embedding. Please try again.")
            else:
                print("No face detected. Please try again.")

        # Press 'q' to quit
        elif key == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv.destroyAllWindows()

    # Close the database connection
    conn.close()


def recognize_faces():
    """Recognize faces in real-time using face_recognition."""
    
    # Retrieve all users from the database
    users = retrieve_all_users()

    # Load known embeddings and names
    known_names = [user['name'] for user in users]
    known_embeddings = [pickle.loads(user['face_encoding']) for user in users]

    if known_embeddings:
        print("Known embeddings shape:", np.array(known_embeddings).shape)
    else:
        print("No known embeddings available!")

    # Initialize webcam
    video_capture = cv.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video.")
            break

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in face_locations:
            # Extract the face region
            face_image = frame[top:bottom, left:right]

            # Generate face embedding using face_recognition
            embedding = generate_embedding(face_image)

            if embedding is not None:
                # Compare the embedding with known embeddings
                distances = face_recognition.face_distance(known_embeddings, embedding)
                best_match_index = np.argmin(distances)
                min_distance = distances[best_match_index]

                # Set a threshold for recognition
                recognition_threshold = 0.6  # Adjust this threshold as needed
                if min_distance < recognition_threshold:
                    name = known_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Draw bounding box and name
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

        # Display the frame
        cv.imshow("Recognize Face", frame)

        # Press 'q' to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv.destroyAllWindows()