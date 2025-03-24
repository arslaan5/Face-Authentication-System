import face_recognition
import numpy as np
import cv2 as cv
import sqlite3
from .utils import create_database, insert_user, retrieve_all_users, generate_embedding
import pickle

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
    
    if image.ndim == 2:  # Grayscale
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:  # BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Ensure the image is in 8-bit format
    if image.dtype != np.uint8:
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # Detect face locations in the image
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=3)

    if not face_locations:
        print("No face detected.")
        return None, None

    # Get the first detected face
    top, right, bottom, left = face_locations[0]
    face = image[top:bottom, left:right]

    # Draw a rectangle around the detected face
    cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0), )

    # Resize and normalize the face for model input
    face = cv.resize(face, (160, 160))
    face = face.astype(np.float32) / 255.0  # Normalize to [0, 1]
    # face = np.expand_dims(face, axis=0)  # Add batch dimension

    return face, (left, top, right - left, bottom - top)  # Return face and bounding box


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
    name = input("Enter your name: ").strip().title()

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

        print("Press 's' to save or 'q' to quit.")

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
    known_embeddings = [np.array(user['face_encoding']).flatten() for user in users]

    # Initialize webcam
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
                recognition_threshold = 0.7
                if min_distance < recognition_threshold:
                    name = known_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Draw bounding box and name
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(frame, name, (left, top - 20), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

        # Display the frame
        cv.imshow("Recognize Face", frame)

        # Press 'q' to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # capture_faces()
    recognize_faces()
