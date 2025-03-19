import sqlite3
import pickle
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
from utils import create_database, insert_user, generate_embedding, retrieve_all_users
from scipy.spatial.distance import cosine


detector = MTCNN()

# Detect face in an image using MTCNN
def detect_face(image_path):
    """Detect face in an image using MTCNN"""
    
    if isinstance(image_path, str):  
        image = cv.imread(image_path)  # Load image from file
    elif isinstance(image_path, np.ndarray):  
        image = image_path  # Use the given image array
    else:
        raise ValueError("Input must be a file path or a NumPy image array")
    
    faces = detector.detect_faces(image)

    embeddings = generate_embedding(image_path)

    if faces:
        facial_area = embeddings['facial_area']  # This will be a dict: {'x':..., 'y':..., 'w':..., 'h':...}
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        x, y = max(0, x), max(0, y)  # Ensure positive coordinates
        
        face = image[y:y+h, x:x+w]  # Extract the detected face

        # Convert BGR to RGB (for displaying with matplotlib)
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Draw a rectangle around the detected face
        cv.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 4)

        # Resize and normalize the face for model input
        face = cv.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = face.astype('float32') / 255  # Normalize to [0, 1]

        return face, facial_area  # Return the processed face instead of full image

    print("No face detected")
    return None, None


def capture_faces():
    """Capture and save faces from webcam in real-time using MTCNN for detection and DeepFace for recognition."""
    
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
        face, facial_area = detect_face(frame)

        if face is not None and facial_area is not None:
            # Extract bounding box coordinates
            x, y, width, height = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            # Draw bounding box around detected face
            cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Display the frame
        cv.imshow("Register Face", frame)

        print("Face detected! Press 's' to save or 'q' to quit.")

        # Press 's' to capture and save the face
        key = cv.waitKey(1) & 0xFF
        if key == ord('s'):
            if face is not None and facial_area is not None:
                # Extract bounding box coordinates
                x, y, width, height = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

                # Draw bounding box around detected face
                cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                face_image = frame[y:y + height, x:x + width]

                # Generate face embedding using DeepFace
                embedding = generate_embedding(face_image)

                # Convert the embedding to a binary format for storage
                embedding_blob = pickle.dumps(embedding)

                # Save the user data and face embedding to the database
                insert_user(conn, name, embedding_blob)
                break
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

# Run face recognition
# capture_faces()

def recognize_faces():
    """Recognize faces in real-time using MTCNN for detection and DeepFace for recognition."""

    # Retrieve all users from the database
    users = retrieve_all_users()

    # Load known embeddings and names
    known_names = [user['name'] for user in users]
    known_embeddings = [np.array(user['face_encoding']['embedding']).flatten() for user in users]

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
        face, facial_area = detect_face(frame)

        if face is not None and facial_area is not None:
            # Extract bounding box coordinates
            x, y, width, height = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']

            # Draw bounding box around detected face
            cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            face_image = frame[y:y + height, x:x + width]

            face_image = cv.resize(face_image, (160, 160))
            face_image = face_image.astype('float32') / 255  

            try:
                # Generate embedding for the detected face
                embedding = generate_embedding(face_image)
                embedding = np.array(embedding['embedding']).flatten()

                # Compare the embedding with known embeddings
                distances = []
                for known_embedding in known_embeddings:
                    distance = cosine(embedding, known_embedding)
                    distances.append(distance)

                if distances:
                    min_distance = min(distances)
                    best_match_index = np.argmin(distances)

                    # Set a threshold for recognition
                    if min_distance < 0.6:  # Adjust this threshold as needed
                        name = known_names[best_match_index]
                    else:
                        name = "No match found"
                else:
                    name = "No known embeddings to compare"
            except Exception as e:
                print("Error during face recognition:", e)
                name = "Unknown"
            
            # Draw bounding box and name
            cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        else:
            # No face detected
            name = "No face detected"

        cv.resize(frame, (720, 480))
        cv.putText(frame, name, (10, 40), cv.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv.imshow("Recognize Face", frame)

        # Press 'q' to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    video_capture.release()
    cv.destroyAllWindows()

    
# Run the function
recognize_faces()
print("Face recognition completed.")

# users = retrieve_all_users()
# print("Retrieved users:")
# for user in users:
#     print("\n------------------------------------------------\n")
#     print(user)
