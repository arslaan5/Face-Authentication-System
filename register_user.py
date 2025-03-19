import cv2
import face_recognition
import pickle

# Capture image from webcam
video_capture = cv2.VideoCapture(0)

print("Press 's' to capture your face and save.")

while True:
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (720, 480))
    cv2.imshow("Register Face", frame)

    # Press 's' to capture and store face
    if cv2.waitKey(1) & 0xFF == ord('s'):
        face_locations = face_recognition.face_locations(frame)
        
        if face_locations:
            face_encoding = face_recognition.face_encodings(frame)[0]
            print(face_encoding)
            # Save encoding
            with open("user_face.pkl", "wb") as f:
                pickle.dump(face_encoding, f)
            
            print("Face registered successfully!")
            break
        else:
            print("No face detected. Try again.")

video_capture.release()
cv2.destroyAllWindows()
