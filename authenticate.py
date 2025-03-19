import cv2
import face_recognition
import pickle

# Load stored face encoding
with open("user_face.pkl", "rb") as f:
    stored_encoding = pickle.load(f)

video_capture = cv2.VideoCapture(0)
cv2.resize(video_capture, (720, 480))

print("Press 'q' to exit after authentication.")

while True:
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.5)
        
        if match[0]:
            print("✅ Authentication Successful! Access Granted.")
            break
        else:
            print("❌ Authentication Failed! Face Not Recognized.")

    cv2.imshow("Face Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
