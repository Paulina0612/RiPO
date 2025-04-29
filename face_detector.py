import face_recognition
import cv2
import threading

face_match = False

def detect_face(image):
    # Detect faces in the image
    face_detection = face_recognition.face_encodings(image)

    if len(face_detection) == 0:
        return False
    else:
        return True
    
def detect_face_locations(image):
    # Detect face locations in the image
    face_locations = face_recognition.face_locations(image)
    return face_locations

def recognize_face(base_image, input_image):
    # Convert images to RGB
    base_image_rgb = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Get face encodings once
    base_face_encodings = face_recognition.face_encodings(base_image_rgb)
    input_face_encodings = face_recognition.face_encodings(input_image_rgb)

    # Check if faces are detected in both images
    if len(base_face_encodings) == 0 or len(input_face_encodings) == 0:
        return False

    # Compare faces
    results = face_recognition.compare_faces([base_face_encodings[0]], input_face_encodings[0])
    return results[0]