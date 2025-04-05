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
    

def recognize_face(base_image, input_image):
    if detect_face(base_image) and detect_face(input_image):
        # Get the face encodings for the known face and the unknown face
        my_face_encoding = face_recognition.face_encodings(base_image)[0]
        unknown_encoding = face_recognition.face_encodings(input_image)[0]

        # Now we can see the two face encodings are of the same person with `compare_faces`!
        results = face_recognition.compare_faces([my_face_encoding], unknown_encoding)

        if results[0]:
            global face_match
            face_match = True
        else:
            face_match = False
    else: 
        face_match = False
