import face_recognition
import cv2
import threading
import pydoc

__doc__ = """
This module provides functions to detect and recognize faces in images using the face_recognition library. It can also detect face locations in images.
It includes functions to check if a face is detected, get face locations, and compare two images to see if they contain the same face.
"""

#face_match = False 

def detect_face(image):
    """
    A function that detects faces in the image.
    
    Args:
        image: The image in which to detect faces.

    Returns:
        True if a face is detected,
        False if face is not detected.
    """
    face_detection = face_recognition.face_encodings(image)

    if len(face_detection) == 0:
        return False
    else:
        return True
    


def detect_face_locations(image):
    """
    A function that returns faces' locations if at least one face is detected in the image.
    
    Args:
        image: The image in which to detect faces' locations.

    Returns:
        None if no face is detected,
        a list of tuples containing the coordinates of the face locations if at least one face is detected.
    """

    if detect_face(image) == False:
        return None
    
    face_locations = face_recognition.face_locations(image)
    return face_locations




def recognize_face(base_image, input_image):
    """
    A function that compares two images to check if they contain the same face.

    Args:
        base_image: The image to compare against.
        input_image: The image to be compared.

    Returns:
        True if the faces match,
        False if the faces do not match.
    """

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


#pydoc.writedoc("face_detector")