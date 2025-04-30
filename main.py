import cv2
import face_detector
import live_face_detection
from matplotlib import pyplot as plt
import pydoc

__doc__ = """
This module provides a simple interface for face recognition using OpenCV.
"""

def menu():
    """
    Function to display the menu options to the user.
    """

    print("Menu ")
    print("1. Recognize face from photo ")
    print("2. Recognize face from video ")
    print("3. Recognize face from webcam \n")
    global option
    option = int(input("Choose option: "))


def get_reference():
    """
    Function to get the reference image from the user.
    """
    global reference 
    reference = input("Enter filename with reference picture: ")
    reference = cv2.imread(reference)


def main():
    """
    Main function to run the face recognition program.
    """
    if option==1:
        img = input("Enter filename with picture: ")
        img = cv2.imread(img)
        if face_detector.recognize_face(reference, img):
            print("It's the same face")
        else:
            print("It's different face")
    elif option==2:
        video_path = input("Enter video path: ")
        live_face_detection.live_face_detection(reference, video_path)
    elif option==3:
        live_face_detection.live_face_detection(reference)
    else:
        print("Wrong number! ")



get_reference()
menu()
main()

#pydoc.writedoc("main")