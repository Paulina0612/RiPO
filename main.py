import cv2
import face_detector
import live_face_detection
from matplotlib import pyplot as plt

def Menu():
    print("Menu ")
    print("1. Recognize face from photo ")
    print("2. Recognize face from video ")
    print("3. Recognize face from webcam \n")
    global option
    option = int(input("Choose option: "))


def GetReference():
    global reference 
    reference = input("Enter filename with reference picture: ")
    reference = cv2.imread(reference)

GetReference()
Menu()
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