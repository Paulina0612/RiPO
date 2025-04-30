import face_recognition 
import numpy as np
import face_detector
import cv2
import threading
from enum import Enum
from PIL import Image, ImageDraw
import pydoc

__doc__ ="""
This module uses the face_recognition library to detect faces in a video stream (either from a webcam or a video file).
It can also apply filter to the detected faces and displays the result in a window. The filter can be changed using keyboard shortcuts.
"""

face_detected = False
event = threading.Event()
ifOpen = True

normal_makeup = ((39, 54, 68), (0, 0, 150), (255, 255, 255), (0, 0, 0))
funny_makeup = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0))

class Filer(Enum):
    """
    Enum for filter options.
    """
    NORMAL_MAKEUP = 1
    FUNNY_MAKEUP = 2
    NONE = 3

filter = Filer.NONE


def face_detection(base_image, video_path: str = None):
    """
    This function detects faces in images and videos. And applies filters to the detected faces. It also displays the result in a window.

    Args:
        base_image: The image to be used as a base for face recognition.
        video_path: The path to the video file. If None, the webcam will be used.
    """
    # Menu for selecting filter
    global filter
    print("Select filter:")
    print("1. Normal Makeup")
    print("2. Funny Makeup")
    print("3. Funny Makeup")
    print("Press ESC to exit and Enter to stop the detection.")

    if video_path is not None:
        # Use video file
        cap = cv2.VideoCapture(video_path)
    else:
        # Use webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counter = 0

    while event.is_set() == False or ifOpen:
        ret, frame = cap.read()

        if frame is None:
           return
        
        if ret:
            if counter % 20 == 0:  # Process every 20th frame
                try:
                    # Perform face recognition
                    global face_detected
                    face_detected = face_detector.recognize_face(base_image, frame.copy())
                except ValueError:
                    face_detected = False

            
            # Show the video frame
            global filter
            frame = use_filter(frame)

            # Display the result on the video frame
            text = "Face Recognized: " + str(face_detected)
            color = (0, 255, 0) if face_detected else (0, 0, 255)  # Green for True, Red for False
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            cv2.imshow("Video", frame)
            
            counter += 1

        # Wait for the ESC key to exit
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            return
        elif cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) <1:
            return
        # 1 key to toggle filter
        elif key == 49 and face_detected:  
            filter = Filer.NORMAL_MAKEUP       
        # 2 key to toggle filter
        elif key == 50 and face_detected:
            filter = Filer.FUNNY_MAKEUP
        # 3 key to toggle filter
        elif key == 51 and face_detected:
            filter = Filer.NONE

    cap.release()
    cv2.destroyAllWindows()


def live_face_detection(base_image, video_path: str = None):
    """
    This function starts the face detection in a separate thread and waits for the user to press Enter to stop the detection.

    Args:
        base_image: The image to be used as a base for face recognition.
        video_path: The path to the video file. If None, the webcam will be used.
    """
    thread = threading.Thread(target=face_detection, args=(base_image.copy(), video_path))
    thread2 = threading.Thread(target=listener)
    thread.start()
    thread2.start()


def listener():
    """
    This function waits for the user to press Enter to stop the face detection and close the window.
    """
    input()
    event.set()


def use_filter(frame):
    """
    This function applies the selected filter to the frame.

    Args:
        frame: The frame to be used as a base for face recognition.

    Returns:
        The frame with the selected filter applied.
    """
    if face_detected == False:
        return frame
    
    makeup_colors=normal_makeup if filter==Filer.NORMAL_MAKEUP else funny_makeup

    if filter == Filer.NORMAL_MAKEUP or filter == Filer.FUNNY_MAKEUP:
        return apply_makeup(frame, makeup_colors)
    elif filter == Filer.NONE:
        return frame
    

def apply_makeup(frame, makeup_colors):
    """
    This function applies the selected makeup colors to the frame.
    
    Args:
        frame: The frame to be used as a base for face recognition.
        makeup_colors: The colors to be used for the makeup.

    Returns:
        The frame with the selected makeup applied.
    """

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(frame)

    pil_image = Image.fromarray(frame)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(
            makeup_colors[0][0], makeup_colors[0][1], makeup_colors[0][2], 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(
            makeup_colors[0][0], makeup_colors[0][1], makeup_colors[0][2], 128))
        d.line(face_landmarks['left_eyebrow'], fill=(
            makeup_colors[0][0], makeup_colors[0][1], makeup_colors[0][2], 150), 
            width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(
            makeup_colors[0][0], makeup_colors[0][1], makeup_colors[0][2], 150), 
            width=5)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(
            makeup_colors[1][0], makeup_colors[1][1], makeup_colors[1][2], 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(
            makeup_colors[1][0], makeup_colors[1][1], makeup_colors[1][2], 128))
        d.line(face_landmarks['top_lip'], fill=(
            makeup_colors[1][0], makeup_colors[1][1], makeup_colors[1][2], 64), 
            width=8)
        d.line(face_landmarks['bottom_lip'], fill=(
            makeup_colors[1][0], makeup_colors[1][1], makeup_colors[1][2], 64), 
            width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(
            makeup_colors[2][0], makeup_colors[2][1], makeup_colors[2][2], 30))
        d.polygon(face_landmarks['right_eye'], fill=(
            makeup_colors[2][0], makeup_colors[2][1], makeup_colors[2][2], 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], 
            fill=(makeup_colors[3][0], makeup_colors[3][1], makeup_colors[3][2], 
                  110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], 
            fill=(makeup_colors[3][0], makeup_colors[3][1], makeup_colors[3][2], 
                  110), width=6)

    return np.array(pil_image)

#pydoc.writedoc("live_face_detection")