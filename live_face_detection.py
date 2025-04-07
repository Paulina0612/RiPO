import face_detector
import cv2
import threading


face_detected = False
event = threading.Event()

def face_detection(base_image, video_path: str = None):
    if video_path is not None:
        # Use video file
        cap = cv2.VideoCapture(video_path)
    else:
        # Use webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counter = 0

    while event.is_set() == False:
        ret, frame = cap.read()

        if frame is None:
            break

        if ret:
            if counter % 20 == 0:  # Process every 20th frame
                try:
                    # Perform face recognition
                    face_detected = face_detector.recognize_face(base_image, frame.copy())
                    print("Face detected:", face_detected)
                except ValueError:
                    face_detected = False
                    print("Error during face recognition.")

            counter += 1

    cap.release()
    cv2.destroyAllWindows()


def live_face_detection(base_image, video_path: str = None):
    thread = threading.Thread(target=face_detection, args=(base_image.copy(), video_path))
    thread2 = threading.Thread(target=listener)
    thread.start()
    thread2.start()
    

def listener():
    input()
    event.set()
    