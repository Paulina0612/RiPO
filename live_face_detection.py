import face_detector
import cv2
import threading

face_detected = False
event = threading.Event()
ifOpen = True

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

            # Display the result on the video frame
            text = "Face Detected: " + str(face_detected)
            color = (0, 255, 0) if face_detected else (0, 0, 255)  # Green for True, Red for False
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Show the video frame
            cv2.imshow("Video", frame)

            counter += 1

        # Wait for the ESC key to exit
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            return

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