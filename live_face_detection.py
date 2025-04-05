import face_detector
import cv2
import threading


face_detected = False

def live_face_detection(base_image, video_path : str = None):
    #global cap

    if video_path is not None:
        # Use video file
        cap = cv2.VideoCapture(video_path)
    else:
        # Use webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    counter = 0

    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        if ret:
            if counter % 30 == 0:
                try:
                    threading.Thread(target=face_detector.recognize_face, args=(base_image, 
                                                                                frame.copy(),)).start()
                except ValueError:
                    pass
            counter += 1


            if face_detector.face_match:
                face_detected = True
            else:
                face_detected = False
                
            print("Face detected:", face_detected)

            cv2.imshow("Live Face Detection", frame)

        key = cv2.waitKey(1)
        if key == 27: # ESC key
            break

    cap.release()


