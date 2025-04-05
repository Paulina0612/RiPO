import face_detector
import cv2
import threading


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


counter = 0

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 20 == 0:
            try:
                threading.Thread(target=face_detector.recognize_face, args=(cv2.imread("photos\\photo.jpg"), frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1


        if face_detector.face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()

