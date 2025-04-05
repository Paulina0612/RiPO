import cv2
import face_detector
import live_face_detection


base_image = cv2.imread('photos\\photo.jpg')

live_face_detection.live_face_detection(base_image, video_path='')


# TODO: Trzeba jeszce dodac rysowanie landmarkow na twarzy

