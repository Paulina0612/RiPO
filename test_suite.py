import unittest
import face_detector
import cv2
import time

class TestStringMethods(unittest.TestCase):

    # Test if the face_detector module detect_face function works correctly 
    def test_if_detect_face_true(self):
        img = cv2.imread("data/rihanna_1.jpg")
        self.assertTrue(face_detector.detect_face(img))

    def test_if_detect_face_false(self):
        img = cv2.imread("data/landscape.jpg")
        self.assertFalse(face_detector.detect_face(img))


    
    # Test if the face_detector module recognize_face function works correctly 
    def test_if_recognize_face_true(self):
        reference = cv2.imread("data/rihanna_1.jpg")
        img = cv2.imread("data/rihanna_2.jpg")
        self.assertTrue(face_detector.recognize_face(img, reference))

    def test_if_recognize_face_false(self):
        reference = cv2.imread("data/rihanna_1.jpg")
        img = cv2.imread("data/landscape.jpg")
        self.assertFalse(face_detector.recognize_face(img, reference))


    

    # Test if the face_detector module recognize_face function works 
    # correctly with aberrations
    def test_if_recognize_face_with_glasses_true(self):
        reference = cv2.imread("data/rihanna_1.jpg")
        img = cv2.imread("data/rihanna_3.jpg")
        self.assertTrue(face_detector.recognize_face(img, reference))

    def test_if_recognize_face_with_sunglasses_false(self):
        reference = cv2.imread("data/rihanna_1.jpg")
        img = cv2.imread("data/rihanna_4.jpg")
        self.assertTrue(face_detector.recognize_face(img, reference))




    # Test face detection speed
    def test_face_detection_speed(self):
        start = time.time()
        img = cv2.imread("data/rihanna_1.jpg")
        face_detector.detect_face(img)
        end = time.time()
        elapsed_time = end - start
        self.assertLess(elapsed_time, 0.2, "Face detection took too long")

    # Test face recognition speed
    def test_face_recognition_speed(self):
        start = time.time()
        reference = cv2.imread("data/rihanna_1.jpg")
        img = cv2.imread("data/rihanna_2.jpg")
        face_detector.recognize_face(reference, img)
        end = time.time()
        elapsed_time = end - start
        self.assertLess(elapsed_time, 0.5, "Face recognition took too long")



    # def if_detect_face_false(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()