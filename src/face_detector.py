import cv2

class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "src/haarcascade_frontalface_default.xml")

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bbox = self.detector.detectMultiScale(gray, 1.1, 15)

        if len(bbox) == 0:
            return None

        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[0] + bbox[2]
        y_max = bbox[1] + bbox[3]

        return (x_min, y_min, x_max, y_max)