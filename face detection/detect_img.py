from time import time
import cv2
import numpy as np
import time
from imutils.video import VideoStream
import imutils

prototxt_path = "deploy.prototxt.txt"
# Net weight
caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"

CONFIDENCE = 0.6
detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)  # load model
path = "fd"
def detect(path):
    img = cv2.imread(path)
    # img = imutils.resize(img, width=800)
    (h, w) = img.shape[:2]
    # prepares image for entrance on the model
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # put image on model
    detector.setInput(blob)
    # propagates image through model
    detections = detector.forward()
    # check confidance of 200 predictions
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < CONFIDENCE:
            continue
    
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    
path = 'Mask.png'
detect(path)



