import cv2
import tensorflow as tf
from mtcnn import MTCNN

path ="WIN_20220512_17_26_19_Pro.jpg"
detector = MTCNN()

img = cv2.imread(path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_list = detector.detect_faces(img)
for res in result_list:
    x, y, w, h = res['box']
image = cv2.rectangle(img,(x, y), (w+x, h+y), (0, 255, 0), 2)
crop = image[y:y+h, x:x+w]
cv2.imshow("image",crop)
cv2.waitKey(0)