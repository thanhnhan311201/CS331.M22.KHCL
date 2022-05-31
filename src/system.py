import os
import pickle
from PIL import Image
import numpy as np
import cv2
from scipy import spatial
import time
import glob
# from tqdm import tqdm
import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from src import facenet
from src import feature_extractor
from src import face_detector

class face_recognition_system:
    def __init__(self, model_path):
        self.extractor = feature_extractor.facenet_FE(model_path)

        self.detector = face_detector.FaceDetector()

        # self.dataset_path = dataset_path
        # if not os.path.exists(self.dataset_path):
        #     os.mkdir(self.dataset_path)

        # self.image_folder = dataset_path + 'images/'

        # self.feature_folder_path = dataset_path + 'feature_vectors/'
        # if not os.path.exists(self.feature_folder_path):
        #     os.mkdir(self.feature_folder_path)

    # def index_dataset(self):
    #     with tf.Graph().as_default():
    #         with tf.Session() as sess:

    #             facenet.load_model(self.model_path)
    #             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    #             for img_path in tqdm(os.listdir(self.image_folder)):


    #                 name = img_path.split('.')[0]
    #                 vector_file = f'{self.feature_folder_path}/{name}.pkl'
                    
    #                 img_path_full = os.path.join(self.image_folder + img_path)
    #                 img = cv2.imread(img_path_full)

    #                 bbox = self.detector.detect(img)
    #                 if bbox is None:
    #                     return 0

    #                 x_min, y_min, x_max, y_max = bbox
    #                 cropped = img[y_min:y_max, x_min:x_max]
    #                 # PIL_image = Image.open(img_path_full).crop((x_min, y_min, x_max, y_max))
                        
    #                 try:
    #                     feed_dict = { images_placeholder: cropped, phase_train_placeholder:False }
    #                     emb = self.sess.run(embeddings, feed_dict=feed_dict)
    #                 except:
    #                     continue

    #                 pickle.dump(emb, open(vector_file, "wb"))

    #     return 'Done!'

    def verify_face(self, face_cmnd, face_img):
        # res_dict = {}

        # max_similarity = 0.0
        # for vector_file in os.listdir(self.feature_folder_path):
            # feature_vectors = pickle.load(open(self.feature_folder_path + vector_file,"rb"))

        feature_face_cmnd = self.extractor.extract(face_cmnd)
        feature_face_img = self.extractor.extract(face_img)

        # calculate the cosine similarity
        cos_sim = 1 - spatial.distance.cosine(feature_face_cmnd, feature_face_img)
        # name = vector_file.split('.')[0]
        # temp = {'similary': round(cos_sim, 2)}
        # res_dict[name] = temp

        # if cos_sim > max_similarity:
        #     max_similarity = cos_sim
        #     face_name = name

        if cos_sim < 0.5:
            return False, cos_sim

        return True, cos_sim

    def verify_face_via_image(self, cmnd_file, image_file):
        cmnd_img = cv2.imread(cmnd_file)
        cmnd_img = cv2.cvtColor(cmnd_img, cv2.COLOR_BGR2RGB)

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_cmnd_bbox = self.detector.detect(cmnd_img)
        face_image_bbox = self.detector.detect(image)
        if face_cmnd_bbox is None or face_image_bbox is None:
            return 0

        # res_dict = {}

        x_min, y_min, x_max, y_max = face_cmnd_bbox
        face_cmnd = cmnd_img[y_min:y_max, x_min:x_max]
        res_cmnd = cv2.rectangle(cmnd_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        x_min, y_min, x_max, y_max = face_image_bbox
        face_img = image[y_min:y_max, x_min:x_max]
        res_img = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        result, cos_sim = self.verify_face(face_cmnd, face_img)

        print(result, cos_sim)
        cv2.imshow("res_cmnd", res_cmnd)
        cv2.imshow("res_img", res_img)
        cv2.waitKey(0)

        return result

    # def recognize_face_via_camera(self):
    #     cap = cv2.VideoCapture(0)

    #     if not cap.isOpened():
    #         raise IOError("Cannot open webcam")

    #     prev_frame_time = 0
    #     new_frame_time = 0

    #     res_dict = {}

    #     while True:
    #         ret, frame = cap.read()

    #         if ret is None:
    #             continue

    #         # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    #         bbox = self.detector.detect(frame)
    #         if bbox is None:
    #             continue

    #         x_min, y_min, x_max, y_max = bbox
    #         cropped = ret[y_min:y_max, x_min:x_max]
    #         # processed_frame = Image.fromarray(np.uint8(frame[y_min:y_max, x_min:x_max])).convert('RGB')
    #         res_frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    #         result = self.verify_face(cropped)
    #         if result == "Unknown":
    #             face_name = "Unknown"
    #             cv2.putText(res_frame, face_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #         else:
    #             face_name, similarity, res_dict = result
    #             cv2.putText(res_frame, f'{face_name}: {round(similarity, 2)}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #         new_frame_time = time.time()
    #         fps = 1 / (new_frame_time - prev_frame_time)
    #         prev_frame_time = new_frame_time
    #         fps = str(int(fps * 100))
    #         cv2.putText(res_frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 1, cv2.LINE_AA)
    #         cv2.putText(res_frame, 'FPS', (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 0), 1, cv2.LINE_AA)

    #         y_axis = 1
    #         for face in res_dict:
    #             string = f'{face}: {res_dict[face]["similary"]}'
    #             cv2.putText(res_frame, string, (500, y_axis * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 1, cv2.LINE_AA)
    #             y_axis += 1

    #         cv2.imshow("Result", res_frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     cap.release()
    #     cv2.destroyAllWindows()