import socket
import pygame
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import numpy as np
import cv2
import time
from collections import Counter
import imutils
from sklearn import neighbors
import math
import os
import os.path
from face_recognition.face_recognition_cli import image_files_in_folder

class Vidcamera1(object):
    def __init__(self):
        self.timer = 0
        self.previousImage = ""
        self.image = ""
        self.clock = pygame.time.Clock()
        self.video = cv2.VideoCapture(0)
        # nếu thêm khuôn mặt mới thì sẽ sử dụng classifier để train lại.
        # classifier = self.train("train_dir", model_save_path="trained_knn_model.clf", n_neighbors=5)

    @staticmethod
    def train(train_dir, model_save_path=None, n_neighbors=5, knn_algo='ball_tree', verbose=False):
        """
        Trains a k-nearest neighbors classifier for face recognition.

        :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

        :param model_save_path: (tùy chọn) đường dẫn để lưu mô hình trên đĩa
        :param n_neighbors: (tùy chọn) số lượng láng giềng để sử dụng trong phân loại. Được chọn tự động nếu không được chỉ định.
        :param knn_algo: (tùy chọn) cấu trúc dữ liệu cơ sở để hỗ trợ knn. Giá trị mặc định là ball_tree.
        :param verbose: độ chi tiết trong quá trình huấn luyện
        :return: trả về bộ phân loại knn đã được huấn luyện trên dữ liệu cho trước.
        """
        X = []
        y = []

        # Lặp qua từng người trong tập huấn luyện.
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Lặp qua từng hình ảnh huấn luyện cho người hiện tại.
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # Nếu không có người (hoặc quá nhiều người) trong hình ảnh huấn luyện, bỏ qua hình ảnh đó.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Thêm mã hóa khuôn mặt cho hình ảnh hiện tại vào tập huấn luyện.
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

        # Xác định số lượng KNN
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        return knn_clf
    
        
    ## processing the frame.
    def predict(self,X_frame, knn_clf=None, model_path='trained_knn_model.clf', distance_threshold=0.5, train_dir='train_dir',model='hog', upsample_times=2):
        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either through knn_clf or model_path")
        # Load a trained KNN model (if not None)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        # Tải các nhãn của bộ dữ liệu kiểm tra.
        test_labels = {}
        if train_dir:
            for sub_dir in os.listdir(train_dir):
                for image_file in os.listdir(os.path.join(train_dir, sub_dir)):
                    image_path = os.path.join(train_dir, sub_dir, image_file)
                    label = sub_dir
                    test_labels[image_path] = label

        X_face_locations = face_recognition.face_locations(X_frame,model=model,number_of_times_to_upsample=upsample_times)

        if len(X_face_locations) == 0:
            return []

        faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=5)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        names = []
        accuracies=[]
        for i in range(len(X_face_locations)):
            if are_matches[i]:
                # Lấy tên nhãn cho dự đoán (nếu nó tồn tại)
                image_path = list(test_labels.keys())[list(test_labels.values()).index(knn_clf.predict([faces_encodings[i]])[0])]
                name = test_labels[image_path]
                confidence= 1 - closest_distances[0][i][0]
            else:
                name = "unknown"
                confidence = 0
            names.append(name)
            accuracies.append(confidence)
        # Kết hợp vị trí khuôn mặt và tên
        return list(zip(names, X_face_locations, accuracies))
    
    def show_prediction_labels_on_image(self,frame, predictions): 
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)

        for name, (top, right, bottom, left), accuracy in predictions:
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
            draw.text((left + 6, bottom - text_height - 20), f"Confidence:{accuracy:.2f}", fill=(255, 255, 255, 255))

        # Loại bỏ thư viện vẽ khỏi bộ nhớ theo hướng dẫn của Pillow
        del draw
        # Lưu hình ảnh dưới định dạng OpenCV để có thể hiển thị nó.
        opencvimage = np.array(pil_image)
        return opencvimage
    
    
    def process_frame(self,frame, knn_clf=None, model_path='trained_knn_model.clf', distance_threshold=0.5, train_dir='train_dir'):
        """
        Xử lý từng khung hình trong luồng video
        :param frame: khung hình để thực hiện dự đoán
        :param knn_clf: (tùy chọn) đối tượng bộ phân loại knn. Nếu không được chỉ định, model_save_path phải được chỉ định.
        :param model_path: (tùy chọn) trỏ đến tệp chứa bộ phân loại knn
        :param distance_threshold: (tùy chọn) ngưỡng khoảng cách cho phân loại khuôn mặt. Càng lớn thì càng có nhiều khả năng xếp nhầm một người lạ thành một người quen biết.
        
        :return: khung hình với các hộp giới hạn và tên cho các khuôn mặt được nhận dạng
        """
        # Lấy các dự đoán.
        predictions = self.predict(frame, knn_clf, model_path, distance_threshold, train_dir,model='hog', upsample_times=2)

        # Hiển thị các dự đoán trên hình ảnh.
        output_frame = self.show_prediction_labels_on_image(frame, predictions)

        return output_frame

    # Vòng lặp chương trình chính:
    def framing(self):
        if self.timer < 1:
            success, data = self.video.read()
            frame = self.process_frame(data)
            self.timer = 0
        else:
            # Đếm ngược thời gian:
            self.timer -= 1
        self.previousImage = self.image
        try:
            self.image = frame
        except:
            self.image = self.previousImage
        output = self.image
        self.clock.tick(1000)
        ret, jpeg = cv2.imencode('.jpg', output)
        return jpeg.tobytes()
