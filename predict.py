import os
import cv2
import numpy as np
import joblib


# Hàm để trích xuất đặc trưng SIFT
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is not None:
        return np.mean(descriptors, axis=0)  # Lấy trung bình của các đặc trưng
    else:
        return np.zeros((128,))  # Nếu không có đặc trưng thì thêm mảng 0


# Hàm để kiểm tra mô hình trên một thư mục
def test_model(model_path, test_data_dir):
    # Tải mô hình đã lưu
    clf = joblib.load(model_path)

    # Duyệt qua từng hình ảnh trong thư mục test
    for img_name in os.listdir(test_data_dir):
        img_path = os.path.join(test_data_dir, img_name)
        image = cv2.imread(img_path)

        if image is not None:
            # Trích xuất đặc trưng SIFT cho hình ảnh
            features = extract_sift_features(image)
            features = features.reshape(1, -1)  # Định dạng lại để phù hợp với đầu vào của mô hình

            # Dự đoán nhãn cho hình ảnh
            prediction = clf.predict(features)
            print(f'Image: {img_name}, Predicted Label: {prediction[0]}')


# Đường dẫn đến mô hình và thư mục dữ liệu kiểm tra
model_path = 'gesture_recognition_model(old).pkl'
test_data_dir = 'data/valid/0'  # Thay đổi đường dẫn này thành thư mục chứa ảnh cần kiểm tra

# Gọi hàm kiểm tra
test_model(model_path, test_data_dir)