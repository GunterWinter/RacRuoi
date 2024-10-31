import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import random
import joblib

# Hàm để tải dữ liệu từ thư mục
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng xám
                if image is not None:
                    images.append(image)
                    labels.append(int(label))  # Chuyển label thành số nguyên
    return images, labels

# Hàm để trích xuất đặc trưng SIFT
def extract_sift_features(images):
    sift = cv2.SIFT_create()
    features = []
    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)  # Không cần chuyển đổi sang xám nữa
        if descriptors is not None:
            features.append(np.mean(descriptors, axis=0))  # Lấy trung bình của các đặc trưng
        else:
            features.append(np.zeros((128,)))  # Nếu không có đặc trưng thì thêm mảng 0
    return np.array(features)

# Đường dẫn đến thư mục dữ liệu
train_data_dir = '../data/train'
valid_data_dir = '../data/valid'

# Tải dữ liệu huấn luyện và xác thực
train_images, train_labels = load_data(train_data_dir)
valid_images, valid_labels = load_data(valid_data_dir)

# Trích xuất đặc trưng SIFT cho dữ liệu huấn luyện
X_train = extract_sift_features(train_images)
y_train = np.array(train_labels)

# Huấn luyện mô hình SVM
clf = svm.SVC(kernel='linear', C=1.0, random_state=42, probability=True)
clf.fit(X_train, y_train)

# Lưu mô hình đã huấn luyện
joblib.dump(clf, 'gesture_recognition_model.pkl')

# Đánh giá mô hình trên tập xác thực
X_valid = extract_sift_features(valid_images)
y_pred = clf.predict(X_valid)

# Tính toán độ chính xác và các chỉ số khác
accuracy = clf.score(X_valid, valid_labels)
f1 = f1_score(valid_labels, y_pred, average='weighted')
precision = precision_score(valid_labels, y_pred, average='weighted')
conf_matrix = confusion_matrix(valid_labels, y_pred)

print(f'Accuracy on validation set: {accuracy * 100:.2f}%')
print(f'F1 Score: {f1:.2f}')
print(f'Precision: {precision:.2f}')
print('Confusion matrix:')
print(conf_matrix)

# Hiển thị 10 bức ảnh dự đoán ngẫu nhiên
indices = random.sample(range(len(y_pred)), 10)
plt.figure(figsize=(15, 6))
for i, index in enumerate(indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(valid_images[index], cmap='gray')  # Hiển thị ảnh xám
    plt.title(f'True: {valid_labels[index]}, Pred: {y_pred[index]}')
    plt.axis('off')
plt.tight_layout()
plt.show()