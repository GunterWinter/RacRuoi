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

# Hàm để trích xuất đặc trưng HOG
def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    features = []
    for img in images:
        # Resize ảnh về kích thước cố định trước khi trích xuất đặc trưng
        img_resized = cv2.resize(img, (64, 128))  # Kích thước có thể điều chỉnh tùy theo yêu cầu
        hog_features = hog.compute(img_resized)  # Tính toán đặc trưng HOG
        features.append(hog_features.flatten())  # Chuyển đổi thành mảng 1 chiều
    return np.array(features)

# Đường dẫn đến thư mục dữ liệu
train_data_dir = '../data/train'
valid_data_dir = '../data/valid'

# Tải dữ liệu huấn luyện và xác thực
train_images, train_labels = load_data(train_data_dir)
valid_images, valid_labels = load_data(valid_data_dir)

# Trích xuất đặc trưng HOG cho dữ liệu huấn luyện
X_train = extract_hog_features(train_images)
y_train = np.array(train_labels)

# Huấn luyện mô hình SVM với kernel RBF
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
clf.fit(X_train, y_train)

# Lưu mô hình đã huấn luyện
joblib.dump(clf, 'hog_model.pkl')

# Đánh giá mô hình trên tập xác thực
X_valid = extract_hog_features(valid_images)
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
    plt.imshow(valid_images[index], cmap='gray')  # Hiển thị ảnh xám gốc
    plt.title(f'True: {valid_labels[index]}, Pred: {y_pred[index]}')
    plt.axis('off')
plt.tight_layout()
plt.show()