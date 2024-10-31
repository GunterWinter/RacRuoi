import cv2
import numpy as np
import mediapipe as mp
import joblib

# Tải mô hình đã huấn luyện
model = joblib.load('hog_model.pkl')

# Khởi tạo các đối tượng cần thiết từ MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Hàm để trích xuất đặc trưng HOG
def extract_hog_features(image):
    hog = cv2.HOGDescriptor()
    image_resized = cv2.resize(image, (64, 128))  # Kích thước có thể điều chỉnh tùy theo yêu cầu
    hog_features = hog.compute(image_resized)
    return hog_features.flatten().reshape(1, -1)  # Trả về mảng 1 chiều

# Hàm để xử lý và dự đoán cử chỉ từ khung hình
def process_frame(frame):
    # Chuyển đổi khung hình sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện bàn tay trong khung hình
    result = hands.process(frame_rgb)

    # Nếu phát hiện thấy bàn tay
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Vẽ các điểm mốc trên bàn tay
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Lấy kích thước khung hình
            h, w, _ = frame.shape

            # Tìm bounding box quanh bàn tay dựa trên các điểm mốc
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Trích xuất ROI và kiểm tra xem nó có hợp lệ không
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size == 0:  # Kiểm tra xem ROI có rỗng không
                continue  # Bỏ qua nếu ROI rỗng

            gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

            # Resize ROI để phù hợp với kích thước ảnh huấn luyện nếu cần
            gray_roi_resized = cv2.resize(gray_roi, (64, 128))  # Kích thước có thể điều chỉnh tùy theo yêu cầu

            # Trích xuất đặc trưng HOG từ ROI đã resize
            features = extract_hog_features(gray_roi_resized)

            gesture_prediction = model.predict(features)[0]

            print(f"Detected Gesture: {gesture_prediction}")
            cv2.putText(frame, f"Gesture: {gesture_prediction}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame


# Mở camera và xử lý từng khung hình
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture video.")
        break

    processed_frame = process_frame(frame)

    # Hiển thị khung hình đã xử lý
    cv2.imshow('Hand Gesture Recognition', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()