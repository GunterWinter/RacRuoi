import cv2
import numpy as np
import mediapipe as mp
import joblib

# Tải mô hình đã huấn luyện (đã bật probability=True)
model = joblib.load('gesture_recognition_model.pkl')

# Khởi tạo các đối tượng cần thiết từ MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


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

            # Trích xuất ROI và chuyển đổi thành ảnh xám
            hand_roi = frame[y_min:y_max, x_min:x_max]
            gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

            # Resize ROI để phù hợp với kích thước ảnh huấn luyện nếu cần
            gray_roi_resized = cv2.resize(gray_roi, (100, 100))  # Giả sử kích thước huấn luyện là 50x50

            # Trích xuất đặc trưng SIFT từ ROI đã resize
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray_roi_resized, None)

            if descriptors is not None:
                features = np.mean(descriptors, axis=0).reshape(1, -1)

                # Dự đoán nhãn và xác suất tương ứng
                gesture_prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                max_probability = np.max(probabilities)

                print(f"Detected Gesture: {gesture_prediction}, Probability: {max_probability:.2f}")

                # Hiển thị nhãn và xác suất trên khung hình
                cv2.putText(frame, f"Gesture: {gesture_prediction}, Prob: {max_probability:.2f}",
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return frame


# Mở camera và xử lý từng khung hình
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)

    # Hiển thị khung hình đã xử lý
    cv2.imshow('Hand Gesture Recognition', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()