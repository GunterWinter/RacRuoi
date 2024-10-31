import cv2
import os
import mediapipe as mp

# Tạo thư mục để lưu ảnh nếu chưa tồn tại
output_dir = 'data/train/5'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Khởi tạo các đối tượng cần thiết từ MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Hàm để tìm số lớn nhất trong tên tệp hiện có
def get_max_image_index(directory):
    max_index = 0
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            try:
                index = int(filename.split('_')[1].split('.')[0])
                if index > max_index:
                    max_index = index
            except ValueError:
                continue
    return max_index

# Mở camera
cap = cv2.VideoCapture(0)
count = get_max_image_index(output_dir) + 1
finish = count +2500
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Chuyển đổi khung hình sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Phát hiện bàn tay trong khung hình
    result = hands.process(frame_rgb)

    # Nếu phát hiện thấy bàn tay
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
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

            # Lưu ảnh xám vào thư mục
            cv2.imwrite(os.path.join(output_dir, f'image_{count}.jpg'), gray_roi)
            count += 1
    if (count >= finish):
        break

    # Hiển thị khung hình đã xử lý
    cv2.imshow('Capture Hand Gesture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()