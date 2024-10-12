import cv2
import mediapipe as mp

# Khởi tạo MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi màu sắc từ BGR sang RGB (MediaPipe sử dụng RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Xử lý hình ảnh để tìm các điểm mốc ngón tay
    results = hands.process(frame_rgb)

    # Nếu có bàn tay được phát hiện
    if results.multi_hand_landmarks:
        finger_count_total = 0
        
        for landmarks in results.multi_hand_landmarks:
            # Vẽ các điểm mốc và đường nối giữa các điểm mốc
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Đếm số ngón tay cho từng bàn tay
            finger_count = 0
            
            # Các chỉ số điểm mốc cho các ngón tay
            tips = [4, 8, 12, 16, 20]  # Chỉ số của các đầu ngón tay
            
            # Kiểm tra nếu ngón tay nào đang giơ lên
            for tip in tips:
                if landmarks[tip].y < landmarks[tip - 2].y:  # So sánh vị trí Y của điểm đầu ngón và khớp
                    finger_count += 1
            
            # Cộng dồn số ngón tay của tất cả các bàn tay
            finger_count_total += finger_count
        
        # Hiển thị số ngón tay đã đếm được
        cv2.putText(frame, f'Total Finger Count: {finger_count_total}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Hiển thị kết quả
    cv2.imshow("Finger Count", frame)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()