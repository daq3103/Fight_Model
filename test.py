from infer import load_model_once, predict_fight_violence

import cv2
import torch
import numpy as np


# CẤU HÌNH CỐ ĐỊNH TỪ MODEL VÀ CONFIG CỦA BẠN
T = 16 
SLIDE_WINDOW = 1
IMAGE_SIZE = 224
NUM_CLASSES = 1
CKPT_PATH = "mvit_v2_s_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = load_model_once(CKPT_PATH, NUM_CLASSES)
# Hàng đợi để lưu trữ 16 frames cần thiết
frame_queue = [] 

def preprocess_frame(frame):
    """Resize và chuẩn hóa frame cho model MViT."""
    # 1. Resize: Bắt buộc phải 224x224
    frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    
    # 2. Chuẩn hóa về [0, 1] và chuyển sang float32
    frame_float = frame_resized.astype(np.float32) / 255.0
    
    # 3. CHÚ Ý: Bạn cần thêm bước chuẩn hóa (normalization) với mean/std 
    # giống như khi huấn luyện model MViT (thường là ImageNet/Kinetics)
    # Ví dụ (cần thay bằng giá trị thực tế bạn dùng):
    # mean = np.array([0.45, 0.45, 0.45])
    # std = np.array([0.225, 0.225, 0.225])
    # frame_norm = (frame_float - mean) / std 

    # Nếu không có thông tin mean/std, tạm thời dùng frame_float
    return frame_float
# Khởi tạo Video Capture (Thay đổi "input_video.mp4" thành 0 nếu dùng webcam)
cap = cv2.VideoCapture("test2.avi")

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Biến lưu trữ kết quả dự đoán gần nhất
last_prediction = (0, 0.0) # (label, confidence)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3.1. Tiền xử lý Frame hiện tại
    processed_frame = preprocess_frame(frame)
    
    # 3.2. Cập nhật Hàng đợi Frame
    frame_queue.append(processed_frame)

    # Đảm bảo hàng đợi không vượt quá kích thước cửa sổ trượt (Slide Window)
    if len(frame_queue) > T:
        # Nếu SLIDE_WINDOW=1, chỉ giữ lại T-1 frames trước đó, loại bỏ frame cũ nhất
        frame_queue = frame_queue[-T:] 

    # 3.3. Kiểm tra xem đã đủ T frames để dự đoán chưa
    if len(frame_queue) == T:
        # Chuyển Hàng đợi frames sang định dạng tensor PyTorch
        # (T, H, W, C) -> (C, T, H, W) -> (1, C, T, H, W)
        
        # Tạo numpy array từ list of frames: (T, H, W, C)
        clip_numpy = np.stack(frame_queue, axis=0) 
        
        # Chuyển đổi định dạng cho PyTorch: (T, H, W, C) -> (C, T, H, W)
        clip_tensor = torch.from_numpy(clip_numpy).permute(3, 0, 1, 2)
        
        # Thêm batch dimension: (1, C, T, H, W)
        clip_tensor = clip_tensor.unsqueeze(0).float() 
        
        # 3.4. CHẠY DỰ ĐOÁN
        label, conf = predict_fight_violence(clip_tensor, CKPT_PATH, NUM_CLASSES)
        
        # Cập nhật kết quả dự đoán gần nhất
        last_prediction = (label, conf)
        
        # 3.5. Xử lý Slide Window (Quan trọng)
        # Nếu SLIDE_WINDOW = 1, ta đã loại bỏ frame cũ nhất ở bước 3.2
        # Nếu SLIDE_WINDOW > 1, ta cần loại bỏ thêm SLIDE_WINDOW - 1 frames
        # Ví dụ: Nếu SLIDE_WINDOW=5, ta chỉ giữ lại 16-5=11 frames cũ nhất
        # frame_queue = frame_queue[SLIDE_WINDOW:] 
        
        # Ở đây, vì đã dùng frame_queue = frame_queue[-T:] ở bước 3.2 và T là kích thước cửa sổ dự đoán
        # nên việc trượt 1 frame (SLIDE_WINDOW=1) đã được tự động xử lý.
        
    # 3.6. HIỂN THỊ KẾT QUẢ TRÊN FRAME (Hiển thị kết quả gần nhất)
    display_text = f"Pred: {'NORMAL' if last_prediction[0] == 1 else 'FIGHT'}"
    confidence_text = f"Conf: {last_prediction[1]:.2f}"
    
    # Hiển thị nhãn
    cv2.putText(frame, display_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Hiển thị độ tin cậy
    cv2.putText(frame, confidence_text, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Hiển thị Frame
    cv2.imshow('Video Inference (MViT-V2)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()