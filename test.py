import cv2
import torch
import numpy as np
from infer import load_model_once, predict_fight_violence
from config import T, SLIDE_WINDOW, IMAGE_SIZE, NUM_CLASSES, CKPT_PATH, DEVICE, MEAN, STD

# Tải model (chỉ load 1 lần)
model = load_model_once(CKPT_PATH, NUM_CLASSES)

if model is None:
    print("Cannot proceed without a loaded model.")
    exit()

# Hàng đợi để lưu trữ 16 frames cần thiết
frame_queue = [] 

def preprocess_frame(frame):
    """Resize và chuẩn hóa frame cho model MViT."""
    # 1. Resize: Bắt buộc phải 224x224
    frame_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
    
    # 2. Chuẩn hóa về [0, 1] và chuyển sang float32
    frame_float = frame_resized.astype(np.float32) / 255.0
    
    # 3. NORMALIZE với mean/std giống như training
    mean = np.array(MEAN)
    std = np.array(STD)
    frame_norm = (frame_float - mean) / std 

    return frame_norm

# Khởi tạo Video Capture (Thay đổi "input_video.mp4" thành 0 nếu dùng webcam)
# cap = cv2.VideoCapture("./dataset/data/train/fight/1.avi")
cap = cv2.VideoCapture("test1.mp4")
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
        
        label, conf = predict_fight_violence(clip_tensor, CKPT_PATH, NUM_CLASSES)
        
        last_prediction = (label, conf)
        
    label = f"Pred: {'FIGHT' if last_prediction[0] == 1 else 'NORMAL'}"
    confidence_text = f"Conf: {last_prediction[1]:.2f}"
    
    # Hiển thị nhãn
    cv2.putText(frame, label, (10, 30), 
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