# Nhiệm vụ: inference 1 video hoặc nhiều video
# Input: path video + checkpoint
# Output: label + probabilities

# Giả sử num_classes = 1 (như trong config của bạn)
import torch
from model_mvit import FightVideoModel

GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOBAL_MODEL = None
def load_model_once(path_to_ckpt, num_classes):
    """Tải và trả về model đã load trọng số (chỉ chạy lần đầu)."""
    global GLOBAL_MODEL
    
    if GLOBAL_MODEL is not None:
        return GLOBAL_MODEL

    print(f"Loading model from {path_to_ckpt}...")
    
    # Khởi tạo mô hình
    model = FightVideoModel(num_classes=num_classes, freeze_backbone=False)

    # Tải trọng số
    try:
        ckpt = torch.load(path_to_ckpt, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        
        # Đảm bảo num_classes khớp với số lớp đầu ra trong checkpoint
        if sd["classifier_head.weight"].shape[0] != num_classes:
             raise ValueError(f"NUM_CLASSES (Config: {num_classes}) không khớp với Checkpoint (Output: {sd['classifier_head.weight'].shape[0]})")
             
        model.load_state_dict(sd)
        model.eval() # Bắt buộc phải chuyển sang chế độ đánh giá
        model.to(GLOBAL_DEVICE)
        
        GLOBAL_MODEL = model # Lưu trữ model đã load
        print("Model loaded successfully.")
        return GLOBAL_MODEL
        
    except Exception as e:
        print(f"ERROR: Failed to load model or checkpoint mismatch. Details: {e}")
        return None
    

def predict_fight_violence(video_tensor: torch.Tensor, 
                            path_to_ckpt: str = "mvit_v2_s_best.pt", 
                            num_classes: int = 1):
    """
    Thực hiện suy luận (inference) để dự đoán nhãn và độ tin cậy.

    Args:
        video_tensor (torch.Tensor): Tensor video đã được chuẩn hóa,
                                     có kích thước (N, C, T, H, W).
                                     Ví dụ: (1, 3, 16, 224, 224).
        path_to_ckpt (str): Đường dẫn đến file trọng số .pt.
        num_classes (int): Số lượng lớp dự đoán (mặc định là 1 cho binary).

    Returns:
        tuple: (label: int, confidence: float)
    """
    model = load_model_once(path_to_ckpt, num_classes)
    
    # 1. Chuyển tensor đầu vào về đúng device
    input_tensor = video_tensor.to(GLOBAL_DEVICE)
    
    # 2. Chạy suy luận (inference)
    with torch.no_grad():
        raw_output = model(input_tensor) 

    probability = torch.sigmoid(raw_output).squeeze().item()  

    # Trả về label dạng int (0 hoặc 1) - với model mới đã đúng mapping
    label = 1 if probability >= 0.5 else 0

    return label, probability