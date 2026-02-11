import torch
from model_mvit import FightVideoModel
from config import DEVICE, NUM_CLASSES, CKPT_PATH, CONFIDENCE_THRESHOLD

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
        
        if sd["classifier_head.weight"].shape[0] != num_classes:
             raise ValueError(f"NUM_CLASSES (Config: {num_classes}) không khớp với Checkpoint (Output: {sd['classifier_head.weight'].shape[0]})")
             
        model.load_state_dict(sd)
        model.eval() 
        model.to(DEVICE)
        
        GLOBAL_MODEL = model # Lưu trữ model đã load
        print("Model loaded successfully.")
        return GLOBAL_MODEL
        
    except Exception as e:
        print(f"ERROR: Failed to load model or checkpoint mismatch. Details: {e}")
        return None
    

def predict_fight_violence(video_tensor: torch.Tensor, 
                            path_to_ckpt: str = CKPT_PATH, 
                            num_classes: int = NUM_CLASSES):

    model = load_model_once(path_to_ckpt, num_classes)
    
    input_tensor = video_tensor.to(DEVICE)
    
    with torch.no_grad():
        raw_output = model(input_tensor) 

    probability = torch.sigmoid(raw_output).squeeze().item()  

    label = 1 if probability >= CONFIDENCE_THRESHOLD else 0

    return label, probability