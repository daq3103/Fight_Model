import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
# Giả định các lớp được import từ các file đã cung cấp:
# - VideoFolderDataset, VideoTransform, to_float_tensor từ dataset_mvit.py
# - FightVideoModel từ model_mvit.py
# LƯU Ý: Nếu bạn chạy file này, hãy đảm bảo các file dataset_mvit.py và model_mvit.py 
# đã được định nghĩa đúng cách trong cùng thư mục.
from dataset_mvit import VideoFolderDataset, VideoTransform
# Hàm to_float_tensor được định nghĩa lại ở đây để đảm bảo tính độc lập nếu cần
def to_float_tensor(label):
    """Đảm bảo nhãn là float tensor có kích thước [1] cho BCEWithLogitsLoss."""
    return torch.tensor(label, dtype=torch.float).view(1) 

from model_mvit import FightVideoModel


def get_args():
    """Định nghĩa và lấy các tham số đầu vào từ dòng lệnh."""
    parser = argparse.ArgumentParser(description="Training script for MViT-V2-S based video classification.")
    
    # Cấu hình chính (Có thể thay đổi qua dòng lệnh)
    parser.add_argument('--data_root', type=str, default='./dataset/data',
                        help='Root directory for the dataset (containing train/val folders).')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Input batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--T', type=int, default=16,
                        help='Number of frames per video clip (T).')
    parser.add_argument('--freeze_backbone', type=bool, default=True,
                        help='Whether to freeze the backbone (MViT) layers for transfer learning.')
    parser.add_argument('--checkpoint_file', type=str, default='mvit_v2_s_best.pt',
                        help='Output filename for the best model checkpoint.')

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Thực hiện một vòng lặp huấn luyện (epoch)."""
    model.train()
    total_loss = 0
    
    for _, (videos, labels) in enumerate(dataloader):
        videos, labels = videos.to(device), labels.to(device)
        
        # Zero Gradients
        optimizer.zero_grad()
        
        # Forward Pass (Outputs là logits)
        outputs = model(videos) 
        
        # Tính Loss
        loss = criterion(outputs, labels)
        
        # Backward Pass & Optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Đánh giá model trên tập kiểm tra/validation."""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Tính toán độ chính xác (Accuracy) cho nhị phân
            probabilities = torch.sigmoid(outputs)
            # Chuyển xác suất > 0.5 thành 1 (Có), còn lại là 0 (Không)
            predictions = (probabilities > 0.5).long() 
            
            # So sánh dự đoán với nhãn mục tiêu (labels.long() là 0 hoặc 1)
            correct_predictions += (predictions.view(-1) == labels.view(-1).long()).sum().item() 
            total_samples += labels.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def main():
    """Hàm chính thực thi toàn bộ quy trình huấn luyện."""
    args = get_args()
    
    # Áp dụng cấu hình từ dòng lệnh
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 1 
    
    # Khởi tạo DataLoader
    train_transform = VideoTransform(size=224, train=True)
    val_transform = VideoTransform(size=224, train=False)

    train_dataset = VideoFolderDataset(
        root=os.path.join(args.data_root, "train"), 
        T=args.T, 
        transform=train_transform,
        target_transform=to_float_tensor
    )
    val_dataset = VideoFolderDataset(
        root=os.path.join(args.data_root, "val"), 
        T=args.T, 
        transform=val_transform,
        target_transform=to_float_tensor
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )

    # Khởi tạo Model, Loss, Optimizer
    model = FightVideoModel(num_classes=NUM_CLASSES, freeze_backbone=args.freeze_backbone).to(DEVICE) 
    criterion = nn.BCEWithLogitsLoss()

    # Lọc tham số cần huấn luyện (chỉ classifier_head nếu freeze_backbone=True)
    if args.freeze_backbone:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    else:
        trainable_params = model.parameters()
        
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)

    # Vòng lặp chính và Checkpointing
    best_val_acc = 0.0
    
    print(f"Bắt đầu Huấn luyện trên thiết bị: {DEVICE}")
    print(f"Tổng số tham số có thể huấn luyện: {sum(p.numel() for p in trainable_params)}")

    for epoch in range(args.num_epochs):
        start_time = time.time()
        
        # Huấn luyện
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Đánh giá
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        end_time = time.time()
        
        print(f"--- Epoch {epoch+1}/{args.num_epochs} | Time: {end_time - start_time:.2f}s ---")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Logic Lưu Checkpoint Tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"--> Đã tìm thấy mô hình tốt hơn! Lưu checkpoint: {args.checkpoint_file}")
            # Lưu state_dict (trọng số) của model
            torch.save(model.state_dict(), args.checkpoint_file)
    
    print("\nQUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT.")
    print(f"Mô hình tốt nhất (Best Val Acc: {best_val_acc:.4f}) đã được lưu tại: {args.checkpoint_file}")


if __name__ == '__main__':
    main()