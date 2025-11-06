# Nhiệm vụ: training loop chính
# Input: train/val dataloader, config, model
# Output: best checkpoint file (ex: mvit_v2_s_best.pt)
import os
import time
import argparse
import torch.nn as nn
import torch
from dataset_mvit import VideoFolderDataset, VideoTransform, to_float_tensor
from model_mvit import FightVideoModel
from torch.utils.data import DataLoader
from config import *


criterion = nn.BCEWithLogitsLoss()

def train(
    data_root: str = DATA_ROOT,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = NUM_EPOCHS,
    t: int = T,
    freeze_backbone: bool = FREEZE_BACKBONE,
    num_workers: int = 0,
    size: int = IMAGE_SIZE,
    ckpt_path: str = CKPT_PATH,
):
    # chuẩn bị data train/val theo tham số
    train_transform = VideoTransform(size=size, train=True)
    val_transform = VideoTransform(size=size, train=False)

    train_dataset = VideoFolderDataset(
        root=os.path.join(data_root, "train"),
        T=t,
        transform=train_transform,
        target_transform=to_float_tensor,
    )
    val_dataset = VideoFolderDataset(
        root=os.path.join(data_root, "val"),
        T=t,
        transform=val_transform,
        target_transform=to_float_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = FightVideoModel(num_classes=NUM_CLASSES, freeze_backbone=freeze_backbone).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()


        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for videos, labels in train_loader:
            videos = videos.to(DEVICE)  
            labels = labels.to(DEVICE)  

            optimizer.zero_grad()
            logits = model(videos)  
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * videos.size(0)

            with torch.no_grad():
                preds = torch.sigmoid(logits)
                preds_cls = (preds >= 0.5).float()
                running_correct += (preds_cls == labels).sum().item()
                running_total += labels.numel()

        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_acc = running_correct / max(1, running_total)


        model.eval()
        val_running_loss = 0.0
        val_running_correct = 0
        val_running_total = 0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(videos)
                loss = criterion(logits, labels)

                val_running_loss += loss.item() * videos.size(0)

                preds = torch.sigmoid(logits)
                preds_cls = (preds >= 0.5).float()
                val_running_correct += (preds_cls == labels).sum().item()
                val_running_total += labels.numel()

        val_loss = val_running_loss / max(1, len(val_loader.dataset))
        val_acc = val_running_correct / max(1, val_running_total)

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={epoch_time:.1f}s"
        )

        # Save best checkpoint by val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": {
                    "NUM_CLASSES": NUM_CLASSES,
                    "BATCH_SIZE": batch_size,
                    "LEARNING_RATE": learning_rate,
                    "NUM_EPOCHS": num_epochs,
                    "T": t,
                    "FREEZE_BACKBONE": freeze_backbone,
                    "DATA_ROOT": data_root,
                    "IMAGE_SIZE": size,
                    "NUM_WORKERS": num_workers,
                },
            }, ckpt_path)
            print(f"Saved new best checkpoint at epoch {epoch} -> {ckpt_path}")

    print(f"Training done. Best epoch: {best_epoch} with val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--t", type=int, default=T, help="Number of frames per clip")
    # Proper toggles for freezing backbone
    freeze_group = parser.add_mutually_exclusive_group()
    freeze_group.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true")
    freeze_group.add_argument("--no-freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.set_defaults(freeze_backbone=FREEZE_BACKBONE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Input spatial size")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)

    args = parser.parse_args()

    train(
        data_root=args.data_root,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        t=args.t,
        freeze_backbone=args.freeze_backbone,
        num_workers=args.num_workers,
        size=args.size,
        ckpt_path=args.ckpt_path,
    )