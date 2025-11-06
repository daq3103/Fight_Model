import os
from dataset_mvit import VideoFolderDataset

# Test dataset mapping
print("=== DEBUGGING DATASET MAPPING ===")

# Test với train dataset
train_dataset = VideoFolderDataset(
    root="./dataset/data/train",
    T=16
)

print(f"Classes found: {train_dataset.classes}")
print(f"Class to index mapping: {train_dataset.class_to_idx}")

# Kiểm tra một vài samples
print(f"\nFirst 5 samples:")
for i in range(min(5, len(train_dataset.samples))):
    path, label = train_dataset.samples[i]
    class_name = os.path.basename(os.path.dirname(path))
    print(f"  {class_name} -> label {label} (path: {os.path.basename(path)})")

# Load một sample thật để test
if len(train_dataset.samples) > 0:
    print(f"\nTesting actual data loading...")
    try:
        video, label = train_dataset[0]
        sample_path, sample_label = train_dataset.samples[0]
        class_name = os.path.basename(os.path.dirname(sample_path))
        print(f"Sample from '{class_name}' folder:")
        print(f"  - Raw label from dataset: {sample_label}")
        print(f"  - Processed label: {label}")
        print(f"  - Video shape: {video.shape}")
        print(f"  - Label tensor: {label}")
    except Exception as e:
        print(f"Error loading sample: {e}")

print("\n=== EXPECTED MAPPING ===")
print("nofight -> 0 (NORMAL)")
print("fight -> 1 (FIGHT)")