import torch

# Kiểm tra thông tin trong checkpoint
ckpt = torch.load("mvit_v2_s_best.pt", map_location="cpu")

print("=== CHECKPOINT INFO ===")
print(f"Keys in checkpoint: {list(ckpt.keys())}")

if "config" in ckpt:
    print(f"\nTraining config in checkpoint:")
    for key, value in ckpt["config"].items():
        print(f"  {key}: {value}")

if "epoch" in ckpt:
    print(f"\nEpoch: {ckpt['epoch']}")
    print(f"Val loss: {ckpt.get('val_loss', 'N/A')}")
    print(f"Val acc: {ckpt.get('val_acc', 'N/A')}")

# Kiểm tra classifier head weight
if "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
    if "classifier_head.weight" in sd:
        classifier_weight = sd["classifier_head.weight"]
        print(f"\nClassifier head shape: {classifier_weight.shape}")
        print(f"Classifier weight values: {classifier_weight.squeeze()}")
        print(f"Classifier bias: {sd.get('classifier_head.bias', 'N/A')}")