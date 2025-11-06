# MViT Fight Detection Configuration
import torch

# Model Configuration
NUM_CLASSES = 1
FREEZE_BACKBONE = True

# Training Configuration  
BATCH_SIZE = 4  
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# Data Configuration
T = 16  # Number of frames per clip
IMAGE_SIZE = 224  # Input spatial size
DATA_ROOT = "./dataset/data"
SLIDE_WINDOW = 1  # Frame sliding window for inference

# Model Path
CKPT_PATH = "mvit_v2_s_best.pt"

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Video Extensions
VIDEO_EXTENSIONS = ("mp4", "avi", "mov")

# Normalization Parameters (ImageNet/Kinetics standard)
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]

# Inference Configuration
CONFIDENCE_THRESHOLD = 0.5