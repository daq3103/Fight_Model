import os
from glob import glob
from torch.utils.data import Dataset
import numpy as np
from decord import VideoReader
from decord import cpu as decord_cpu


def load_video_frames(path, T):
    vr = VideoReader(path, ctx=decord_cpu(0))
    num_frames = len(vr)
    if num_frames <= 0:
        raise RuntimeError(f"Cannot read frames from video: {path}")

    if num_frames < T:
        idxs = np.linspace(0, num_frames - 1, num=num_frames, dtype=np.int64)
    else:
        idxs = np.linspace(0, num_frames - 1, num=T, dtype=np.int64)

    batch = vr.get_batch(idxs).asnumpy()  # [T,H,W,3], uint8
    import torch as _torch
    video = _torch.from_numpy(batch).permute(3, 0, 1, 2).contiguous()  # [C,T,H,W]
    return video.float() / 255

def to_float_tensor(label):
    # Đảm bảo nhãn là float tensor có kích thước [1]
    return torch.tensor(label, dtype=torch.float).view(1)


class VideoFolderDataset(Dataset):
    def __init__(self, root, classes=None, T=16, transform=None, target_transform=None):

        self.root = root
        self.T = T
        self.transform = transform
        self.target_transform = target_transform

        # 1) Lấy danh sách lớp
        if classes is None:
            classes = sorted(
                [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            )
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        if 'FIGHT' in self.classes and 'NONFIGHT' in self.classes:
             self.class_to_idx = {'NONFIGHT': 0, 'FIGHT': 1}
        # 2) Gom danh sách video + nhãn
        exts = ("mp4", "avi")
        samples = []
        for c in self.classes:
            cdir = os.path.join(root, c)
            paths = []
            for ext in exts:
                paths += glob(os.path.join(cdir, f"*.{ext}"))
            samples += [(p, self.class_to_idx[c]) for p in paths]

        self.samples = samples  # [(path, label_idx), ...]

    def __getitem__(self, index):
        path, label = self.samples[index]
        video = load_video_frames(path, self.T)  # [C,T,H,W] float (0..1)

        if self.transform is not None:
            video = self.transform(video)  # vẫn [C,T,H,W]
        if self.target_transform is not None:
            label = self.target_transform(label)  # thường là int -> tensor


        label = torch.as_tensor(label, dtype=torch.float32).view(1)

        return video, label

    def __len__(self):
        return len(self.samples)


import torch
import torch.nn.functional as F


class VideoTransform:
    def __init__(self, size=224, train=True):
        self.size = size
        self.train = train
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

    def __call__(self, x):  
        # x is [C, T, H, W]; resize spatial dims treating T as batch
        x = x.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
        x = F.interpolate(
            x, size=(self.size, self.size), mode="bilinear", align_corners=False
        )
        x = x.permute(1, 0, 2, 3).contiguous()  # back to [C, T, H, W]
        if self.train and torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[3])  
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x
    
