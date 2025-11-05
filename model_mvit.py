import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

class FightVideoModel(nn.Module):
    def __init__(self, num_classes,freeze_backbone = False):
        super().__init__()
        weights = MViT_V2_S_Weights.KINETICS400_V1
        self.backbone = mvit_v2_s(weights=weights)

        in_features = self.backbone.head[1].in_features
        self.backbone.head = nn.Identity()
        self.classifier_head = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            for param in self.classifier_head.parameters():
                param.requires_grad = True
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier_head(features)
        return output

