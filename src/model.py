"""Small CNN for single-channel DAPI tile classification.

Trained from scratch (no pretrained weights) -- at 64x64, single-channel
input, a small custom CNN is fast to train on CPU and entirely sufficient for
a binary clean/artifact decision. A pretrained ResNet-18/EfficientNet-B0 is a
documented alternative (see project docs) for when more capacity is wanted,
but it requires replicating the DAPI channel to 3 channels or editing the
first conv layer, and is unnecessary at this problem's scale.
"""

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global average pool -> size-agnostic
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(-1)  # raw logits (use BCEWithLogitsLoss / sigmoid)
