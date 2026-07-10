"""Small CNN for single-channel DAPI tile classification.

Vendored from ../../../src/model.py (Checkpoint 1). Kept identical and
copied rather than imported across the two project trees on purpose: this
backend is deployed standalone (e.g. to a Hugging Face Space container),
which only ships the files inside PROJECT_INTERVIEW_WEB_APP/ -- it cannot
depend on a sibling `src/` package that won't exist in that container.
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
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x).squeeze(-1)
