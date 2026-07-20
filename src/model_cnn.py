from __future__ import annotations

import torch
from torch import nn


class NoiseCNN(nn.Module):
    """CNN with optional structured auxiliary heads for source recognition."""

    def __init__(self, num_classes: int, auxiliary_heads: bool = False):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.num_classes = num_classes
        self.auxiliary_heads = bool(auxiliary_heads)
        self.classifier = nn.Linear(64, num_classes)
        if self.auxiliary_heads:
            self.combo_classifier = nn.Linear(64, (2**num_classes) - 1)
            self.count_classifier = nn.Linear(64, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))

    def forward_with_auxiliary(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.auxiliary_heads:
            raise RuntimeError("Auxiliary heads are not enabled for this model")
        embedding = self.encode(x)
        return (
            self.classifier(embedding),
            self.combo_classifier(embedding),
            self.count_classifier(embedding),
        )
