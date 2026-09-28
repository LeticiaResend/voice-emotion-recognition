"""
Model 1 -- a CNN trained from scratch,:
    Four blocks: two 3x3 convolutions + batch-norm + ReLU, then 2x2 max-pooling;
    32 -> 64 -> 128 -> 256 channels.
    Global average pooling -> dropout -> one fully connected layer.
    Small on purpose: 1.2M parameters, trains on a laptop CPU.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two 3x3 conv+BN+ReLU, then 2x2 max-pool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.net(x)


class ScratchCNN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, n_frames)
        x = self.blocks(x)
        x = self.global_pool(x).flatten(1)  # (batch, 256)
        x = self.dropout(x)
        return self.fc(x)  # (batch, n_classes)


if __name__ == "__main__":
    model = ScratchCNN(n_classes=7)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print(f"(slide 17 states ~1.2M -- {'matches' if 0.9e6 < n_params < 1.6e6 else 'MISMATCH, check architecture'})")

    dummy = torch.randn(4, 1, 64, 301)  # batch of 4, matching features.py output shape
    out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (expect (4, 7))")

    # Show the spatial size shrinking through each block, useful for sanity-checking pooling.
    x = dummy
    for i, block in enumerate(model.blocks):
        x = block(x)
        print(f"  after block {i+1}: {tuple(x.shape)}")
