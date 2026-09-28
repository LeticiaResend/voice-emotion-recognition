"""
Model 2 -- transfer learning: ResNet-18 pretrained on ImageNet,
backbone frozen, only a new FC head trained. The spectrogram is copied into
3 channels so it looks like an RGB image to the pretrained network.

Needs the actual ImageNet weights as a local .pth file, passed via
`pretrained_path`. Without one, the backbone is randomly initialized --
fine to check the code runs, but not real transfer learning.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_transfer_model(n_classes: int, pretrained_path: str | None = None) -> nn.Module:
    model = tvm.resnet18(weights=None)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # fc.* missing is expected (replaced below); anything else missing
        # or unexpected means the checkpoint doesn't match resnet18.
        bad_missing = [k for k in missing if not k.startswith("fc.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint at {pretrained_path} doesn't cleanly match torchvision's "
                f"resnet18 architecture.\n  unexpected keys: {unexpected}\n  "
                f"missing (non-fc) keys: {bad_missing}\nRefusing to silently use a "
                f"possibly-mismatched checkpoint."
            )
        print(f"Loaded pretrained weights from {pretrained_path} "
              f"({len(state_dict)} tensors, matched cleanly except the fc head).")
    else:
        print("WARNING: no pretrained_path given -- backbone is randomly initialized, "
              "this is NOT real transfer learning.")

    # Freeze the whole backbone first...
    for p in model.parameters():
        p.requires_grad = False

    # ...then replace and unfreeze just the head.
    in_features = model.fc.in_features  # 512 for resnet18
    model.fc = nn.Linear(in_features, n_classes)
    for p in model.fc.parameters():
        p.requires_grad = True

    return model


if __name__ == "__main__":
    model = build_transfer_model(n_classes=7, pretrained_path=None)  # no weights yet -- structure check only

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}  (slide 18 states ~11M)")
    print(f"Trainable parameters (just the new head): {trainable_params:,}")

    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}  (expect (4, 7))")
