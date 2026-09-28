"""
Dataset for Model 2 (transfer learning).

Turns the spectrogram into a pseudo-image for ResNet-18: 
dB -> clip -> scale to [0,1] -> resize to 224x224 -> repeat to 3 channels ->
normalize with ImageNet stats (frozen backbone expects that).

Different preprocessing from dataset.py (Model 1), which z-scores with our
own EmoDB stats. Not shared between the two on purpose.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from features import load_audio, fix_length, fix_length_random, melspectrogram_db, spec_augment

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DB_MIN, DB_MAX = -80.0, 0.0  
IMAGE_SIZE = 224

EMOTIONS = ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


def melspec_to_imagenet_input(mel_db: np.ndarray, size: int = IMAGE_SIZE) -> torch.Tensor:
    """(n_mels, n_frames) raw dB spectrogram -> (3, size, size) ImageNet-ready tensor."""
    clipped = np.clip(mel_db, DB_MIN, DB_MAX)
    scaled = (clipped - DB_MIN) / (DB_MAX - DB_MIN)  # [0, 1]

    x = torch.from_numpy(scaled).float().unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, n_frames)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.squeeze(0).repeat(3, 1, 1)  # (3, size, size)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


class EmoDBTransferDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, speakers: list[str],
                 augment: bool = False, seed: int = 0, audio_cache: dict | None = None):
        self.df = metadata_df[metadata_df["speaker_id"].astype(str).isin(speakers)].reset_index(drop=True)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.audio_cache = audio_cache if audio_cache is not None else {}
        for fp in self.df["filepath"]:
            if fp not in self.audio_cache:
                self.audio_cache[fp] = load_audio(fp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self.audio_cache[row["filepath"]]

        y = fix_length_random(y, rng=self.rng) if self.augment else fix_length(y)
        mel_db = melspectrogram_db(y)  # raw dB, NOT z-scored -- see module docstring
        if self.augment:
            mel_db = spec_augment(mel_db, mask_value=DB_MIN, rng=self.rng)

        x = melspec_to_imagenet_input(mel_db)
        label = EMOTION_TO_IDX[row["emotion"]]
        return x, label


if __name__ == "__main__":
    from paths import EMODB_METADATA_CSV
    df = pd.read_csv(EMODB_METADATA_CSV, dtype={"speaker_id": str})
    speakers = sorted(df["speaker_id"].unique())[:3]
    ds = EmoDBTransferDataset(df, speakers, augment=False)
    x, y = ds[0]
    print(f"one example: x.shape={tuple(x.shape)}, x.dtype={x.dtype}, "
          f"min={x.min():.2f}, max={x.max():.2f}, label={y} ({EMOTIONS[y]})")
    assert x.shape == (3, 224, 224)
    print("shape check passed.")
