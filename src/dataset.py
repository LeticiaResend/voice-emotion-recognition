"""
PyTorch Dataset over EmoDB, built on top of features.py.
Train-time: random crop/pad + SpecAugment
Eval-time (val/test/app): deterministic center-crop/pad, no masking.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features import (
    load_audio, fix_length, fix_length_random, melspectrogram_db,
    standardize, spec_augment,
)

# Fixed label order, shared everywhere.
EMOTIONS = ["anger", "boredom", "disgust", "fear", "happiness", "neutral", "sadness"]
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}


class EmoDBDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, speakers: list[str],
                 mean: float, std: float, augment: bool = False, seed: int = 0,
                 audio_cache: dict | None = None):
        self.df = metadata_df[metadata_df["speaker_id"].astype(str).isin(speakers)].reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        # Decode each clip's waveform once and reuse across epochs (loading
        # from disk is the slow part; only crop/pad/mask change per epoch)
        self.audio_cache = audio_cache if audio_cache is not None else {}
        for fp in self.df["filepath"]:
            if fp not in self.audio_cache:
                self.audio_cache[fp] = load_audio(fp)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self.audio_cache[row["filepath"]]

        if self.augment:
            y = fix_length_random(y, rng=self.rng)
        else:
            y = fix_length(y)

        mel_db = melspectrogram_db(y)
        mel_db = standardize(mel_db, self.mean, self.std)

        if self.augment:
            mel_db = spec_augment(mel_db, rng=self.rng)

        x = torch.from_numpy(mel_db).unsqueeze(0)  # (1, n_mels, n_frames)
        label = EMOTION_TO_IDX[row["emotion"]]
        return x, label


if __name__ == "__main__":
    import json
    from features import fit_standardizer, extract_features

    from paths import EMODB_METADATA_CSV, EMODB_FOLDS_JSON
    df = pd.read_csv(EMODB_METADATA_CSV, dtype={"speaker_id": str})
    folds = json.loads(open(EMODB_FOLDS_JSON).read())
    fold0 = folds[0]

    train_feats = [extract_features(fp) for fp in
                   df[df["speaker_id"].isin(fold0["train_speakers"])]["filepath"]]
    mean, std = fit_standardizer(train_feats)

    train_ds = EmoDBDataset(df, fold0["train_speakers"], mean, std, augment=True)
    val_ds = EmoDBDataset(df, [fold0["val_speaker"]], mean, std, augment=False)
    test_ds = EmoDBDataset(df, fold0["test_speakers"], mean, std, augment=False)

    print(f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    x, y = train_ds[0]
    print(f"one training example: x.shape={tuple(x.shape)}, x.dtype={x.dtype}, label={y} ({EMOTIONS[y]})")

    x2, _ = train_ds[0]
    print(f"augmentation is randomized: two draws of the same clip differ = {not torch.allclose(x, x2)}")

    x_eval_a, _ = val_ds[0]
    x_eval_b, _ = val_ds[0]
    print(f"eval path is deterministic: two draws of the same clip are identical = {torch.allclose(x_eval_a, x_eval_b)}")
