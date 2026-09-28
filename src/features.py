"""
Audio -> fixed-size mel-spectrogram, per slide 9 of the course deck:
    - resample to 16 kHz mono
    - fix to a 3-second window (pad short clips, center-crop long ones)
    - short-time spectrum: 25 ms frames every 10 ms
    - mel scale: 64 bands, in dB, standardised

Shared by training (EmoDB files) and the app's inference API (micrecordings) 
-- resampling stays in even though EmoDB is already 16kHz,
since a browser mic is usually more.
"""

import numpy as np
import librosa

SR = 16000
DURATION_S = 3.0
TARGET_LEN = int(SR * DURATION_S)  # 48000 samples

N_FFT = 400          # 25 ms at 16 kHz
HOP_LENGTH = 160      # 10 ms at 16 kHz
N_MELS = 64


def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Load any audio file, resampled to mono `sr` Hz."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def fix_length(y: np.ndarray, target_len: int = TARGET_LEN) -> np.ndarray:
    """Center-crop if longer than target, zero-pad (symmetric) if shorter.

    Deterministic -- used for validation/test and for the live app, where we
    want the same clip to always produce the same features.
    """
    n = len(y)
    if n == target_len:
        return y
    if n > target_len:
        start = (n - target_len) // 2
        return y[start: start + target_len]
    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant")


def fix_length_random(y: np.ndarray, target_len: int = TARGET_LEN,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Same idea as fix_length, but the crop/pad position is randomized.

    This is the "random time shifts" augmentation slide 17 mentions for
    Model 1 -- used only during training, never for validation/test/app.
    """
    rng = rng or np.random.default_rng()
    n = len(y)
    if n == target_len:
        return y
    if n > target_len:
        start = int(rng.integers(0, n - target_len + 1))
        return y[start: start + target_len]
    pad_total = target_len - n
    pad_left = int(rng.integers(0, pad_total + 1))
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant")


def spec_augment(mel_db: np.ndarray, n_time_masks: int = 2, time_mask_width: int = 30,
                  n_freq_masks: int = 2, freq_mask_width: int = 8,
                  mask_value: float = 0.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """SpecAugment: mask random time and frequency stripes.
    Applied only to training examples, on the *standardized*
    spectrogram (so mask_value=0.0 corresponds to "the dataset average").
    """
    rng = rng or np.random.default_rng()
    mel_db = mel_db.copy()
    n_mels, n_frames = mel_db.shape

    for _ in range(n_time_masks):
        width = int(rng.integers(0, time_mask_width + 1))
        if width == 0 or width >= n_frames:
            continue
        start = int(rng.integers(0, n_frames - width))
        mel_db[:, start:start + width] = mask_value

    for _ in range(n_freq_masks):
        width = int(rng.integers(0, freq_mask_width + 1))
        if width == 0 or width >= n_mels:
            continue
        start = int(rng.integers(0, n_mels - width))
        mel_db[start:start + width, :] = mask_value

    return mel_db


def melspectrogram_db(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Waveform (fixed length) -> (N_MELS, n_frames) log-mel spectrogram in dB."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def extract_features(path: str) -> np.ndarray:
    """One audio file -> a fixed-size (N_MELS, n_frames) log-mel spectrogram."""
    y = load_audio(path)
    y = fix_length(y)
    return melspectrogram_db(y)


def fit_standardizer(mel_dbs: list[np.ndarray]) -> tuple[float, float]:
    """Mean/std over a *training* set of spectrograms, to standardize with."""
    stacked = np.concatenate([m.ravel() for m in mel_dbs])
    return float(stacked.mean()), float(stacked.std())


def standardize(mel_db: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (mel_db - mean) / (std + 1e-8)


if __name__ == "__main__":
    import pandas as pd
    from paths import EMODB_METADATA_CSV, EMODB_FOLDS_JSON

    df = pd.read_csv(EMODB_METADATA_CSV)

    # 1. Confirm every clip produces the exact same feature shape.
    shapes = set()
    all_feats = []
    for fp in df["filepath"]:
        feat = extract_features(fp)
        shapes.add(feat.shape)
        all_feats.append(feat)
    print(f"Distinct output shapes across all 535 clips: {shapes}")
    assert len(shapes) == 1, "clips are producing inconsistent feature shapes!"

    # 2. Sanity-check the raw (pre-standardization) dB values.
    stacked = np.concatenate([f.ravel() for f in all_feats])
    print(f"Raw log-mel dB values: min={stacked.min():.1f}, max={stacked.max():.1f}, "
          f"mean={stacked.mean():.1f}, std={stacked.std():.1f}")
    assert not np.isnan(stacked).any(), "NaNs in the features!"

    # 3. Standardize with fold 0's train speakers only (no leakage), then
    #    check the standardized values land close to N(0, 1).
    import json
    folds = json.loads(open(EMODB_FOLDS_JSON).read())
    fold0 = folds[0]
    train_mask = df["speaker_id"].astype(str).isin(fold0["train_speakers"])
    train_feats = [f for f, m in zip(all_feats, train_mask) if m]
    mean, std = fit_standardizer(train_feats)
    print(f"\nFold 0 train-set standardizer: mean={mean:.2f}, std={std:.2f}")

    standardized_train = np.concatenate([standardize(f, mean, std).ravel() for f in train_feats])
    print(f"Standardized train values: mean={standardized_train.mean():.3f}, std={standardized_train.std():.3f}")
