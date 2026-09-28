"""
Speaker-independent evaluation for EmoDB (slide 7): a random split leaks
speaker identity into train/test, so the model can learn to recognize the
*speaker* instead of the *emotion* 

5 outer folds via GroupKFold on speaker_id: each speaker is the test
speaker in exactly one fold. Within a fold's 8 train speakers, 1 more is
held out as val (early stopping) -- so every stage evaluates on unseen
voices.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from paths import EMODB_METADATA_CSV as METADATA_CSV, EMODB_FOLDS_JSON as OUT_JSON

N_FOLDS = 5


def build_folds(df: pd.DataFrame) -> list[dict]:
    speakers_sorted = sorted(df["speaker_id"].unique())
    gkf = GroupKFold(n_splits=N_FOLDS)

    folds = []
    for fold_id, (train_val_idx, test_idx) in enumerate(
        gkf.split(df, groups=df["speaker_id"])
    ):
        test_speakers = sorted(df.iloc[test_idx]["speaker_id"].unique())
        train_val_speakers = sorted(df.iloc[train_val_idx]["speaker_id"].unique())

        # One validation speaker per fold, rotating deterministically through
        # the remaining speakers so different folds don't all reuse speaker 03.
        val_speaker = train_val_speakers[fold_id % len(train_val_speakers)]
        train_speakers = [s for s in train_val_speakers if s != val_speaker]

        folds.append({
            "fold": fold_id,
            "train_speakers": train_speakers,
            "val_speaker": val_speaker,
            "test_speakers": test_speakers,
        })

    return folds


def verify_folds(df: pd.DataFrame, folds: list[dict]) -> None:
    all_speakers = set(df["speaker_id"].unique())

    # Every speaker must be the test speaker in exactly one fold.
    test_speaker_counts = {}
    for fold in folds:
        for s in fold["test_speakers"]:
            test_speaker_counts[s] = test_speaker_counts.get(s, 0) + 1
    assert set(test_speaker_counts) == all_speakers, "not every speaker was held out as test somewhere"
    assert all(c == 1 for c in test_speaker_counts.values()), "a speaker was held out as test more than once"

    # Within each fold: train / val / test speakers must be disjoint.
    for fold in folds:
        train_s = set(fold["train_speakers"])
        val_s = {fold["val_speaker"]}
        test_s = set(fold["test_speakers"])
        assert train_s.isdisjoint(val_s), f"fold {fold['fold']}: val speaker also in train"
        assert train_s.isdisjoint(test_s), f"fold {fold['fold']}: test speaker also in train"
        assert val_s.isdisjoint(test_s), f"fold {fold['fold']}: val speaker also in test"
        assert train_s | val_s | test_s == all_speakers, f"fold {fold['fold']}: speakers don't add up to all 10"

    print("All checks passed: every speaker held out as test exactly once, "
          "and train/val/test speakers never overlap within a fold.\n")


def summarize_folds(df: pd.DataFrame, folds: list[dict]) -> None:
    for fold in folds:
        test_df = df[df["speaker_id"].isin(fold["test_speakers"])]
        val_df = df[df["speaker_id"] == fold["val_speaker"]]
        train_df = df[df["speaker_id"].isin(fold["train_speakers"])]

        print(f"Fold {fold['fold']}: "
              f"train speakers={fold['train_speakers']} ({len(train_df)} clips), "
              f"val speaker={fold['val_speaker']} ({len(val_df)} clips), "
              f"test speakers={fold['test_speakers']} ({len(test_df)} clips)")
        test_counts = test_df["emotion"].value_counts().to_dict()
        print(f"          test-set emotion counts: {test_counts}")


if __name__ == "__main__":
    df = pd.read_csv(METADATA_CSV, dtype={"speaker_id": str})

    folds = build_folds(df)
    verify_folds(df, folds)
    summarize_folds(df, folds)

    Path(OUT_JSON).write_text(json.dumps(folds, indent=2))
    print(f"\nSaved fold definitions to {OUT_JSON}")
