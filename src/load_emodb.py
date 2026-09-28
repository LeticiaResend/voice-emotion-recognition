"""
Load EmoDB (Berlin Database of Emotional Speech) into a metadata table.

Filename convention (official EmoDB docs), e.g. "03a01Fa.wav":
    [0:2]  speaker id        -> "03"
    [2:5]  text code         -> "a01" (which of the 10 sentences was spoken)
    [5]    emotion letter    -> "F"   (see EMOTION_MAP below)
    [6]    version letter    -> "a"   (recording take, if the corpus has
                                        several distinct recordings of the
                                        same speaker/text/emotion combo)
"""

from pathlib import Path
import pandas as pd

EMOTION_MAP = {
    "W": "anger",      
    "L": "boredom",     
    "E": "disgust",    
    "A": "fear",        
    "F": "happiness",   
    "T": "sadness",  
    "N": "neutral",     
}

# Speaker id -> (age, sex), from the official EmoDB documentation.
SPEAKER_INFO = {
    "03": (31, "male"), "08": (34, "female"), "09": (21, "female"),
    "10": (32, "male"), "11": (26, "male"), "12": (30, "male"),
    "13": (32, "female"), "14": (35, "female"), "15": (25, "male"),
    "16": (31, "female"),
}


def load_emodb(wav_dir: str | Path) -> pd.DataFrame:
    wav_dir = Path(wav_dir)
    rows = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        stem = wav_path.stem  # e.g. "03a01Fa"
        speaker_id = stem[0:2]
        text_code = stem[2:5]
        emotion_letter = stem[5]
        version = stem[6:] if len(stem) > 6 else ""

        if emotion_letter not in EMOTION_MAP:
            raise ValueError(f"Unrecognised emotion letter {emotion_letter!r} in {wav_path.name}")

        age, sex = SPEAKER_INFO.get(speaker_id, (None, None))
        rows.append({
            "filepath": str(wav_path),
            "filename": wav_path.name,
            "speaker_id": speaker_id,
            "speaker_age": age,
            "speaker_sex": sex,
            "text_code": text_code,
            "emotion_letter": emotion_letter,
            "emotion": EMOTION_MAP[emotion_letter],
            "version": version,
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    from paths import EMODB_WAV_DIR, EMODB_METADATA_CSV

    df = load_emodb(EMODB_WAV_DIR)

    print(f"Total clips: {len(df)}")
    print(f"Speakers: {sorted(df['speaker_id'].unique())} ({df['speaker_id'].nunique()} total)")
    print()
    print("Clips per emotion:")
    print(df["emotion"].value_counts())
    print()
    print("Clips per speaker:")
    print(df["speaker_id"].value_counts().sort_index())
    print()
    print("Sample rows:")
    print(df.sample(5, random_state=0).to_string(index=False))

    df.to_csv(EMODB_METADATA_CSV, index=False)
    print(f"\nSaved metadata to {EMODB_METADATA_CSV}")
