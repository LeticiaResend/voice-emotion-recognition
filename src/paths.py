"""
Central place for project paths, so every script works no matter where the
project folder lives on disk (this sandbox, your laptop, wherever) instead
of being hardcoded to one machine.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../voice_emotion_project
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
APP_DIR = PROJECT_ROOT / "app"

EMODB_WAV_DIR = RAW_DIR / "emodb" / "wav"
EMODB_METADATA_CSV = PROCESSED_DIR / "emodb_metadata.csv"
EMODB_FOLDS_JSON = PROCESSED_DIR / "emodb_folds.json"
RESNET18_WEIGHTS = MODELS_DIR / "resnet18-f37072fd.pth"

# Make sure the directories we write into exist.
for d in (PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
