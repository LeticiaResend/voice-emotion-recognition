from pathlib import Path

import torch
import torch.nn.functional as F

from model_scratch import ScratchCNN
from features import extract_features, standardize
from dataset import EMOTIONS


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "scratch_deploy.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = ScratchCNN(n_classes=len(EMOTIONS))

    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
	    state_dict = checkpoint["state_dict"]
    else:
        raise KeyError(
            f"Didnt find state_dict in checkpoint. "
            f"Keys found: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    mean = checkpoint["mean"]
    std = checkpoint["std"]

    return model, mean, std


MODEL, MEAN, STD = load_model()


def predict(audio_path: str):

    mel_db = extract_features(audio_path)

    
    mel_db = standardize(mel_db, MEAN, STD)


    x = torch.from_numpy(mel_db).float()

    x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)

    
    with torch.no_grad():
        logits = MODEL(x)
        probabilities = F.softmax(logits, dim=1)[0]

    predicted_idx = probabilities.argmax().item()
    predicted_emotion = EMOTIONS[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    all_probabilities = {
        emotion: float(probabilities[i].item())
        for i, emotion in enumerate(EMOTIONS)
    }

    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": all_probabilities,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Uso:")
        print("python src/predict.py caminho/para/audio.wav")
        raise SystemExit(1)

    result = predict(sys.argv[1])

    print("\nPredicao:")
    print("Emotion:", result["emotion"])
    print(f"Confidence: {result['confidence']:.2%}")

    print("\nProbabilidades:")
    for emotion, prob in sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"{emotion:12s}: {prob:.2%}")