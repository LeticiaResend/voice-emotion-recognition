from pathlib import Path
import sys

import gradio as gr
import matplotlib.pyplot as plt
import librosa.display
import os

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict import predict
from features import load_audio, fix_length, melspectrogram_db


def make_spectrogram(audio_path):
    y = load_audio(audio_path)
    y = fix_length(y)
    mel_db = melspectrogram_db(y)

    fig, ax = plt.subplots(figsize=(10, 4))

    librosa.display.specshow(
        mel_db,
        sr=16000,
        hop_length=160,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )

    ax.set_title("Mel-spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel frequency")

    fig.tight_layout()

    return fig


def analyze_audio(audio_path):
    if audio_path is None:
        return "No audio provided.", None, {}

    result = predict(audio_path)

    emotion = result["emotion"]
    confidence = result["confidence"]

    label = f"{emotion.capitalize()} — {confidence:.1%}"

    probabilities = {
        emotion_name.capitalize(): probability
        for emotion_name, probability in result["probabilities"].items()
    }

    spectrogram = make_spectrogram(audio_path)

    return label, spectrogram, probabilities


with gr.Blocks(title="Voice Emotion Recognition") as demo:

    gr.Markdown(
        """
        # Voice Emotion Recognition

        Record your voice or upload an audio file.

        The CNN will analyze the speech and predict the emotional state.
        """
    )

    audio = gr.Audio(
        sources=["microphone", "upload"],
        type="filepath",
        label="Record or upload audio",
    )

    analyze_button = gr.Button("Analyze emotion")

    prediction = gr.Textbox(
        label="Predicted emotion",
        interactive=False,
    )

    spectrogram = gr.Plot(
        label="Mel-spectrogram"
    )

    probabilities = gr.Label(
        label="Emotion probabilities",
        num_top_classes=7,
    )

    analyze_button.click(
        fn=analyze_audio,
        inputs=audio,
        outputs=[
            prediction,
            spectrogram,
            probabilities,
        ],
    )

if __name__ == "__main__":
    demo.launch()
