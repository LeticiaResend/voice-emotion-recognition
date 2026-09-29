
# Voice Emotion Recognition — CNN project

Speech emotion recognition system built for the 5RO11 course assignment at
ENSTA Paris: two CNNs (one trained from scratch, one via transfer learning)
that classify German speech recordings into one of seven emotions,
evaluated on speakers never seen during training, with a live web app for
real-time predictions from a recorded voice.

**Authors:** Letícia Maria Resende, Beatriz Araujo Cavalcante — ENSTA Paris, 2026

## Overview

- **Task:** classify a short speech clip into one of 7 emotions (anger,
  boredom, disgust, fear, happiness, neutral, sadness).
- **Data:** EmoDB (Berlin Database of Emotional Speech), 535 clips, 10
  German speakers.
- **Models:** a 4-block CNN trained from scratch on mel-spectrograms
  (Model 1), and a frozen ImageNet-pretrained ResNet-18 with a new
  classification head (Model 2, transfer learning).
- **Evaluation:** 5-fold leave-speakers-out cross-validation, so every
  reported number comes from speakers the model never trained on.
- **Demo:** a Gradio web app records or accepts an uploaded voice clip,
  shows its mel-spectrogram, and predicts the emotion live.

## What's here

```
data/
  raw/emodb/wav/           535 EmoDB audio clips -- see "Get the data" below
  processed/                generated metadata + fold definitions
models/
  resnet18-f37072fd.pth     see "Model 2 setup" below
  scratch_deploy.pt         Model 1 checkpoint used by the live app
  transfer_deploy.pt        Model 2 checkpoint (fold 0), kept for reference
src/
  paths.py                 central path config -- everything else imports from here
  load_emodb.py            parses filenames -> data/processed/emodb_metadata.csv
  make_splits.py           builds the 5-fold leave-speakers-out split -> emodb_folds.json
  features.py               audio -> mel-spectrogram pipeline (shared by both models)
  dataset.py                PyTorch Dataset for Model 1 (from-scratch CNN)
  model_scratch.py          Model 1 architecture (4-block CNN, ~1.2M params)
  train_scratch.py          trains + evaluates Model 1 on one fold
  dataset_transfer.py       PyTorch Dataset for Model 2 (spectrogram -> pseudo-RGB image)
  model_transfer.py         Model 2 architecture (frozen ResNet-18 + new head)
  train_transfer.py         trains + evaluates Model 2 on one fold
  aggregate_results.py      combines the 5 folds into overall metrics
  predict.py                 inference pipeline: audio file -> predicted emotion
app/
  app.py                    Gradio web app (live demo)
```

## Get the data

```bash
mkdir -p data/raw/emodb/wav
curl -L -o /tmp/emodb.zip http://emodb.bilderbar.info/download/download.zip
unzip -q /tmp/emodb.zip -d /tmp/emodb_src
cp /tmp/emodb_src/wav/*.wav data/raw/emodb/wav/
```

Check you got everything:

```bash
ls data/raw/emodb/wav | wc -l   # expect 535
du -sh data/raw/emodb/wav       # expect ~47M
```

Citation: Burkhardt et al., *A Database of German Emotional Speech*,
Interspeech 2005.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have a GPU (CUDA) or an Apple Silicon Mac (MPS), `train_scratch.py`
and `train_transfer.py` detect and use it automatically -- this was tested
on CPU only, so expect training to be faster with one.

## Model 2 setup (one manual step)

Model 2 needs the ImageNet-pretrained ResNet-18 weights:

```bash
wget https://download.pytorch.org/models/resnet18-f37072fd.pth -O models/resnet18-f37072fd.pth
```

The filename's hash prefix (`f37072fd`) should match the start of
`sha256sum models/resnet18-f37072fd.pth` -- a quick way to confirm the file
wasn't corrupted or tampered with.

## Running things, in order

The outputs of steps 1-2 (`emodb_metadata.csv`, `emodb_folds.json`) are
already committed, so you can skip to step 3+ -- but re-running 1-2 is
safe and deterministic if you want to see how they work.

```bash
cd src
python3 load_emodb.py      # 1. parse EmoDB filenames -> metadata CSV
python3 make_splits.py     # 2. build the 5-fold leave-speakers-out split
python3 features.py        # 3. sanity-check the spectrogram pipeline (optional)

python3 model_scratch.py   # 4. verify Model 1's architecture (param count, shapes)
python3 train_scratch.py --fold 0     # 5. train + evaluate Model 1 on fold 0
python3 train_scratch.py --fold 1     #    ... repeat for folds 1-4 for the full
python3 train_scratch.py --fold 2     #    leave-speakers-out result
python3 train_scratch.py --fold 3
python3 train_scratch.py --fold 4

python3 model_transfer.py  # 6. verify Model 2's architecture
python3 train_transfer.py --fold 0    # 7. train + evaluate Model 2 on fold 0
python3 train_transfer.py --fold 1    #    ... same, folds 1-4
python3 train_transfer.py --fold 2
python3 train_transfer.py --fold 3
python3 train_transfer.py --fold 4

python3 aggregate_results.py  # 8. combine the 5 folds into final metrics
```

Each `train_*.py --fold N` prints per-epoch progress and saves a result
file (`data/processed/scratch_foldN_result.pt` or `transfer_foldN_result.pt`)
with the training history, best model weights, and test-set confusion
matrix for that fold.

## Where things stand (final results)

- **Model 1 (from scratch)**: all 5 folds trained. Per-fold test UAR: 0.65,
  0.33, 0.56, 0.65, 0.69 (mean 0.57 ± 0.13). Pooled UAR (one confusion
  matrix over all 535 clips, each scored by a model that never saw that
  speaker) = 0.562.
- **Model 2 (transfer learning)**: all 5 folds trained. Per-fold test UAR:
  0.55, 0.47, 0.62, 0.59, 0.49 (mean 0.55 ± 0.06). Pooled UAR = 0.539.
- Model 1 has the higher mean UAR but much more fold-to-fold variance --
  some held-out speaker pairs are harder than others, expected on a
  10-speaker corpus. Model 2 is more consistent but never reaches Model 1's
  best folds.
- **Known limitation, not a bug**: happiness/anger confusion, in opposite
  directions per model. Model 1 predicts anger instead of happiness
  (happiness recall 0.25, 31/71 happiness clips misclassified as anger).
  Model 2 does the reverse -- predicts happiness instead of anger (46/127
  anger clips misclassified as happiness) -- though its happiness recall is
  much better (0.56). Both are high-arousal emotions that look acoustically
  similar on a spectrogram.
- Model 2 also struggles more with disgust (recall 0.24 vs Model 1's 0.70)
  -- disgust is the rarest class in the corpus (~46 clips).
- Class weighting uses sqrt-inverse-frequency (`class_weight_power=0.5` in
  both training scripts) -- tuned on fold 0 after raw inverse frequency
  over-predicted the rarest class (disgust).
- Live inference app: the application can record audio from the
  microphone or accept an uploaded audio file, display its Mel-spectrogram,
  and predict the emotion with class probabilities. The current app uses
  Model 1 (`scratch_deploy.pt`).
## Inference pipeline

A dedicated inference pipeline was implemented in `src/predict.py` to use
the trained model on new, unseen audio recordings.

For each input audio file, the script:

1. loads and resamples the audio using the same preprocessing used during training;
2. fixes the signal to a 3-second window;
3. computes the 64-band log-Mel spectrogram;
4. applies the training mean and standard deviation stored in the checkpoint;
5. loads the trained scratch CNN in evaluation mode;
6. runs the model and applies softmax to obtain class probabilities;
7. returns the predicted emotion, confidence score, and probabilities for all
   seven emotion classes.

The inference script can also be used independently of the web application:

```bash
python src/predict.py path/to/audio.wav
```

## Live web application

The live application is implemented in `app/app.py` using Gradio.

It provides two ways of supplying audio:

- recording directly from the user's microphone;
- uploading an existing audio file.

After clicking **Analyze emotion**, the application:

1. sends the recording to the inference pipeline;
2. displays the predicted emotion and confidence;
3. displays the probabilities for all seven emotion classes;
4. generates and displays the corresponding Mel-spectrogram.

## Running the application locally

To run the application locally, first activate the virtual environment and make sure the dependencies are installed.

On Windows:
```bash
venv\Scripts\activate
pip install -r requirements.txt
```
On Linux/macOS:
```bash
source venv/bin/activate
pip install -r requirements.txt
```
Then, from the root directory of the repository, run:
```bash
python app/app.py
```
The application will normally be available locally at:
http://127.0.0.1:7860

## Permanent deployment

The application is permanently deployed on Hugging Face Spaces and can be accessed at:

https://bibi724-voice-emotion-recognition.hf.space

The application is deployed using Gradio on Hugging Face Spaces. The Space automatically installs the dependencies from `requirements.txt`, loads the trained CNN model, and launches the Gradio interface.

The public application allows users to record or upload an audio file, visualize its Mel-spectrogram, and obtain the predicted emotion and class probabilities.
