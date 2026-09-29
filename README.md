
The raw audio and the ResNet weights aren't tracked in this repo (see
`.gitignore`) -- the corpus is licensed for research use, not
redistribution, and the weights are a 45MB third-party file. Both are
one command to fetch, below.

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

Run the application locally with:

```bash
python app/app.py
```
## Temporary public link

The web application uses Gradio with `share=True`, which generates a temporary public URL when the app is launched.

To start the application, run:

```bash
python app/app.py
```

This link can be opened from another computer or mobile device and allows remote access to the application. The public URL is temporary and remains active only while the application is running on the host computer. If the Python process is stopped, the terminal is closed, the computer is turned off, or the internet connection is lost, the link will no longer be available.
When the application is started again, Gradio will generate a new temporary public URL.