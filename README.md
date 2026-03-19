# Spoken Digit Recognition

[Open the live app](https://digit-classifier-model-1.streamlit.app)

End-to-end spoken digit recognition project built around a lightweight CNN, reusable Python modules, a deployed Streamlit app, and evaluation utilities for comparing baseline and enhanced checkpoints.

## Live app

- Streamlit app: [https://digit-classifier-model-1.streamlit.app](https://digit-classifier-model-1.streamlit.app)
- Repository: [https://github.com/okonp07/digit-classifier-model](https://github.com/okonp07/digit-classifier-model)

Use the live app here:

https://digit-classifier-model-1.streamlit.app

## What is in this repository

- `digit_recognition/`: reusable package for preprocessing, datasets, model loading, inference, and training.
- `training.py`: top-level training API used in the examples below.
- `evaluation.py`: real-world evaluation and visualization helpers.
- `streamlit_app.py`: interactive web app for uploading audio and comparing checkpoints.
- `models/`: packaged baseline and enhanced model checkpoints.
- `tests/`: smoke tests for the model, preprocessing, and inference API.
- `Digit_Classification_from_Audio.ipynb`: original notebook retained as a reference artifact.

## Use the app

### Live app

1. Open [https://digit-classifier-model-1.streamlit.app](https://digit-classifier-model-1.streamlit.app).
2. Choose `Enhanced model`, `Original model`, or `Compare both` from the sidebar.
3. Upload an audio file containing one spoken digit.
4. Review the predicted digit, confidence scores, waveform, MFCC view, and audio-quality checks.
5. Repeat with new files to compare outputs in the prediction history table.

### Supported audio formats

- `WAV`
- `MP3`
- `M4A`
- `FLAC`
- `OGG`

### Best results

- Speak a single digit from `0` to `9`.
- Keep the clip short, ideally around `1-2 seconds`.
- Record in a quiet environment when possible.
- Use clear pronunciation.

## Run locally

```bash
git clone https://github.com/okonp07/digit-classifier-model.git
cd digit-classifier-model

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Recommended Python version:

- Python `3.11` or `3.12`

If you deploy on Streamlit Community Cloud, choose the Python version from the deployment `Advanced settings`. Streamlit Community Cloud defaults can change over time, and the model stack in this repo is most reliable on Python `3.11` or `3.12`. See the official docs: [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [Manage dependencies](https://docs.streamlit.io/deploy/concepts/dependencies).

The app will look for:

- `models/enhanced_digit_model.pth`
- `models/lightweight_digit_model.pth`

Those checkpoints are already included in the repo.

When Streamlit starts locally, open the URL shown in your terminal, usually:

```text
http://localhost:8501
```

## Python API

### Inference

```python
from digit_recognition import DigitPredictor

predictor = DigitPredictor("models/enhanced_digit_model.pth")
digit, confidence, probabilities = predictor.predict_from_file("sample.wav")

print(digit, confidence)
```

You can also predict from a NumPy array:

```python
import librosa

audio, sr = librosa.load("sample.wav", sr=22050)
digit, confidence, probabilities = predictor.predict_from_array(audio, sr)
```

### Training

```python
from training import prepare_multi_datasets, train_enhanced_model

datasets = prepare_multi_datasets(
    data_dir="data",
    download=True,
    max_samples_per_digit=400,
)

model, metrics = train_enhanced_model(
    datasets=datasets,
    epochs=20,
    use_augmentation=True,
    save_path="models/enhanced_digit_model.pth",
)
```

### Evaluation

Name your recordings with the true digit somewhere in the filename, for example `3_phone.wav` or `digit_7.wav`.

```python
from evaluation import analyze_and_visualize_results, test_real_world_performance
from digit_recognition import DigitPredictor

results = test_real_world_performance(
    recordings_path="real_world_recordings",
    original_model=DigitPredictor("models/lightweight_digit_model.pth"),
    enhanced_model=DigitPredictor("models/enhanced_digit_model.pth"),
)

summary, figure = analyze_and_visualize_results(results)
print(summary)
```

## Streamlit app

The app supports:

- single-file upload for WAV, MP3, M4A, FLAC, and OGG
- original vs enhanced model comparison
- confidence distribution per digit
- waveform and MFCC visualizations
- lightweight audio-quality checks
- session prediction history

Live deployment:

- [https://digit-classifier-model-1.streamlit.app](https://digit-classifier-model-1.streamlit.app)

Run it locally:

```bash
streamlit run streamlit_app.py
```

## Training data

The training pipeline supports:

- Free Spoken Digit Dataset (FSDD)
- Google Speech Commands digit classes

`prepare_multi_datasets(..., download=True)` downloads and prepares both datasets automatically.

Important evaluation detail:

- validation is created with group-aware splitting to reduce speaker leakage
- augmentation is applied to the training split only
- “real-world” evaluation reports actual prediction correctness when filenames contain labels

## Project structure

```text
digit-classifier-model/
├── .github/workflows/ci.yml
├── Digit_Classification_from_Audio.ipynb
├── Dockerfile
├── evaluation.py
├── models/
│   ├── enhanced_digit_model.pth
│   └── lightweight_digit_model.pth
├── digit_recognition/
│   ├── __init__.py
│   ├── audio.py
│   ├── datasets.py
│   ├── model.py
│   ├── predictor.py
│   └── training.py
├── requirements-dev.txt
├── requirements.txt
├── streamlit_app.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_audio.py
│   └── test_models.py
└── training.py
```

## Docker

```bash
docker build -t digit-recognition .
docker run -p 8501:8501 digit-recognition
```

## Development

```bash
python -m pip install -r requirements-dev.txt
pytest
flake8 digit_recognition tests training.py evaluation.py streamlit_app.py
```

## Notes

- The original notebook is still present, but the repository no longer depends on it for inference, evaluation, or app usage.
- The included checkpoints were produced outside this refactor. If you retrain, the package will save new compatible checkpoints.
