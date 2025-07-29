import torch
##import torchaudio
import numpy as np
import librosa
import soundfile as sf
import os
import sys

# === Add repo root to Python path so we can import from pytorch.models ===
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# === FIX PATH ===
from pytorch.models import Cnn14  # Clean import now that __init__.py exists

# === CONFIGURATION ===
checkpoint_path = '/Users/nreip/Sandbox/audioset_tagging_cnn/pretrained_models/Cnn14_mAP=0.431.pth'
audio_path = '/Users/nreip/Sandbox/audioset_tagging_cnn/updated_audio.wav'
sample_rate = 32000  # Required for PANNs

# === LOAD AUDIO ===
waveform, sr = sf.read(audio_path)
if sr != sample_rate:
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Cnn14(
    sample_rate=sample_rate,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527
)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# === PREPARE INPUT ===
if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)  # Convert to mono
waveform = waveform[None, :]  # Add batch dimension
waveform_tensor = torch.Tensor(waveform).to(device)

# === INFERENCE ===
with torch.no_grad():
    output = model(waveform_tensor)
    prediction = torch.sigmoid(output['clipwise_output']).cpu().numpy()[0]

# === LOAD CLASS LABELS ===
labels_path = '/Users/nreip/Sandbox/audioset_tagging_cnn/metadata/class_labels_indices.csv'
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Class labels file not found: {labels_path}")

with open(labels_path, 'r') as f:
    labels = [line.strip().split(',')[2] for line in f.readlines()[1:]]

# === TOP PREDICTIONS ===
top_indices = prediction.argsort()[-10:][::-1]
print("\nTop 10 Predictions:")
for idx in top_indices:
    print(f"{labels[idx]}: {prediction[idx]:.3f}")