import os
import csv
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt
import logging

# === CONFIGURATION ===
input_folder = '/Users/nreip/Sandbox/pauldata/'
output_folder = './cnn14_ready_audio'
metadata_csv = os.path.join(output_folder, 'metadata.csv')

segment_duration_sec = 10
target_sample_rate = 32000
bandpass = True
bandpass_low, bandpass_high = 50, 14000
filter_order = 5

# Only these labels are allowed in the CSV
ALLOWED = {"boat", "marinelife", "uuv_underwater", "uuv_surface"}

# Map folder names -> final labels (edit/add aliases to fit your directory names)
LABEL_MAP = {
    "boat": "boat",
    "boats": "boat",
    "ship": "boat",
    "ships": "boat",

    "marinelife": "marinelife",
    "marine_life": "marinelife",
    "marine-life": "marinelife",
    "biota": "marinelife",

    "uuv_underwater": "uuv_underwater",
    "uuv-underwater": "uuv_underwater",
    "uuv_submerged": "uuv_underwater",

    "uuv_surface": "uuv_surface",
    "uuv-surface": "uuv_surface",
    "uuv_topside": "uuv_surface",
}

os.makedirs(output_folder, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def bandpass_filter(data, sr, lowcut, highcut, order):
    nyq = 0.5 * sr
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


with open(metadata_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['filepath', 'label'])  # Adjust if using multi-label
    
    for root, _, files in os.walk(input_folder):
        label = os.path.basename(root)  # You can adjust how labels are inferred
        for fname in files:
            if not fname.lower().endswith('.wav'):
                continue

            in_path = os.path.join(root, fname)
            try:
                data, sr = sf.read(in_path)
                if data.ndim > 1:
                    data = data.mean(axis=1)  # Downmix to mono

                if sr != target_sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=target_sample_rate)
                    sr = target_sample_rate

                if bandpass:
                    data = bandpass_filter(data, sr, bandpass_low, bandpass_high, filter_order)

                # Normalize
                data = data.astype(np.float32)
                peak = np.max(np.abs(data)) + 1e-9
                data /= peak

                # Segment and write
                segment_samples = int(segment_duration_sec * sr)
                num_segments = len(data) // segment_samples

                if num_segments == 0:
                    logging.warning(f"Skipped short file: {fname} ({len(data)/sr:.2f} sec)")
                    continue

                for i in range(num_segments):
                    segment = data[i * segment_samples: (i + 1) * segment_samples]
                    out_fname = f"{os.path.splitext(fname)[0]}_seg{i:03d}.wav"
                    out_path = os.path.join(output_folder, out_fname)
                    sf.write(out_path, segment, sr, subtype='PCM_16')
                    writer.writerow([out_path, label])

                logging.info(f"Processed: {fname}, segments: {num_segments}")

            except Exception as e:
                logging.warning(f"Failed to process {fname}: {e}")