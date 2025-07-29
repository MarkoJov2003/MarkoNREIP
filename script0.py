import soundfile as sf  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, resample, butter, filtfilt
import os
import logging

# === CONFIGURATION ===
wav_folder = '/Users/nreip/Sandbox/new_wavfiles/'
chunk_duration_sec = 300
target_sample_rate = 4000
filter_order = 5
num_channels = 56
segment_duration_sec = 60
output_dir = "./normalizedspectrograms"
os.makedirs(output_dir, exist_ok=True)

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# === FILTER FUNCTION ===
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# === LOOP THROUGH .WAV FILES IN DIRECTORY ===
for file in os.listdir(wav_folder):
    if not file.lower().endswith(".wav"):
        continue

    filepath = os.path.join(wav_folder, file)
    print(f"\n=== Processing: {file} ===")

    try:
        with sf.SoundFile(filepath) as f:
            original_sample_rate = f.samplerate
            frames_to_read = int(chunk_duration_sec * original_sample_rate)
            data = f.read(frames=frames_to_read)
    except Exception as e:
        print(f"Skipping {file} due to read error: {e}")
        continue

    if data.ndim == 1:
        print(f"Skipping {file}: only one channel found.")
        continue

    total_segments = chunk_duration_sec // segment_duration_sec
    actual_channels = data.shape[1]

    if actual_channels < num_channels:
        logging.warning(f"Warning: Expected {num_channels} channels, but file has only {actual_channels}. Using available channels.")

    for ch in range(min(num_channels, actual_channels)):
        for seg in range(total_segments):  # Loop over segments within the channel
            start_idx = int(seg * segment_duration_sec * original_sample_rate)
            end_idx = int((seg + 1) * segment_duration_sec * original_sample_rate)
            channel_data = data[start_idx:end_idx, ch]

            # === LOW-PASS FILTER BEFORE DOWNSAMPLING ===
            cutoff_frequency = target_sample_rate / 2
            filtered_data = butter_lowpass_filter(channel_data, cutoff=cutoff_frequency, fs=original_sample_rate, order=filter_order)

            # === DOWNSAMPLE ===
            number_of_samples = int(len(filtered_data) * target_sample_rate / original_sample_rate)
            data_resampled = resample(filtered_data, number_of_samples)

            # === GENERATE SPECTROGRAM ===
            frequencies, times, Sxx = spectrogram(data_resampled, fs=target_sample_rate)

            # === CONVERT TO dB AND NORMALIZE ===
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)
            Sxx_dB_normalized = (Sxx_dB - np.min(Sxx_dB)) / (np.max(Sxx_dB) - np.min(Sxx_dB))  

            # === PLOT AND EXPORT ===
            output_image = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_ch{ch:02d}_seg{seg:02d}.png")
        
            plt.figure(figsize=(10, 5))
            plt.pcolormesh(times, frequencies, Sxx_dB_normalized, shading = 'gouraud', cmap = 'viridis')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            plt.title(f"Spectrogram (First {chunk_duration_sec} sec, Downsampled to {target_sample_rate} Hz)")
            plt.colorbar(label='Power/Frequency (dB/Hz)')
            plt.tight_layout()
            plt.savefig(output_image, dpi=300)
            plt.close()

            logging.info(f"Spectrogram saved as: {output_image}")