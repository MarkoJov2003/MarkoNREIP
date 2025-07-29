import soundfile as sf
import numpy as np

# === CONFIGURATION ===
large_wav_path = '/Users/nreip/Sandbox/wavfiles/PASS001_Langnabba_20191111-090500_ADC_Volts_40dB_009_00008_ffffffffffffff.wav'
ref_wav_path = '/Users/nreip/Sandbox/audioset_tagging_cnn/your_audio.wav' 
output_path = '/Users/nreip/Sandbox/audioset_tagging_cnn/updated_audio.wav'

# === LOAD REFERENCE FILE TO GET DESIRED CHANNEL COUNT ===
ref_data, ref_samplerate = sf.read(ref_wav_path)
if len(ref_data.shape) == 1:
    ref_channels = 1
else:
    ref_channels = ref_data.shape[1]

# === LOAD LARGE FILE ===
large_data, large_samplerate = sf.read(large_wav_path)

# Ensure large_data is 2D
if len(large_data.shape) == 1:
    large_data = large_data[:, np.newaxis]
large_channels = large_data.shape[1]

# === Select Channel to Process ===
ch = 9 #Replace 9 for the channel number that you want to use
processed_data = large_data[1000:150999, ch] 

# === SAVE UPDATED FILE ===
sf.write(output_path, processed_data, large_samplerate)
print(f"Saved updated WAV file to: {output_path}")