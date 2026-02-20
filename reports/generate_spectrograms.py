"""Generate spectrograms and transcriptions for the progress report."""

import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display

# Paths
AUDIO_DIR = "/home/vedaangh/voice_lab"
OUTPUT_DIR = "/home/vedaangh/voice_lab/reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Audio files to process
AUDIO_FILES = [
    ("test_input.wav", "Input Speech (Question)"),
    ("test_output.wav", "Output Speech (Model Response)"),
    ("test_input_val.wav", "Validation Input"),
    ("test_output_val.wav", "Validation Output"),
]

def generate_spectrogram(audio_path, title, output_name):
    """Generate and save a mel spectrogram."""
    audio, sr = sf.read(audio_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', 
                                    sr=sr, fmax=8000, ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_name}.png"), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f"{output_name}.pdf"), bbox_inches='tight')
    plt.close()
    
    duration = len(audio) / sr
    print(f"Generated: {output_name} ({duration:.2f}s)")
    return duration

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper."""
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"].strip()
    except ImportError:
        # Fallback: use transformers
        from transformers import pipeline
        transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")
        result = transcriber(audio_path)
        return result["text"].strip()

# Generate spectrograms
print("Generating spectrograms...")
for filename, title in AUDIO_FILES:
    audio_path = os.path.join(AUDIO_DIR, filename)
    output_name = filename.replace(".wav", "_spectrogram")
    generate_spectrogram(audio_path, title, output_name)

# Generate combined figure (input vs output)
print("\nGenerating combined comparison figure...")
fig, axes = plt.subplots(2, 1, figsize=(12, 5))

for idx, (filename, title) in enumerate([AUDIO_FILES[0], AUDIO_FILES[1]]):
    audio_path = os.path.join(AUDIO_DIR, filename)
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                    sr=sr, fmax=8000, ax=axes[idx], cmap='viridis')
    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Time (s)')
    axes[idx].set_ylabel('Freq (Hz)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "input_output_comparison.png"), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "input_output_comparison.pdf"), bbox_inches='tight')
plt.close()
print("Generated: input_output_comparison")

# Transcribe audio files
print("\nTranscribing audio files...")
transcriptions = {}
for filename, title in AUDIO_FILES[:2]:  # Just input and output
    audio_path = os.path.join(AUDIO_DIR, filename)
    try:
        text = transcribe_audio(audio_path)
        transcriptions[filename] = text
        print(f"{title}: {text}")
    except Exception as e:
        print(f"Could not transcribe {filename}: {e}")
        transcriptions[filename] = "[Transcription unavailable]"

# Save transcriptions
with open(os.path.join(OUTPUT_DIR, "transcriptions.txt"), "w") as f:
    for filename, text in transcriptions.items():
        f.write(f"{filename}:\n{text}\n\n")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
