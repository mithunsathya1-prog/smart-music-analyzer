# src/features/extract_features.py
import os
import librosa
import numpy as np
import pandas as pd

# Directories
RAW_DIR = 'data/raw'
FEATURES_DIR = 'data/features'
FEATURES_CSV = os.path.join(FEATURES_DIR, 'features.csv')

# Create features directory if not exists
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)

# Function to extract audio features
def extract_features(y, sr):
    try:
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # RMS
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        # Zero-crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
        # Spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        return np.hstack([mfccs_mean, chroma_mean, tempo, rms, zcr, contrast])
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

data = []
labels = []

# Loop over all genres and audio files
for genre in os.listdir(RAW_DIR):
    genre_path = os.path.join(RAW_DIR, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(genre_path, file)
                try:
                    y, sr = librosa.load(file_path, sr=22050, mono=True)

                    # ---- Original ----
                    feats = extract_features(y, sr)
                    if feats is not None:
                        data.append(feats)
                        labels.append(genre)

                    # ---- Time-stretch augmentations ----
                    for rate in [0.9, 1.1]:
                        try:
                            y_stretch = librosa.effects.time_stretch(y, rate=rate)
                            feats = extract_features(y_stretch, sr)
                            if feats is not None:
                                data.append(feats)
                                labels.append(genre)
                        except Exception as e:
                            print(f"Time-stretch failed for {file_path}: {e}")

                    # ---- Pitch-shift augmentations ----
                    for steps in [-2, 2]:
                        try:
                            y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
                            feats = extract_features(y_pitch, sr)
                            if feats is not None:
                                data.append(feats)
                                labels.append(genre)
                        except Exception as e:
                            print(f"Pitch-shift failed for {file_path}: {e}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Save to CSV
features = pd.DataFrame(data)
features['label'] = labels
features.to_csv(FEATURES_CSV, index=False)
print(f"Features saved to {FEATURES_CSV}")
