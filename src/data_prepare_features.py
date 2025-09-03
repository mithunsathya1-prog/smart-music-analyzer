# src/data_prepare_features.py
import os
import librosa
import numpy as np
import pandas as pd

# Paths
RAW_DIR = 'data/raw'           # Folder containing genre subfolders
FEATURES_DIR = 'data/features'
FEATURES_CSV = os.path.join(FEATURES_DIR, 'features.csv')

# Create features directory if not exists
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)

# Feature extraction function
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

    return np.hstack([mfccs_mean, chroma_mean, contrast])  # total 26 features

# Collect features and labels
data = []
labels = []

for genre in os.listdir(RAW_DIR):
    genre_path = os.path.join(RAW_DIR, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(genre_path, file)
                try:
                    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
                    feats = extract_features(y, sr)
                    data.append(feats)
                    labels.append(genre)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Save to CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv(FEATURES_CSV, index=False)
print(f"Features saved to {FEATURES_CSV}")
