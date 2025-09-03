# app/streamlit_app.py
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tempfile
import soundfile as sf
import plotly.graph_objects as go

# --- Load model, label encoder, and scaler ---
MODEL_PATH = 'models/best_baseline.pkl'
LE_PATH = 'models/label_encoder.pkl'
SCALER_PATH = 'models/scaler.pkl'

clf = joblib.load(MODEL_PATH)
le = joblib.load(LE_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("🎵 Smart Music Analyzer")

uploaded_file = st.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3"])

if uploaded_file:
    # Save uploaded file temporarily
    tmp_orig = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_orig.write(uploaded_file.read())
    tmp_orig.flush()

    # Load full audio (no 30s limit)
    y, sr = librosa.load(tmp_orig.name, sr=22050, mono=True)

    # --- Original audio playback ---
    st.subheader("Original Audio")
    st.audio(tmp_orig.name, format='audio/wav')

    # --- Tempo adjustment ---
    tempo_factor = st.slider("Adjust Playback Speed (Tempo)", 0.5, 1.5, 1.0, 0.05)
    y_speed = librosa.effects.time_stretch(y, rate=tempo_factor)

    # --- Pitch (Scale) adjustment ---
    pitch_shift = st.slider("Adjust Pitch (Semitones)", -12, 12, 0, 1)
    y_shifted = librosa.effects.pitch_shift(y_speed, sr=sr, n_steps=pitch_shift)

    # Save adjusted audio temporarily
    tmp_adjusted = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp_adjusted.name, y_shifted, sr)
    st.subheader(f"Adjusted Audio (Tempo x{tempo_factor}, Pitch {pitch_shift} semitones)")
    st.audio(tmp_adjusted.name, format='audio/wav')

    # --- Waveform visualization ---
    st.subheader("Waveform")
    downsample_factor = max(1, len(y_shifted)//5000)
    times = np.arange(0, len(y_shifted), downsample_factor) / sr
    y_plot = y_shifted[::downsample_factor]

    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=times, y=y_plot, mode='lines', line=dict(color='royalblue')))
    fig_wave.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude", height=300)
    st.plotly_chart(fig_wave, use_container_width=True)

    # --- Feature extraction for genre prediction ---
    mfccs = librosa.feature.mfcc(y=y_shifted, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    chroma = librosa.feature.chroma_stft(y=y_shifted, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    contrast = librosa.feature.spectral_contrast(y=y_shifted, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    features = np.hstack([mfccs_mean, chroma_mean, contrast_mean]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = clf.predict(features_scaled)
    genre = le.inverse_transform(prediction)[0]

    # --- Tempo & Time Signature ---
    tempo, beat_frames = librosa.beat.beat_track(y=y_shifted, sr=sr)
    tempo_value = float(tempo)
    beats_sec = librosa.frames_to_time(beat_frames, sr=sr)
    avg_interval = np.mean(np.diff(beats_sec)) if len(beats_sec) > 1 else 0
    beats_per_bar = round(4 * (60 / tempo_value) / avg_interval) if avg_interval != 0 else 4
    time_signature = f"{beats_per_bar}/4"

    # --- Key / Scale estimation ---
    chroma_cq = librosa.feature.chroma_cqt(y=y_shifted, sr=sr)
    chroma_mean_key = np.mean(chroma_cq, axis=1)
    key_index = np.argmax(chroma_mean_key)
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = keys[key_index]

    # --- Display results ---
    st.subheader("Analysis Results")
    st.write(f"**Predicted Genre:** {genre}")
    st.write(f"**Estimated Tempo:** {tempo_value:.2f} BPM")
    st.write(f"**Estimated Time Signature:** {time_signature}")
    st.write(f"**Estimated Key (Scale):** {estimated_key}")
