# src/app/streamlit_app.py
import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import soundfile as sf
import plotly.graph_objects as go
import os
import subprocess
from pathlib import Path
import shutil

# --- Load model, label encoder, and scaler ---
MODEL_PATH = 'models/best_baseline.pkl'
LE_PATH = 'models/label_encoder.pkl'
SCALER_PATH = 'models/scaler.pkl'

clf = joblib.load(MODEL_PATH)
le = joblib.load(LE_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("🎵 Smart Music Analyzer")

# --- Create a cross-platform temporary folder ---
TEMP_DIR = Path(tempfile.gettempdir()) / "temp_audio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Stem splitting function ---
def split_audio(input_file: str, output_dir: str = "stems_output", karaoke=False):
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_file).resolve()

    # Copy to safe path to avoid spaces
    safe_file = TEMP_DIR / input_path.name
    shutil.copy(input_path, safe_file)

    # Demucs command
    cmd = ["demucs", "-n", "htdemucs", "-o", output_dir, str(safe_file)]
    if karaoke:
        cmd = ["demucs", "-n", "htdemucs", "--two-stems=vocals", "-o", output_dir, str(safe_file)]

    subprocess.run(cmd, check=True, shell=True)

    model_dir = os.listdir(output_dir)[0]
    song_name = safe_file.stem
    result_path = os.path.join(output_dir, model_dir, song_name)
    return result_path

# --- Helper: Save audio temporarily in high quality ---
def save_temp_audio(y, sr):
    # Ensure shape is compatible with soundfile
    if y.ndim == 1:
        y_out = y
    elif y.ndim == 2:
        if y.shape[0] == 2:  # shape (2, samples)
            y_out = y.T
        else:  # already (samples, 2)
            y_out = y
    else:
        y_out = np.squeeze(y)

    y_out = y_out.astype(np.float32)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        # Use PCM_16 for Streamlit Cloud compatibility
        sf.write(tmp.name, y_out, sr, subtype='PCM_16')
        return tmp.name

# --- Lyrics generation function using Whisper ---
def generate_lyrics(audio_path):
    try:
        import whisper
    except ModuleNotFoundError:
        st.error("Whisper not installed. Please install `whisper` to enable lyrics generation.")
        return []

    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    text = result['text']

    # Split into stanzas
    lines = text.split('\n')
    stanzas = []
    stanza = []
    for line in lines:
        if line.strip() == "":
            continue
        stanza.append(line.strip())
        if len(stanza) == 4:
            stanzas.append('\n'.join(stanza))
            stanza = []
    if stanza:
        stanzas.append('\n'.join(stanza))
    return stanzas

# --- File uploader ---
uploaded_file = st.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_orig:
        tmp_orig.write(uploaded_file.read())
        tmp_orig.flush()
        tmp_orig_path = tmp_orig.name

    # Load audio
    y, sr = librosa.load(tmp_orig_path, sr=None, mono=False)

    # Original audio playback
    st.subheader("Original Audio")
    st.audio(tmp_orig_path, format='audio/wav')

    # Tempo adjustment
    tempo_factor = st.slider("Adjust Playback Speed (Tempo)", 0.5, 1.5, 1.0, 0.05)
    y_speed = librosa.effects.time_stretch(y, rate=tempo_factor)

    # Pitch adjustment
    pitch_shift = st.slider("Adjust Pitch (Semitones)", -12, 12, 0, 1)
    y_shifted = librosa.effects.pitch_shift(y_speed, sr=sr, n_steps=pitch_shift)

    # Normalize
    y_shifted = y_shifted / np.max(np.abs(y_shifted))

    # Save adjusted audio
    tmp_adjusted_path = save_temp_audio(y_shifted, sr)
    st.subheader(f"Adjusted Audio (Tempo x{tempo_factor}, Pitch {pitch_shift} semitones)")
    st.audio(tmp_adjusted_path, format='audio/wav')

    # Waveform visualization
    st.subheader("Waveform")
    downsample_factor = max(1, y_shifted.shape[-1] // 5000)
    times = np.arange(0, y_shifted.shape[-1], downsample_factor) / sr
    y_plot = y_shifted[..., ::downsample_factor] if y_shifted.ndim == 2 else y_shifted[::downsample_factor]

    fig_wave = go.Figure()
    if y_shifted.ndim == 2:
        fig_wave.add_trace(go.Scatter(x=times, y=y_plot[0], mode='lines', name='Left', line=dict(color='royalblue')))
        fig_wave.add_trace(go.Scatter(x=times, y=y_plot[1], mode='lines', name='Right', line=dict(color='orange')))
    else:
        fig_wave.add_trace(go.Scatter(x=times, y=y_plot, mode='lines', line=dict(color='royalblue')))
    fig_wave.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude", height=300)
    st.plotly_chart(fig_wave, use_container_width=True)

    # Feature extraction for genre prediction
    y_mono = librosa.to_mono(y_shifted) if y_shifted.ndim == 2 else y_shifted
    mfccs = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=13)
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(librosa.feature.chroma_stft(y=y_mono, sr=sr), axis=1),
        np.mean(librosa.feature.spectral_contrast(y=y_mono, sr=sr), axis=1)
    ]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = clf.predict(features_scaled)
    genre = le.inverse_transform(prediction)[0]

    # Tempo & Time Signature
    tempo, beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
    beats_sec = librosa.frames_to_time(beat_frames, sr=sr)
    avg_interval = np.mean(np.diff(beats_sec)) if len(beats_sec) > 1 else 0
    beats_per_bar = round(4 * (60 / tempo) / avg_interval) if avg_interval != 0 else 4
    time_signature = f"{beats_per_bar}/4"

    # Key estimation
    chroma_cq = librosa.feature.chroma_cqt(y=y_mono, sr=sr)
    key_index = np.argmax(np.mean(chroma_cq, axis=1))
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    estimated_key = keys[key_index]

    # Display results
    st.subheader("Analysis Results")
    st.write(f"**Predicted Genre:** {genre}")
    st.write(f"**Estimated Tempo:** {tempo:.2f} BPM")
    st.write(f"**Estimated Time Signature:** {time_signature}")
    st.write(f"**Estimated Key (Scale):** {estimated_key}")

    # Stem splitting
    st.subheader("Stem Splitting")
    split_mode = st.radio("Choose splitting mode", ["Full Stems (Vocals, Drums, Bass, Piano, Other)", "Karaoke (Remove Vocals)"])

    if st.button("Run Stem Separation"):
        with st.spinner("Splitting audio..."):
            try:
                stems_path = split_audio(tmp_orig_path, karaoke=(split_mode == "Karaoke (Remove Vocals)"))
                st.success("Stems created!")

                for stem_file in os.listdir(stems_path):
                    stem_full_path = os.path.join(stems_path, stem_file)
                    st.write(f"**{stem_file}**")
                    st.audio(stem_full_path)
                    with open(stem_full_path, "rb") as f:
                        st.download_button(f"Download {stem_file}", f, file_name=stem_file)

            except Exception as e:
                st.error(f"Stem splitting failed: {e}")

    # Lyrics Generation
    st.subheader("Lyrics Generation")
    if st.button("Generate Lyrics"):
        with st.spinner("Generating lyrics..."):
            stanzas = generate_lyrics(tmp_orig_path)
            if stanzas:
                for i, stanza in enumerate(stanzas, 1):
                    st.markdown(f"**Stanza {i}:**\n{stanza}")
            else:
                st.write("No lyrics generated or Whisper not installed.")

    # Cleanup temporary files
    for f_path in [tmp_orig_path, tmp_adjusted_path]:
        try:
            os.remove(f_path)
        except PermissionError:
            st.warning(f"Temporary file {f_path} is still in use and could not be deleted.")
