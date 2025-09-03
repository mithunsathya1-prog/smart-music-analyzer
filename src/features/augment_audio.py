import librosa
import numpy as np
import soundfile as sf

def augment_speed(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate)

def augment_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise
