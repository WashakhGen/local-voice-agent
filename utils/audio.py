import numpy as np
import sounddevice as sd
import soundfile as sf

from core.config import CONFIGS


def record() -> np.ndarray:
    input("\nPress Enter to talk...")
    frames = []
    with sd.InputStream(
        samplerate=CONFIGS.STT_SR,
        channels=1,
        dtype="float32",
        callback=lambda data, *_: frames.append(data.copy()),
    ):
        input("Recording... press Enter to stop.")
    return np.concatenate(frames)[:, 0] if frames else np.zeros(0, np.float32)


def load_wav(path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != CONFIGS.STT_SR:
        # Crude linear resample; fine for testing, use a proper resampler later.
        n = int(len(audio) * CONFIGS.STT_SR / sr)
        audio = np.interp(
            np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio
        )
    return audio.astype(np.float32)


def play(audio: np.ndarray, sample_rate: int) -> None:
    if len(audio):
        sd.play(audio, sample_rate)
        sd.wait()
