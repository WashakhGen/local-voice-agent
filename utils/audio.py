import queue
import time

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


class Player:
    """Non-blocking playback: put() chunks, they play back-to-back."""

    def __init__(self, sample_rate: int):
        self.q, self.buf = queue.Queue(), np.zeros(0, np.float32)
        self.stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._fill,
            latency="low",
        )
        self.stream.start()

    def _fill(self, out, frames, *_):
        while len(self.buf) < frames and not self.q.empty():
            self.buf = np.concatenate([self.buf, self.q.get_nowait()])
        n = min(frames, len(self.buf))
        out[:n, 0], out[n:, 0] = self.buf[:n], 0
        self.buf = self.buf[n:]

    def put(self, audio: np.ndarray) -> None:
        self.q.put(audio.astype(np.float32))

    def wait(self) -> None:
        while not self.q.empty() or len(self.buf):
            time.sleep(0.02)

    def stop(self) -> None:  # barge-in later: drop everything queued
        self.q = queue.Queue()
        self.buf = np.zeros(0, np.float32)
