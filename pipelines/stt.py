import fermion
import numpy as np

from core.config import CONFIGS
from utils.audio import load_wav


class PhononSTT:
    name = "phonon"

    def __init__(self, model: str = CONFIGS.STT_MODEL):
        self.speech = fermion.load_speech(model)
        self.last_stats = {}

    def transcribe(self, audio: np.ndarray) -> str:
        text, decode_s, _ = self.speech.transcribe_array(audio.astype(np.float32))
        self.last_stats = {"stt_engine_ms": decode_s * 1000}
        return text.strip()

    def warmup(self) -> None:
        # Real speech: silence lets the engine skip decoding, so nothing gets warmed.
        self.transcribe(load_wav(CONFIGS.WARMUP_WAV))
