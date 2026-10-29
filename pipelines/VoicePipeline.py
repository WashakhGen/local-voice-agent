import time
from dataclasses import dataclass

import numpy as np

from core.config import CONFIGS
from pipelines.llm import LocalLLM
from pipelines.stt import PhononSTT
from pipelines.tts import MLXKokoroTTS
from utils.audio import Player, load_wav
from utils.metrics import time_into_log


@dataclass
class Turn:
    user_text: str
    reply_text: str
    speech: np.ndarray
    timings: dict


class VoicePipeline:
    def __init__(
        self,
        stt: PhononSTT | None = None,
        llm: LocalLLM | None = None,
        tts: MLXKokoroTTS | None = None,
    ):
        self.stt = stt or PhononSTT()
        self.llm = llm or LocalLLM()
        self.tts = tts or MLXKokoroTTS(voice=CONFIGS.DEFAULT_VOICE)

    @property
    def name(self) -> str:
        return f"{self.stt.name}_{self.llm.name}_{self.tts.name}"

    def warmup(self) -> dict:
        t = {}
        for key, component in (("stt", self.stt), ("llm", self.llm), ("tts", self.tts)):
            with time_into_log(t, key):
                component.warmup()
        # one silent end-to-end turn: STT sees a real input length, and all three models run back to back
        with time_into_log(t, "full_turn"):
            self(load_wav(CONFIGS.WARMUP_WAV))
        self.llm.reset()
        return t

        return t

    def __call__(self, audio: np.ndarray, player: Player | None = None) -> Turn | None:
        t = {"audio_ms": len(audio) / CONFIGS.STT_SR * 1000}
        t0 = time.perf_counter()

        with time_into_log(t, "stt_ms"):
            user_text = self.stt.transcribe(audio)
        if not user_text:
            return None
        t.update(self.stt.last_stats)

        chunks, tts_ms = [], 0.0
        for sentence in self.llm.stream(user_text):
            t1 = time.perf_counter()
            chunk = self.tts.synthesize(sentence)
            tts_ms += (time.perf_counter() - t1) * 1000
            if not chunks:
                t["first_audio_ms"] = (time.perf_counter() - t0) * 1000
            if player:
                player.put(chunk)
            chunks.append(chunk)
        t.update(self.llm.last_stats)
        t["tts_ms"] = tts_ms
        t["total_ms"] = (time.perf_counter() - t0) * 1000

        speech = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        t["speech_ms"] = len(speech) / self.tts.sample_rate * 1000
        reply = self.llm.history[-1]["content"]
        return Turn(user_text, reply, speech, t)
