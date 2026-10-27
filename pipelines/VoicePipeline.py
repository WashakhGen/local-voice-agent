from dataclasses import dataclass

import numpy as np

from core.config import CONFIGS
from pipelines.llm import LocalLLM
from pipelines.stt import PhononSTT
from pipelines.tts import KokoroTTS
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
        tts: KokoroTTS | None = None,
    ):
        self.stt = stt or PhononSTT()
        self.llm = llm or LocalLLM()
        self.tts = tts or KokoroTTS(voice=CONFIGS.DEFAULT_VOICE)

    @classmethod
    def from_pretrained(cls) -> "VoicePipeline":
        return cls(stt=PhononSTT(), llm=LocalLLM(), tts=KokoroTTS())

    @property
    def name(self) -> str:
        return f"{self.stt.name}_{self.llm.name}_{self.tts.name}"

    def warmup(self) -> dict:
        t = {}
        for key, component in (("stt", self.stt), ("llm", self.llm), ("tts", self.tts)):
            with time_into_log(t, key):
                component.warmup()
        return t

    def __call__(self, audio: np.ndarray) -> Turn | None:
        t = {"audio_ms": len(audio) / CONFIGS.STT_SR * 1000}

        with time_into_log(t, "stt_ms"):
            user_text = self.stt.transcribe(audio)
        if not user_text:
            return None
        t.update(self.stt.last_stats)

        with time_into_log(t, "llm_ms"):
            reply = self.llm.reply(user_text)
        t.update(self.llm.last_stats)

        with time_into_log(t, "tts_ms"):
            speech = self.tts.synthesize(reply)
        t.update(self.tts.last_stats)

        t["speech_ms"] = len(speech) / self.tts.sample_rate * 1000
        t["total_ms"] = t["stt_ms"] + t["llm_ms"] + t["tts_ms"]
        return Turn(user_text, reply, speech, t)
