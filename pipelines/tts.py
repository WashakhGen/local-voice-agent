import numpy as np
from huggingface_hub import snapshot_download
from mlx_audio.tts.utils import load as load_tts

from core.config import CONFIGS

SENTENCE_SPLIT = r"(?<=[.!?])\s+"
WARMUP_TEXTS = (
    "Hello there.",
    "The capital of France is Paris.",
    "This is a longer warmup sentence so the GPU sees a bigger input shape.",
)


class MLXKokoroTTS:
    """Same Kokoro-82M model, running on MLX instead of PyTorch MPS."""

    name = "kokoro_mlx"
    sample_rate = 24_000

    def __init__(
        self,
        voice: str = "af_heart",
        repo_id: str = CONFIGS.TTS_REPO,
        cache_dir: str = CONFIGS.HF_CACHE_DIR,
    ):
        path = snapshot_download(
            repo_id, cache_dir=cache_dir, allow_patterns=["*.json", "*.safetensors"]
        )
        # config.json has no model_type, and the snapshot folder name is a hash, so name it explicitly
        self.model = load_tts(path, model_type="kokoro")
        self.voice = voice
        self.last_stats = {}

    def synthesize(self, text: str) -> np.ndarray:
        chunks = [
            np.asarray(r.audio, dtype=np.float32).reshape(-1)
            for r in self.model.generate(
                text, voice=self.voice, speed=1.2, split_pattern=SENTENCE_SPLIT
            )
        ]
        self.last_stats = {"tts_chunks": len(chunks)}
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def warmup(self) -> None:
        for text in WARMUP_TEXTS:
            self.synthesize(text)
