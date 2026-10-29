import warnings

# Kokoro 0.9.4 (latest) calls PyTorch APIs deprecated in torch 2.x; harmless at inference.
warnings.filterwarnings("ignore", message="dropout option adds dropout")
warnings.filterwarnings("ignore", message=".*weight_norm.*is deprecated")
warnings.filterwarnings("ignore", message=".*torch.jit.script.*is deprecated")


import numpy as np
from huggingface_hub import hf_hub_download
from kokoro import KModel, KPipeline

from core.config import CONFIGS


class KokoroTTS:
    name = "kokoro"
    sample_rate = 24_000

    def __init__(
        self,
        voice: str = "af_heart",
        repo_id: str = CONFIGS.TTS_REPO,
        cache_dir: str = CONFIGS.HF_CACHE_DIR,
        device: str = "mps",
    ):
        self.voice, model = self.download_model(repo_id, cache_dir, device, voice)
        self.pipeline = KPipeline(lang_code="a", repo_id=repo_id, model=model)
        self.last_stats = {}

    def download_model(self, repo_id, cache_dir, device, voice):
        config = hf_hub_download(repo_id, "config.json", cache_dir=cache_dir)
        weights = hf_hub_download(repo_id, "kokoro-v1_0.pth", cache_dir=cache_dir)
        voice = hf_hub_download(repo_id, f"voices/{voice}.pt", cache_dir=cache_dir)
        model = KModel(repo_id=repo_id, config=config, model=weights).to(device).eval()
        return voice, model

    def synthesize(self, text: str) -> np.ndarray:
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _, _, audio in self.pipeline(text, voice=self.voice)
            if audio is not None
        ]
        self.last_stats = {"tts_chunks": len(chunks)}
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def warmup(self) -> None:
        # MPS compiles kernels per input shape; warm a few lengths.
        for text in (
            "Hello there.",
            "The capital of France is Paris.",
            "This is a longer warmup sentence so the GPU sees a bigger input shape.",
        ):
            self.synthesize(text)
