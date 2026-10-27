import time

from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate

from core.config import CONFIGS


class LocalLLM:
    name = "local"

    def __init__(
        self,
        model: str = CONFIGS.LLM_MODEL,
        max_tokens: int = 200,
        cache_dir: str = CONFIGS.HF_CACHE_DIR,
    ):
        model_path = snapshot_download(model, cache_dir=cache_dir)
        self.model, self.tokenizer = load(model_path)[:2]
        self.max_tokens = max_tokens
        self.last_stats = {}
        self.reset()

    def reset(self) -> None:
        self.history = [{"role": "system", "content": CONFIGS.SYSTEM_PROMPT}]

    def reply(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        prompt = self.tokenizer.apply_chat_template(
            self.history, add_generation_prompt=True
        )
        t0 = time.perf_counter()
        ttft, parts, last = None, [], None
        for r in stream_generate(
            self.model, self.tokenizer, prompt, max_tokens=self.max_tokens
        ):
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            parts.append(r.text)
            last = r
        reply = "".join(parts).strip()
        self.history.append({"role": "assistant", "content": reply})
        self.last_stats = {
            "llm_ttft_ms": ttft or 0.0,
            "llm_prompt_tokens": last.prompt_tokens if last else 0,
            "llm_output_tokens": last.generation_tokens if last else 0,
            "llm_prefill_tps": last.prompt_tps if last else 0.0,
            "llm_decode_tps": last.generation_tps if last else 0.0,
        }
        return reply

    def warmup(self) -> None:
        # First generate compiles MLX kernels; do it now, then forget the exchange.
        self.reply("Hi")
        self.reset()
