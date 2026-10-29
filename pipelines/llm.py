import time

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from core.config import CONFIGS


class LocalLLM:
    name = "local"

    def __init__(
        self,
        model: str = CONFIGS.LLM_MODEL,
        max_tokens: int = 80,
        cache_dir: str = CONFIGS.HF_CACHE_DIR,
    ):
        model_path = snapshot_download(model, cache_dir=cache_dir)
        self.model, self.tokenizer = load(model_path)[:2]
        self.max_tokens = max_tokens
        self.last_stats = {}
        self.reset()

    def reset(self) -> None:
        self.history = [{"role": "system", "content": CONFIGS.SYSTEM_PROMPT}]
        self.cache = make_prompt_cache(self.model)
        self.cached = self.tokenizer.apply_chat_template(self.history)
        self.model(mx.array(self.cached)[None], cache=self.cache)
        mx.eval([c.state for c in self.cache])


    def reply(self, text: str) -> str:
        self.history.append({"role": "user", "content": text})
        tokens = self.tokenizer.apply_chat_template(
            self.history, add_generation_prompt=True
        )
        n = 0
        while (
            n < min(len(tokens) - 1, len(self.cached)) and tokens[n] == self.cached[n]
        ):
            n += 1
        trim_prompt_cache(self.cache, self.cache[0].offset - n)

        t0 = time.perf_counter()
        ttft, parts, out, last = None, [], [], None

        for r in stream_generate(
            self.model,
            self.tokenizer,
            tokens[n:],
            max_tokens=self.max_tokens,
            prompt_cache=self.cache,
        ):
            if ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            parts.append(r.text)

            if r.token not in self.tokenizer.eos_token_ids:
                out.append(r.token)
            last = r

        self.cached = tokens + out
        trim_prompt_cache(self.cache, self.cache[0].offset - len(self.cached))

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
        self.reply("Hi")
        self.reset()
