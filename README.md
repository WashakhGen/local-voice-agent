# Local Voice Agent

A fully local, low-latency voice agent (speech-to-text, then LLM, then text-to-speech) running on Apple Silicon. each step measures where the latency goes and what actually reduces it.

## Stack

| Stage | Model | Runtime |
|---|---|---|
| STT | [Phonon-2.5-English](https://huggingface.co/FermionResearch/Phonon-2.5-English) (quantized Parakeet TDT 0.6B) | fermion on MLX |
| LLM | [Qwen3-4B-Instruct-2507 4-bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit) | mlx-lm |
| TTS | [Kokoro-82M](https://huggingface.co/prince-canuma/Kokoro-82M) | mlx-audio |

All three models run on the GPU through MLX. Nothing is sent to an API.

## Results

Measured on an M4 Pro with 24 GB of RAM. The input is an 8 s spoken question (`samples/long.wav`) and the reply is two sentences. "First audio" is the time from the end of the user's speech to the first synthesized audio being ready.

| Version | First audio |
|---|---|
| Gemini API as the LLM | ~5100 ms |
| Local, sequential (STT, then the full LLM reply, then the full TTS) | ~1040 ms |
| **Local, streaming** | **~300 ms** |

What got it there:

- **Prompt cache.** The KV cache is kept between turns, so each turn only prefills its new tokens. Time to first token stays around 100 ms instead of growing with the conversation history.
- **Short replies.** A spoken-style system prompt and `max_tokens=80`. Decode is limited by memory bandwidth (~95 tokens/s), so fewer tokens means proportionally less time.
- **Sentence streaming.** The LLM yields each sentence (the first chunk can end at a comma) and TTS synthesizes it while the LLM keeps generating.
- **Kokoro on MLX instead of PyTorch MPS.** Same speed once warm, but no slowdown on unseen input lengths. In a live conversation every sentence is unseen.
- **Non-blocking playback.** A sounddevice output stream plays chunks back-to-back while generation continues.

Things that did not help: `torch.compile` (wrong framework, since the models run on MLX), and `mx.compile` on the STT encoder (saves ~2 ms but recompiles for every new audio length).

## Setup

Requires macOS on Apple Silicon and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run main.py
```

The LLM and TTS weights download to `models/huggingface` on the first run. The STT model path is set by `STT_MODEL` in `core/config.py`. Settings can be overridden in a `.env` file.

`main.py` runs `SAMPLE` through the pipeline several times, prints the timings, and plays the reply.

## Layout

```
core/config.py              settings (model paths, system prompt, voice)
pipelines/stt.py            PhononSTT
pipelines/llm.py            LocalLLM: prompt cache, sentence streaming
pipelines/tts.py            MLXKokoroTTS
pipelines/VoicePipeline.py  wires STT, LLM, TTS and playback; records timings
utils/audio.py              recording, wav loading, streaming Player
utils/metrics.py            timing helpers
```

## Roadmap

1. ~~Baseline pipeline and latency measurement~~
2. ~~Per-stage optimization~~
3. ~~Streaming~~
4. Voice activity detection, end-of-speech detection, barge-in
5. RAG over uploaded PDFs
6. Web frontend with live conversation and transcript
