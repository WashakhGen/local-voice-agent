from pydantic_settings import BaseSettings, SettingsConfigDict


class Configs(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    MODELS_DIR: str = "./models"
    RUNS_DIR: str = "./runs"
    WARMUP_WAV: str = "./samples/long.wav"

    MODEL_DIR: str = "./models"
    HF_CACHE_DIR: str = "./models/huggingface"

    STT_MODEL: str = "./models/fermion/speech/FermionResearch__Phonon-2.5-English/model_phonon25_en_pen2u1125_int6"
    LLM_MODEL: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    TTS_REPO: str = "prince-canuma/Kokoro-82M"
    LOG_DIR: str = "./logs"
    STT_SR: int = 16_000

    DEFAULT_VOICE: str = "am_michael"

    SYSTEM_PROMPT: str = (
        "You are a helpful voice assistant. Your replies are spoken aloud. "
        "Answer in at most two short sentences. Write numbers and units as words. "
        "Never use markdown, lists, emoji, or symbols."
    )


CONFIGS = Configs()
