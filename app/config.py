"""App configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # TTS
    assistant_speaker: str = "Ryan"
    assistant_language: str = "English"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    # Pipeline
    pipeline_chunk_size: int = 64
    tts_sub_batch_size: int = 32
    unit_sub_batch_size: int = 32

    # Modal volumes
    data_volume_path: str = "/data"
    checkpoints_volume_path: str = "/checkpoints"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_prefix": "VOICELAB_"}


settings = Settings()
