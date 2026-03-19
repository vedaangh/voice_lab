"""
Modal application — defines GPU workers and serves the FastAPI app.

Deploy: modal deploy app/modal_app.py
Dev:    modal serve app/modal_app.py
"""

import modal

app = modal.App("voicelab")

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

pipeline_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "sox")
    .pip_install(
        "torch", "transformers", "librosa", "soundfile", "numpy",
        "qwen-tts", "joblib", "huggingface_hub",
        "datasets", "pyarrow", "torchcodec",
        "fastapi", "uvicorn", "pydantic-settings", "python-multipart",
    )
    .add_local_dir("app", remote_path="/root/app")
)

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "transformers", "datasets", "pyarrow", "numpy",
        "soundfile", "torchcodec", "wandb", "pydantic", "pydantic-settings",
        "accelerate",
    )
    .add_local_dir("app", remote_path="/root/app")
)

# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

data_volume = modal.Volume.from_name("voicelab-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("voicelab-checkpoints", create_if_missing=True)


@app.cls(
    image=pipeline_image,
    gpu="A10G",
    volumes={"/data": data_volume},
    timeout=3600,
)
class PipelineWorker:
    @modal.enter()
    def load(self):
        from app.tts import TTS
        from app.units import UnitExtractor

        self.tts = TTS()
        self.extractor = UnitExtractor()

    @modal.method()
    def run(self, input_path: str, output_dir: str, assistant_speaker: str | None = None):
        from app.config import settings
        from app.tts import Voice
        from app.pipeline import run_pipeline

        speaker = assistant_speaker or settings.assistant_speaker
        voice = Voice(speaker=speaker, language=settings.assistant_language)

        data_volume.reload()

        run_pipeline(
            input_path=input_path,
            tts=self.tts,
            extractor=self.extractor,
            output_dir=output_dir,
            assistant_voice=voice,
            chunk_size=settings.pipeline_chunk_size,
            tts_sub_batch_size=settings.tts_sub_batch_size,
            unit_sub_batch_size=settings.unit_sub_batch_size,
        )
        data_volume.commit()


@app.cls(
    image=training_image,
    gpu="H100",
    volumes={"/data": data_volume, "/checkpoints": checkpoints_volume},
    timeout=86400,
    # wandb-secret is optional — create it in Modal dashboard to enable logging
)
class TrainingWorker:
    @modal.method()
    def run(self, config_json: str):
        import json
        from app.training.config import TrainingConfig
        from app.training.trainer import run_training

        data_volume.reload()

        config = TrainingConfig(**json.loads(config_json))
        result = run_training(config)

        checkpoints_volume.commit()
        return result


@app.function(image=pipeline_image)
@modal.asgi_app()
def web():
    from app.api.main import create_app

    fastapi_app = create_app()
    fastapi_app.state.pipeline_worker = PipelineWorker()
    fastapi_app.state.training_worker = TrainingWorker()
    return fastapi_app
