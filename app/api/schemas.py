"""Pydantic request/response models."""

from pydantic import BaseModel


class DatasetGenerateRequest(BaseModel):
    input_path: str
    assistant_speaker: str | None = None


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None


class VoicesResponse(BaseModel):
    speakers: list[str]
    assistant_speaker: str


class ChooseVoiceRequest(BaseModel):
    speaker: str


class TrainingStartRequest(BaseModel):
    dataset_id: str

    # Phases
    train_encoder: bool = True
    train_decoder: bool = True

    # Training hyperparameters
    batch_size: int = 4
    encoder_num_epochs: int = 10
    encoder_learning_rate: float = 1e-4
    decoder_num_epochs: int = 10
    decoder_learning_rate: float = 1e-4

    # Checkpoints (for resume)
    adapter_checkpoint: str | None = None
    adapter_resume: bool = False
    decoder_checkpoint: str | None = None
    decoder_resume: bool = False

    # Misc
    cache_whisper_features: bool = True
    wandb_project: str = "voice_lab"


