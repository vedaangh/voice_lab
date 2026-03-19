"""Pydantic training configuration (replaces Hydra DictConfig from research)."""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel

MODEL_DTYPE = torch.bfloat16


class TrainingConfig(BaseModel):
    job_id: str = ""

    # Phases
    train_encoder: bool = True
    train_decoder: bool = True

    # Architecture
    whisper_name: str = "openai/whisper-large-v3"
    llm_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    adapter_hidden_dim: int = 2048
    adapter_ds_rate: int = 5
    decoder_hidden_dim: int = 4096
    decoder_num_heads: int = 32
    decoder_num_layers: int = 2
    decoder_intermediate_dim: int = 11008
    decoder_upsample_rate: int = 25

    # Training — shared
    batch_size: int = 4
    warmup_ratio: float = 0.05
    gradient_clip: float = 1.0
    val_ratio: float = 0.01
    seed: int = 42
    max_answer_tokens: int = 128

    # Training — per phase
    encoder_num_epochs: int = 10
    encoder_learning_rate: float = 1e-4
    decoder_num_epochs: int = 10
    decoder_learning_rate: float = 1e-4

    # Data
    dataset_id: str = ""

    # Checkpoints (for resume)
    adapter_checkpoint: Optional[str] = None
    adapter_resume: bool = False
    decoder_checkpoint: Optional[str] = None
    decoder_resume: bool = False

    # Misc
    prompt_template: str = "original.yaml"
    wandb_project: str = "voice_lab"
    cache_whisper_features: bool = True

    model_config = {"arbitrary_types_allowed": True}
