"""
Unified data loader for encoder and decoder training.
Provides: question_audio, answer text, and answer_units (pre-extracted discrete units).

The original yuekai/InstructS2S-200K dataset's speech_token field uses a different unit
vocabulary (not K=1000 HuBERT centroids). We re-synthesized answer audio via Azure TTS
and extracted K=1000 units using our UnitExtractor. The processed dataset is saved to
disk by prepare_training_dataset.py to avoid repeated processing.
"""

from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

PROCESSED_DATASET_DIR = Path("/home/ubuntu/voice_lab/data/processed/training_dataset")
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SAMPLES = 480_000  # 30 seconds at 16 kHz
NUM_WORKERS = 4


def _pad_or_trim(audio: np.ndarray, target_length: int = MAX_AUDIO_SAMPLES) -> np.ndarray:
    audio_length = audio.shape[0]
    if audio_length >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - audio_length), mode="constant", constant_values=0)


def load_instruct_dataset():
    """
    Load pre-processed dataset from disk.
    Run prepare_training_dataset.py first if the dataset doesn't exist.
    """
    if not PROCESSED_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_DATASET_DIR}. "
            "Run: python dataset_processing/prepare_training_dataset.py"
        )
    return load_from_disk(str(PROCESSED_DATASET_DIR))


def make_collate_fn(
    whisper_processor: WhisperProcessor,
    tokenizer,
    max_answer_tokens: int | None = 128,
):
    """
    Create collate function.
    If max_answer_tokens is set, truncates answers to prevent OOM on long responses.
    """

    def collate(batch):
        question_audio_list = [_pad_or_trim(item["question_audio"]["array"]) for item in batch]

        whisper_inputs = whisper_processor(
            question_audio_list,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        answers = [item["answer"] for item in batch]
        tokenizer_kwargs = {
            "add_special_tokens": False,
            "padding": False,
            "return_attention_mask": False,
        }
        if max_answer_tokens is not None:
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = max_answer_tokens

        answer_encoding = tokenizer(answers, **tokenizer_kwargs)
        answer_input_ids = [
            torch.tensor(ids, dtype=torch.long) for ids in answer_encoding["input_ids"]
        ]

        unit_ids = [torch.tensor(item["answer_units"], dtype=torch.long) for item in batch]

        return {
            "input_features": whisper_inputs.input_features,
            "answer_input_ids": answer_input_ids,
            "unit_ids": unit_ids,
            "unit_lengths": torch.tensor([len(u) for u in unit_ids], dtype=torch.long),
        }

    return collate


def get_dataloaders(
    tokenizer,
    batch_size: int = 4,
    val_ratio: float = 0.01,
    seed: int = 42,
    whisper_name: str = "openai/whisper-large-v3",
    max_answer_tokens: int | None = 128,
):
    dataset = load_instruct_dataset()
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    train_dataset, val_dataset = split["train"], split["test"]

    whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
    collate_fn = make_collate_fn(
        whisper_processor=whisper_processor,
        tokenizer=tokenizer,
        max_answer_tokens=max_answer_tokens,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader
