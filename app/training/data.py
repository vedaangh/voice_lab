"""
Data loading for training — reads parquet shards produced by the pipeline.

Replaces research/data_loader.py with parquet-based loading instead of load_from_disk().
"""

import glob
import os

import numpy as np
import torch
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SAMPLES = 480_000  # 30 seconds at 16 kHz
NUM_WORKERS = 4


def _pad_or_trim(audio: np.ndarray, target_length: int = MAX_AUDIO_SAMPLES) -> np.ndarray:
    audio_length = audio.shape[0]
    if audio_length >= target_length:
        return audio[:target_length]
    return np.pad(audio, (0, target_length - audio_length), mode="constant", constant_values=0)


def load_parquet_dataset(data_dir: str):
    """Load training data from parquet shards produced by the pipeline."""
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "shard_*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No shard_*.parquet files in {data_dir}")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    ds = ds.cast_column("question_audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    return ds


def make_collate_fn(
    whisper_processor: WhisperProcessor,
    tokenizer,
    max_answer_tokens: int | None = 128,
    whisper_cache: dict | None = None,
):
    """Create collate function.

    If whisper_cache is provided, uses pre-computed Whisper encoder outputs
    instead of running whisper_processor on raw audio.
    """

    def collate(batch):
        if whisper_cache is not None:
            input_features = torch.stack([whisper_cache[item["__idx__"]] for item in batch])
        else:
            question_audio_list = [
                _pad_or_trim(item["question_audio"]["array"]) for item in batch
            ]
            whisper_inputs = whisper_processor(
                question_audio_list,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = whisper_inputs.input_features

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
            "input_features": input_features,
            "answer_input_ids": answer_input_ids,
            "unit_ids": unit_ids,
            "unit_lengths": torch.tensor([len(u) for u in unit_ids], dtype=torch.long),
        }

    return collate


class IndexedDataset:
    """Wraps an HF Dataset to inject __idx__ into each row for cache lookups."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["__idx__"] = idx
        return item


def get_dataloaders(
    tokenizer,
    data_dir: str,
    batch_size: int = 4,
    val_ratio: float = 0.01,
    seed: int = 42,
    whisper_name: str = "openai/whisper-large-v3",
    max_answer_tokens: int | None = 128,
    whisper_cache: dict | None = None,
):
    dataset = load_parquet_dataset(data_dir)
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    train_dataset, val_dataset = split["train"], split["test"]

    whisper_processor = WhisperProcessor.from_pretrained(whisper_name)
    collate_fn = make_collate_fn(
        whisper_processor=whisper_processor,
        tokenizer=tokenizer,
        max_answer_tokens=max_answer_tokens,
        whisper_cache=whisper_cache,
    )

    if whisper_cache is not None:
        train_dataset = IndexedDataset(train_dataset)
        val_dataset = IndexedDataset(val_dataset)

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
