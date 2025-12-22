"""
Data loader for decoder training: text input -> discrete speech units.
Extracts units from answer_audio using mHuBERT + K-means.
"""
import torch
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader

from decoder.unit_extractor import UnitExtractor, TARGET_SAMPLE_RATE

DATASET_NAME = "yuekai/InstructS2S-200K"
NUM_WORKERS = 0


def load_instruct_dataset(filter_num_proc: int | None = None):
    dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    dataset = dataset.cast_column("answer_audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    keep_columns = {"round", "answer_audio", "answer"}
    remove_columns = [col for col in dataset.column_names if col not in keep_columns]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    dataset = dataset.filter(
        lambda example: example["round"] == 1,
        num_proc=filter_num_proc,
    )
    dataset = dataset.remove_columns(["round"])

    return dataset


def make_collate_fn(unit_extractor: UnitExtractor, tokenizer):
    """Create a collate function that tokenizes text and extracts units from audio."""

    def collate(batch):
        answers = [item["answer"] for item in batch]

        text_encoding = tokenizer(
            answers,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )

        unit_ids = []
        for item in batch:
            audio_array = item["answer_audio"]["array"]
            units = unit_extractor.extract_units(audio_array)
            unit_ids.append(units)

        unit_lengths = torch.tensor([len(u) for u in unit_ids], dtype=torch.long)

        return {
            "input_ids": text_encoding.input_ids,
            "attention_mask": text_encoding.attention_mask,
            "unit_ids": unit_ids,
            "unit_lengths": unit_lengths,
        }

    return collate


def split_dataset(dataset, val_ratio: float = 0.01, seed: int = 42):
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def get_dataloaders(
    unit_extractor: UnitExtractor,
    tokenizer,
    batch_size: int = 4,
    val_ratio: float = 0.01,
    seed: int = 42,
):
    dataset = load_instruct_dataset()
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=val_ratio, seed=seed)

    collate_fn = make_collate_fn(unit_extractor=unit_extractor, tokenizer=tokenizer)

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


