import numpy as np
import torch
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from transformers import WhisperProcessor

try:
    from encoder.config import MAX_AUDIO_SAMPLES, TARGET_SAMPLE_RATE
except ModuleNotFoundError:
    from config import MAX_AUDIO_SAMPLES, TARGET_SAMPLE_RATE

DATASET_NAME = "yuekai/InstructS2S-200K"
NUM_WORKERS = 16


def _pad_or_trim(audio: np.ndarray, target_length: int = MAX_AUDIO_SAMPLES) -> np.ndarray:
    audio_length = audio.shape[0]
    if audio_length >= target_length:
        return audio[:target_length]

    padding = target_length - audio_length
    return np.pad(audio, (0, padding), mode="constant", constant_values=0)


def load_instruct_dataset(filter_num_proc: int | None = None):
    dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)
    dataset = dataset.cast_column("question_audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))

    keep_columns = {"round", "question_audio", "answer"}
    remove_columns = [col for col in dataset.column_names if col not in keep_columns]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)

    dataset = dataset.filter(
        lambda example: example["round"] == 1,
        num_proc=filter_num_proc,
    )
    dataset = dataset.remove_columns(["round"])

    return dataset


def make_collate_fn(processor: WhisperProcessor, tokenizer, max_audio_samples: int = MAX_AUDIO_SAMPLES):
    """Create a collate function that pads audio to 30 s and tokenizes answers without padding."""

    def collate(batch):
        answers = [item["answer"] for item in batch]
        audio_features = []
        original_lengths = []

        for item in batch:
            audio_array = item["question_audio"]["array"]
            original_lengths.append(len(audio_array))
            audio_features.append(_pad_or_trim(audio_array, target_length=max_audio_samples))

        processor_outputs = processor(
            audio_features,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        answer_encoding = tokenizer(
            answers,
            add_special_tokens=False,
            padding=False,
            return_attention_mask=False,
        )

        answer_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in answer_encoding["input_ids"]]
        answer_token_lengths = torch.tensor([len(ids) for ids in answer_encoding["input_ids"]], dtype=torch.long)

        return {
            "input_features": processor_outputs.input_features,
            "speech_lengths": torch.tensor(original_lengths, dtype=torch.long),
            "answer_text": answers,
            "answer_input_ids": answer_input_ids,
            "answer_token_lengths": answer_token_lengths,
        }

    return collate


def split_dataset(dataset, val_ratio: float = 0.01, seed: int = 42):
    split = dataset.train_test_split(test_size=val_ratio, seed=seed)
    return split["train"], split["test"]


def get_dataloaders(
    tokenizer,
    batch_size: int = 4,
    val_ratio: float = 0.01,
    seed: int = 42,
):
    dataset = load_instruct_dataset()
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=val_ratio, seed=seed)

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    collate_fn = make_collate_fn(processor=processor, tokenizer=tokenizer)

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
