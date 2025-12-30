"""
Prepares the final training dataset by joining the HuggingFace dataset with our
processed units parquet. Saves to disk so training can load directly without
reprocessing (avoids filling disk with HF cache files).

Run once before training:
    python dataset_processing/prepare_training_dataset.py
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset, Audio

DATASET_NAME = "yuekai/InstructS2S-200K"
PROCESSED_PARQUET = Path("/home/ubuntu/voice_lab/data/processed/train.parquet")
OUTPUT_DIR = Path("/home/ubuntu/voice_lab/data/processed/training_dataset")
TARGET_SAMPLE_RATE = 16_000


def main():
    print(f"Loading HuggingFace dataset: {DATASET_NAME}", flush=True)
    hf_dataset = load_dataset(DATASET_NAME, split="train")
    hf_dataset = hf_dataset.cast_column("question_audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))
    print(f"Loaded {len(hf_dataset)} rows", flush=True)

    print(f"Loading processed units from: {PROCESSED_PARQUET}", flush=True)
    units_df = pd.read_parquet(PROCESSED_PARQUET)
    valid_ids = set(units_df["id"].tolist())
    id_to_units = dict(zip(units_df["id"], units_df["answer_units"]))
    print(f"Loaded {len(units_df)} processed units", flush=True)

    print("Adding original IDs...", flush=True)
    def add_original_id(example, idx):
        example["_original_id"] = idx
        return example
    hf_dataset = hf_dataset.map(add_original_id, with_indices=True, num_proc=8)

    print("Filtering to valid IDs and round=1...", flush=True)
    hf_dataset = hf_dataset.filter(
        lambda x: x["_original_id"] in valid_ids and x["round"] == 1,
        num_proc=8,
    )
    print(f"Filtered to {len(hf_dataset)} rows", flush=True)

    print("Adding answer units...", flush=True)
    def add_units(example):
        example["answer_units"] = id_to_units[example["_original_id"]]
        return example
    hf_dataset = hf_dataset.map(add_units, num_proc=8)

    keep_columns = {"question_audio", "answer", "answer_units"}
    remove_columns = [col for col in hf_dataset.column_names if col not in keep_columns]
    if remove_columns:
        hf_dataset = hf_dataset.remove_columns(remove_columns)

    print(f"Saving to: {OUTPUT_DIR}", flush=True)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    hf_dataset.save_to_disk(str(OUTPUT_DIR))
    print(f"Done! Saved {len(hf_dataset)} samples", flush=True)
    print(f"Columns: {hf_dataset.column_names}", flush=True)


if __name__ == "__main__":
    main()
