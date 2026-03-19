"""
Real end-to-end training test — runs on Modal with real GPU and models.

    modal run tests/test_training.py

Tests the full training stack:
  Synthetic data creation → model loading → Whisper feature caching →
  Phase 1 (adapter, 1 epoch) → Phase 2 (decoder, 1 epoch) →
  Checkpoint verification → cleanup
"""

import json
from pathlib import Path

from app.modal_app import app, pipeline_image, training_image, data_volume, checkpoints_volume

TEST_DATASET_ID = "test_training"
TEST_DATA_DIR = f"/data/output/{TEST_DATASET_ID}"
TEST_JOB_ID = "test_train_001"
TEST_CHECKPOINT_DIR = f"/checkpoints/runs/{TEST_JOB_ID}"
NUM_SAMPLES = 10


@app.function(image=pipeline_image, volumes={"/data": data_volume}, timeout=300)
def setup_test_data():
    """Create a minimal synthetic parquet shard matching pipeline output schema."""
    import numpy as np
    from datasets import Audio, Dataset, Features, Sequence, Value

    rng = np.random.default_rng(42)
    out = Path(TEST_DATA_DIR)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(NUM_SAMPLES):
        duration = rng.uniform(1.0, 2.0)
        audio = rng.standard_normal(int(duration * 16000)).astype(np.float32) * 0.1
        rows.append({
            "question_audio": {"array": audio, "sampling_rate": 16000},
            "question_text": f"Test question number {i}",
            "answer": f"This is test answer number {i}.",
            "answer_units": rng.integers(0, 1000, size=rng.integers(10, 50)).tolist(),
            "voice_id": f"voice_{i % 3}",
        })

    features = Features({
        "question_audio": Audio(sampling_rate=16000),
        "question_text": Value("string"),
        "answer": Value("string"),
        "answer_units": Sequence(Value("int32")),
        "voice_id": Value("string"),
    })

    Dataset.from_list(rows, features=features).to_parquet(str(out / "shard_0000.parquet"))
    data_volume.commit()
    return {"num_samples": NUM_SAMPLES}


@app.function(
    image=training_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/checkpoints": checkpoints_volume},
    timeout=1800,
)
def run_training_test():
    """Run a minimal training loop with real models on real GPU."""
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from app.training.config import TrainingConfig
    from app.training.trainer import run_training

    data_volume.reload()

    config = TrainingConfig(
        job_id=TEST_JOB_ID,
        dataset_id=TEST_DATASET_ID,
        train_encoder=True,
        train_decoder=True,
        batch_size=2,
        encoder_num_epochs=1,
        decoder_num_epochs=1,
        val_ratio=0.3,
        cache_whisper_features=True,
    )

    result = run_training(config)
    checkpoints_volume.commit()
    return result


@app.function(
    image=training_image,
    volumes={"/data": data_volume, "/checkpoints": checkpoints_volume},
    timeout=300,
)
def verify_and_cleanup():
    """Check checkpoint files exist and contain expected keys, then clean up."""
    import shutil

    import torch

    checkpoints_volume.reload()
    data_volume.reload()

    errors = []
    ckpt_dir = Path(TEST_CHECKPOINT_DIR)

    # --- config.json ---
    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        errors.append("config.json not found")
    else:
        config = json.loads(config_path.read_text())
        if config.get("job_id") != TEST_JOB_ID:
            errors.append(f"config.json job_id mismatch: {config.get('job_id')}")

    # --- Phase 1 checkpoints ---
    encoder_dir = ckpt_dir / "encoder"
    for name in ["best_adapter.pt", "last_adapter.pt"]:
        p = encoder_dir / name
        if not p.exists():
            errors.append(f"Missing encoder checkpoint: {name}")
        elif p.stat().st_size < 1000:
            errors.append(f"Encoder checkpoint suspiciously small: {name} ({p.stat().st_size} bytes)")

    # --- Phase 2 checkpoints ---
    decoder_dir = ckpt_dir / "decoder"
    for name in ["best_decoder.pt", "last_decoder.pt"]:
        p = decoder_dir / name
        if not p.exists():
            errors.append(f"Missing decoder checkpoint: {name}")
        elif p.stat().st_size < 1000:
            errors.append(f"Decoder checkpoint suspiciously small: {name} ({p.stat().st_size} bytes)")

    # --- Validate checkpoint contents ---
    if not errors:
        adapter_ckpt = torch.load(str(encoder_dir / "best_adapter.pt"), map_location="cpu")
        for key in ["adapter_state_dict", "epoch", "optimizer_state_dict", "val_loss"]:
            if key not in adapter_ckpt:
                errors.append(f"Adapter checkpoint missing key: {key}")

        decoder_ckpt = torch.load(str(decoder_dir / "best_decoder.pt"), map_location="cpu")
        for key in ["decoder_state_dict", "epoch", "optimizer_state_dict", "val_loss"]:
            if key not in decoder_ckpt:
                errors.append(f"Decoder checkpoint missing key: {key}")

    # --- Cleanup ---
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    data_dir = Path(TEST_DATA_DIR)
    if data_dir.exists():
        shutil.rmtree(data_dir)
    checkpoints_volume.commit()
    data_volume.commit()

    return {"passed": len(errors) == 0, "errors": errors}


@app.local_entrypoint()
def main():
    print("=== Integration Test: Training Pipeline on Modal ===\n")

    print("1. Creating synthetic test data...")
    setup_result = setup_test_data.remote()
    print(f"   Created {setup_result['num_samples']} samples.")

    print("2. Running training (real models, 1 epoch per phase on A10G)...")
    train_result = run_training_test.remote()
    print(f"   Training complete: {train_result['status']}")
    print(f"   Adapter checkpoint: {train_result.get('adapter_checkpoint')}")
    print(f"   Decoder checkpoint: {train_result.get('decoder_checkpoint')}")

    print("3. Verifying checkpoints...")
    verify_result = verify_and_cleanup.remote()

    if verify_result["passed"]:
        print("\n   PASSED — all checkpoints valid")
    else:
        print("\n   FAILED:")
        for err in verify_result["errors"]:
            print(f"     - {err}")
        raise SystemExit(1)
