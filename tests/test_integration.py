"""
Real end-to-end integration test — runs on Modal with real GPU, models, and volume.

    modal run tests/test_integration.py

Tests the full stack:
  Modal container boot → model loading (@modal.enter) → Qwen TTS synthesis →
  mHuBERT unit extraction → parquet serialization → volume storage → readback
"""

import json
from pathlib import Path

from app.modal_app import app, pipeline_image, data_volume, PipelineWorker

TEST_INPUT_PATH = "/data/input/test_integration.jsonl"
TEST_OUTPUT_DIR = "/data/output/test_integration"

TEST_ROWS = [
    {"question_text": "What is Python?", "answer": "Python is a programming language."},
    {"question_text": "What is machine learning?", "answer": "Machine learning is a subset of AI."},
]

NUM_SPEAKERS = 9
EXPECTED_TOTAL = len(TEST_ROWS) * NUM_SPEAKERS


@app.function(image=pipeline_image, volumes={"/data": data_volume}, timeout=120)
def setup_test_input():
    """Write test JSONL to the Modal volume."""
    p = Path(TEST_INPUT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in TEST_ROWS))
    data_volume.commit()


@app.function(image=pipeline_image, volumes={"/data": data_volume}, timeout=120)
def verify_and_cleanup():
    """Read pipeline output, run assertions, clean up, return results."""
    import shutil

    import numpy as np
    from datasets import Audio, Dataset

    data_volume.reload()
    out = Path(TEST_OUTPUT_DIR)
    errors = []

    # --- shards exist ---
    shards = sorted(out.glob("shard_*.parquet"))
    if not shards:
        errors.append("No shard files found")
        _cleanup()
        return {"passed": False, "errors": errors}

    # --- load all rows ---
    all_rows = []
    for shard in shards:
        ds = Dataset.from_parquet(str(shard))
        all_rows.extend(ds.to_list())

    # --- total rows ---
    if len(all_rows) != EXPECTED_TOTAL:
        errors.append(f"Row count: expected {EXPECTED_TOTAL}, got {len(all_rows)}")

    # --- all 9 voices ---
    voice_ids = {r["voice_id"] for r in all_rows}
    if len(voice_ids) != NUM_SPEAKERS:
        errors.append(f"Voice count: expected {NUM_SPEAKERS}, got {len(voice_ids)}: {sorted(voice_ids)}")

    # --- question texts match input ---
    question_texts = {r["question_text"] for r in all_rows}
    expected_texts = {r["question_text"] for r in TEST_ROWS}
    if question_texts != expected_texts:
        errors.append(f"Question texts: expected {expected_texts}, got {question_texts}")

    # --- answer_units valid ---
    for i, row in enumerate(all_rows):
        units = row["answer_units"]
        if not isinstance(units, list) or len(units) == 0:
            errors.append(f"Row {i}: answer_units empty or not a list")
            break
        if not all(0 <= u < 1000 for u in units):
            errors.append(f"Row {i}: answer_units has values outside [0, 999]")
            break

    # --- audio round-trip ---
    ds = Dataset.from_parquet(str(shards[0]))
    ds = ds.cast_column("question_audio", Audio(sampling_rate=16_000))
    audio_item = ds[0]["question_audio"]
    arr = audio_item["array"]
    if arr is None or len(arr) == 0:
        errors.append("Audio array is empty")
    if audio_item["sampling_rate"] != 16_000:
        errors.append(f"Sample rate: expected 16000, got {audio_item['sampling_rate']}")

    # --- dataset_info.json ---
    info_path = out / "dataset_info.json"
    if not info_path.exists():
        errors.append("dataset_info.json missing")
    else:
        info = json.loads(info_path.read_text())
        if info.get("source_items") != len(TEST_ROWS):
            errors.append(f"source_items: expected {len(TEST_ROWS)}, got {info.get('source_items')}")
        if info.get("total_rows") != EXPECTED_TOTAL:
            errors.append(f"total_rows: expected {EXPECTED_TOTAL}, got {info.get('total_rows')}")

    # --- cleanup ---
    _cleanup()

    return {
        "passed": len(errors) == 0,
        "total_rows": len(all_rows),
        "voices": sorted(voice_ids),
        "errors": errors,
    }


def _cleanup():
    import shutil

    Path(TEST_INPUT_PATH).unlink(missing_ok=True)
    out = Path(TEST_OUTPUT_DIR)
    if out.exists():
        shutil.rmtree(out)
    data_volume.commit()


@app.local_entrypoint()
def main():
    print("=== Integration Test: Full Pipeline on Modal ===\n")

    print("1. Writing test input to volume...")
    setup_test_input.remote()

    print("2. Running PipelineWorker (real TTS + real unit extraction on GPU)...")
    worker = PipelineWorker()
    worker.run.remote(
        input_path=TEST_INPUT_PATH,
        output_dir=TEST_OUTPUT_DIR,
    )
    print("   Pipeline complete.")

    print("3. Verifying output...")
    result = verify_and_cleanup.remote()

    if result["passed"]:
        print(f"\n   PASSED — {result['total_rows']} rows, voices: {result['voices']}")
    else:
        print(f"\n   FAILED:")
        for err in result["errors"]:
            print(f"     - {err}")
        raise SystemExit(1)
