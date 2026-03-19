"""
End-to-end integration test — hits the real Modal web app and verifies the full stack.

    modal run tests/test_e2e.py

Tests:
  Real web endpoint (health, voices, validation) →
  Dataset generation via POST /dataset/generate → poll status →
  Verify pipeline output (parquet shards, audio, units) →
  Training via POST /training/start → poll status →
  Verify checkpoints (adapter + decoder) → cleanup
"""

import json
import time
from pathlib import Path

from app.modal_app import app, pipeline_image, data_volume, checkpoints_volume, web

TEST_INPUT_PATH = "/data/input/test_e2e.jsonl"
TEST_ROWS = [
    {"question_text": "What is Python?", "answer": "Python is a programming language."},
    {"question_text": "What is machine learning?", "answer": "Machine learning is a subset of AI."},
]
NUM_SPEAKERS = 9
EXPECTED_PIPELINE_ROWS = len(TEST_ROWS) * NUM_SPEAKERS


@app.function(image=pipeline_image, volumes={"/data": data_volume}, timeout=120)
def setup_test_input():
    """Write test JSONL to the data volume."""
    p = Path(TEST_INPUT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in TEST_ROWS))
    data_volume.commit()


@app.function(
    image=pipeline_image,
    volumes={"/data": data_volume, "/checkpoints": checkpoints_volume},
    timeout=300,
)
def verify_and_cleanup(pipeline_job_id, training_job_id):
    """Verify outputs on both volumes, then clean up."""
    import shutil

    import torch
    from datasets import Audio, Dataset

    errors = []
    pipeline_rows = 0

    # ── Verify pipeline output ───────────────────────────────
    data_volume.reload()
    out = Path(f"/data/output/{pipeline_job_id}")
    shards = sorted(out.glob("shard_*.parquet"))

    if not shards:
        errors.append("No shard files found")
    else:
        all_rows = []
        for shard in shards:
            ds = Dataset.from_parquet(str(shard))
            all_rows.extend(ds.to_list())
        pipeline_rows = len(all_rows)

        if pipeline_rows != EXPECTED_PIPELINE_ROWS:
            errors.append(f"Row count: expected {EXPECTED_PIPELINE_ROWS}, got {pipeline_rows}")

        voice_ids = {r["voice_id"] for r in all_rows}
        if len(voice_ids) != NUM_SPEAKERS:
            errors.append(f"Voice count: expected {NUM_SPEAKERS}, got {len(voice_ids)}")

        # Audio round-trip
        ds = Dataset.from_parquet(str(shards[0]))
        ds = ds.cast_column("question_audio", Audio(sampling_rate=16000))
        audio_item = ds[0]["question_audio"]
        if audio_item["array"] is None or len(audio_item["array"]) == 0:
            errors.append("Audio array is empty")

        # answer_units valid
        for i, row in enumerate(all_rows[:5]):
            units = row["answer_units"]
            if not isinstance(units, list) or len(units) == 0:
                errors.append(f"Row {i}: answer_units empty")
                break
            if not all(0 <= u < 1000 for u in units):
                errors.append(f"Row {i}: units outside [0, 999]")
                break

        # dataset_info.json
        if not (out / "dataset_info.json").exists():
            errors.append("dataset_info.json missing")

    # ── Verify training checkpoints ──────────────────────────
    checkpoints_volume.reload()
    ckpt_dir = Path(f"/checkpoints/runs/{training_job_id}")

    config_path = ckpt_dir / "config.json"
    if not config_path.exists():
        errors.append("config.json not found")
    else:
        cfg = json.loads(config_path.read_text())
        if cfg.get("job_id") != training_job_id:
            errors.append(f"config.json job_id mismatch: {cfg.get('job_id')}")

    for dirname, prefix in [("encoder", "adapter"), ("decoder", "decoder")]:
        phase_dir = ckpt_dir / dirname
        for variant in ["best", "last"]:
            p = phase_dir / f"{variant}_{prefix}.pt"
            if not p.exists():
                errors.append(f"Missing: {p.name}")
            elif p.stat().st_size < 1000:
                errors.append(f"{p.name} too small ({p.stat().st_size} bytes)")

    if not errors:
        adapter_ckpt = torch.load(str(ckpt_dir / "encoder/best_adapter.pt"), map_location="cpu")
        for key in ["adapter_state_dict", "epoch", "optimizer_state_dict", "val_loss"]:
            if key not in adapter_ckpt:
                errors.append(f"Adapter checkpoint missing: {key}")

        decoder_ckpt = torch.load(str(ckpt_dir / "decoder/best_decoder.pt"), map_location="cpu")
        for key in ["decoder_state_dict", "epoch", "optimizer_state_dict", "val_loss"]:
            if key not in decoder_ckpt:
                errors.append(f"Decoder checkpoint missing: {key}")

    # ── Cleanup ──────────────────────────────────────────────
    Path(TEST_INPUT_PATH).unlink(missing_ok=True)

    if pipeline_job_id:
        p = Path(f"/data/output/{pipeline_job_id}")
        if p.exists():
            shutil.rmtree(p)

    if training_job_id:
        p = Path(f"/checkpoints/runs/{training_job_id}")
        if p.exists():
            shutil.rmtree(p)

    data_volume.commit()
    checkpoints_volume.commit()

    return {"passed": len(errors) == 0, "errors": errors, "pipeline_rows": pipeline_rows}


@app.local_entrypoint()
def main():
    import httpx

    print("=== End-to-End Integration Test (API + Pipeline + Training) ===\n")

    # ── Setup ────────────────────────────────────────────────
    print("1. Writing test input to volume...")
    setup_test_input.remote()

    # ── Get the real web URL ─────────────────────────────────
    base_url = web.web_url
    print(f"   Web URL: {base_url}")

    pipeline_job_id = None
    training_job_id = None

    try:
        with httpx.Client(base_url=base_url, timeout=60) as client:

            # ── Health ───────────────────────────────────────
            print("2. Testing GET /health...")
            r = client.get("/health")
            assert r.status_code == 200 and r.json() == {"status": "ok"}, f"Health failed: {r.status_code} {r.text}"
            print("   OK")

            # ── Voices ───────────────────────────────────────
            print("3. Testing GET /voices...")
            r = client.get("/voices")
            assert r.status_code == 200, f"Voices failed: {r.status_code}"
            speakers = r.json()["speakers"]
            assert len(speakers) == NUM_SPEAKERS, f"Expected {NUM_SPEAKERS} speakers, got {len(speakers)}"
            print(f"   OK — {len(speakers)} speakers")

            # ── API validation ───────────────────────────────
            print("4. Testing API validation...")
            r = client.post("/dataset/generate", json={"input_path": "not_jsonl.txt"})
            assert r.status_code == 400, f"Non-JSONL should be 400, got {r.status_code}"

            r = client.post("/training/start", json={"dataset_id": ""})
            assert r.status_code == 400, f"Empty dataset_id should be 400, got {r.status_code}"

            r = client.get("/dataset/status/nonexistent")
            assert r.json()["status"] == "not_found"

            r = client.get("/training/status/nonexistent")
            assert r.json()["status"] == "not_found"
            print("   OK")

            # ── Dataset generation ───────────────────────────
            print("5. POST /dataset/generate...")
            r = client.post("/dataset/generate", json={"input_path": "test_e2e.jsonl"})
            assert r.status_code == 200, f"Generate failed: {r.status_code} {r.text}"
            pipeline_job_id = r.json()["job_id"]
            assert r.json()["status"] == "started"
            print(f"   Started job {pipeline_job_id}")

            # ── Poll pipeline ────────────────────────────────
            print("6. Polling dataset status...")
            for i in range(120):
                r = client.get(f"/dataset/status/{pipeline_job_id}")
                status = r.json()["status"]
                if status == "complete":
                    print(f"   Pipeline complete (~{i * 10}s)")
                    break
                elif status == "failed":
                    raise AssertionError(f"Pipeline failed: {r.json().get('detail')}")
                time.sleep(10)
            else:
                raise AssertionError("Pipeline timed out (20 min)")

            # ── Training ─────────────────────────────────────
            print("7. POST /training/start...")
            r = client.post("/training/start", json={
                "dataset_id": pipeline_job_id,
                "batch_size": 2,
                "encoder_num_epochs": 1,
                "decoder_num_epochs": 1,
            })
            assert r.status_code == 200, f"Training start failed: {r.status_code} {r.text}"
            training_job_id = r.json()["job_id"]
            assert r.json()["status"] == "started"
            print(f"   Started job {training_job_id}")

            # ── Poll training ────────────────────────────────
            print("8. Polling training status...")
            for i in range(120):
                r = client.get(f"/training/status/{training_job_id}")
                status = r.json()["status"]
                if status == "complete":
                    print(f"   Training complete (~{i * 10}s)")
                    break
                elif status == "failed":
                    raise AssertionError(f"Training failed: {r.json().get('detail')}")
                time.sleep(10)
            else:
                raise AssertionError("Training timed out (20 min)")

    except Exception as e:
        print(f"\n   ERROR: {e}")
        # Still run cleanup
        if pipeline_job_id or training_job_id:
            print("9. Cleaning up after failure...")
            verify_and_cleanup.remote(pipeline_job_id or "none", training_job_id or "none")
        raise SystemExit(1)

    # ── Verify + cleanup ─────────────────────────────────────
    print("9. Verifying outputs and cleaning up...")
    result = verify_and_cleanup.remote(pipeline_job_id, training_job_id)

    if result["passed"]:
        print(f"\n   PASSED — {result['pipeline_rows']} pipeline rows, checkpoints valid")
    else:
        print("\n   FAILED:")
        for err in result["errors"]:
            print(f"     - {err}")
        raise SystemExit(1)
