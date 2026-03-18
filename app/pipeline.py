"""
Data generation pipeline.

Takes JSONL text pairs (question_text, answer), synthesizes both sides
with Qwen3-TTS, extracts speech units from answer audio, and saves
Parquet shards with HuggingFace Audio features (FLAC-encoded).

Questions are synthesized with all 9 voices (9x augmentation).
Answers use a fixed assistant voice.
Supports resume: existing shards are skipped on restart.
"""

import json
import logging
from itertools import islice
from pathlib import Path


def batched(iterable, n):
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

import numpy as np
import pyarrow.parquet as pq
from datasets import Audio, Dataset, Features, Sequence, Value

from app.tts import TTS, Voice, TARGET_SAMPLE_RATE
from app.units import UnitExtractor

logger = logging.getLogger(__name__)

DATASET_FEATURES = Features({
    "question_audio": Audio(sampling_rate=TARGET_SAMPLE_RATE),
    "question_text": Value("string"),
    "answer": Value("string"),
    "answer_units": Sequence(Value("int32")),
    "voice_id": Value("string"),
})


def load_input_data(input_path: str) -> list[dict]:
    """Load a JSONL file. Each line must have 'question_text' and 'answer' keys."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = []
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question_text" not in obj or "answer" not in obj:
                raise ValueError(
                    f"Line {i + 1}: missing 'question_text' or 'answer'. Got keys: {list(obj.keys())}"
                )
            rows.append({"question_text": obj["question_text"], "answer": obj["answer"]})

    if not rows:
        raise ValueError(f"Input file is empty: {input_path}")
    return rows


def _synthesize_sub_batched(
    tts: TTS, texts: list[str], voices: list[Voice], sub_batch_size: int,
) -> list[tuple[np.ndarray, int]]:
    """Call tts.synthesize_batch in sub-batches to limit peak VRAM."""
    results: list[tuple[np.ndarray, int]] = []
    for start in range(0, len(texts), sub_batch_size):
        batch = tts.synthesize_batch(
            texts[start : start + sub_batch_size],
            voices[start : start + sub_batch_size],
        )
        results.extend(batch)
    return results


def _extract_units_sub_batched(
    extractor: UnitExtractor, audio_list: list[np.ndarray], sub_batch_size: int,
) -> list:
    """Call extractor.extract_units_batch in sub-batches to limit peak VRAM."""
    results = []
    for start in range(0, len(audio_list), sub_batch_size):
        batch = extractor.extract_units_batch(audio_list[start : start + sub_batch_size])
        results.extend(batch)
    return results


def _find_completed_shards(output_dir: Path) -> set[int]:
    """Return chunk indices whose shards already exist and are valid."""
    done: set[int] = set()
    for p in output_dir.glob("shard_*.parquet"):
        try:
            idx = int(p.stem.split("_")[1])
            if pq.read_metadata(p).num_rows > 0:
                done.add(idx)
        except Exception:
            logger.warning("Ignoring corrupt shard: %s", p)
    return done


def run_pipeline(
    input_path: str,
    tts: TTS,
    extractor: UnitExtractor,
    output_dir: str,
    assistant_voice: Voice,
    chunk_size: int = 64,
    tts_sub_batch_size: int = 32,
    unit_sub_batch_size: int = 32,
) -> None:
    """
    Generate a speech dataset from JSONL text pairs.

    For each (question_text, answer) pair and each of 9 speakers,
    writes one row with FLAC-encoded question audio, answer text,
    and mHuBERT speech units. Output is Parquet shards, one per chunk.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_input_data(input_path)
    speakers = TTS.list_voices()
    logger.info(
        "Loaded %d items, %d speakers => %d output rows",
        len(data), len(speakers), len(data) * len(speakers),
    )

    completed = _find_completed_shards(out)
    if completed:
        logger.info("Resuming: %d shards already complete", len(completed))

    for chunk_idx, chunk in enumerate(batched(data, chunk_size)):
        if chunk_idx in completed:
            continue

        chunk = list(chunk)
        n = len(chunk)

        # Build TTS batch: answers (n) + questions (n * num_speakers)
        texts: list[str] = []
        voices: list[Voice] = []

        for item in chunk:
            texts.append(item["answer"])
            voices.append(assistant_voice)

        for speaker in speakers:
            voice = Voice(speaker=speaker, language=assistant_voice.language)
            for item in chunk:
                texts.append(item["question_text"])
                voices.append(voice)

        all_results = _synthesize_sub_batched(tts, texts, voices, tts_sub_batch_size)

        answer_audios = [audio for audio, _sr in all_results[:n]]
        answer_units = _extract_units_sub_batched(extractor, answer_audios, unit_sub_batch_size)

        rows: list[dict] = []
        for s_idx, speaker in enumerate(speakers):
            offset = n + s_idx * n
            for i, item in enumerate(chunk):
                q_audio, _ = all_results[offset + i]
                rows.append({
                    "question_audio": {"array": q_audio, "sampling_rate": TARGET_SAMPLE_RATE},
                    "question_text": item["question_text"],
                    "answer": item["answer"],
                    "answer_units": answer_units[i].tolist(),
                    "voice_id": speaker,
                })

        shard_path = out / f"shard_{chunk_idx:04d}.parquet"
        Dataset.from_list(rows, features=DATASET_FEATURES).to_parquet(str(shard_path))
        logger.info("Wrote %s (%d rows)", shard_path.name, len(rows))
        del all_results, answer_audios, answer_units, rows

    # Write metadata
    info = {
        "features": DATASET_FEATURES.to_dict(),
        "source_items": len(data),
        "total_rows": len(data) * len(speakers),
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    logger.info("Pipeline complete. Output: %s", out)
