"""
Azure TTS + Unit Extraction Pipeline.
Processes yuekai/InstructS2S-200K: synthesizes answer audio via Azure TTS,
extracts K=1000 units, saves to parquet.

Output: parquet with (id, answer_units) - join with original dataset by id.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
import azure.cognitiveservices.speech as speechsdk

from unit_extractor import UnitExtractor

AZURE_SPEECH_KEY = (
    "8uC03srfUiOfgeOMF3YGZxailptuSES0tY31w3wqGDclCMCOv3poJQQJ99BLACfhMk5XJ3w3AAAAACOGobrR"
)
AZURE_REGION = "swedencentral"
VOICE = "en-US-JennyNeural"

TTS_CONCURRENCY = 50
UNIT_BATCH_SIZE = 16
CHUNK_SIZE = 500

OUTPUT_DIR = Path("/home/ubuntu/voice_lab/data/processed")
OUTPUT_PARQUET = OUTPUT_DIR / "train.parquet"


def synthesize_one(item_id: int, text: str) -> tuple[int, bytes | None]:
    """
    Synthesize text to 16kHz mono PCM wav bytes.
    Returns (item_id, wav_bytes) or (item_id, None) on failure.
    """
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.speech_synthesis_voice_name = VOICE
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    for attempt in range(3):
        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return (item_id, result.audio_data)

    return (item_id, None)


def synthesize_batch(items: list[tuple[int, str]]) -> dict[int, bytes]:
    """
    Synthesize batch of texts concurrently.
    items: [(id, text), ...]
    Returns: {id: wav_bytes} for successful syntheses
    """
    results = {}
    with ThreadPoolExecutor(max_workers=TTS_CONCURRENCY) as executor:
        futures = {
            executor.submit(synthesize_one, item_id, text): item_id for item_id, text in items
        }
        for future in as_completed(futures):
            item_id, wav_bytes = future.result()
            if wav_bytes is not None:
                results[item_id] = wav_bytes
    return results


def extract_units_from_wavs(
    wav_data: dict[int, bytes], extractor: UnitExtractor
) -> dict[int, list[int]]:
    """
    Load wav bytes and extract units in batches.
    Returns: {id: unit_ids}
    """
    ids = list(wav_data.keys())
    audio_arrays = []

    for item_id in ids:
        audio, sr = sf.read(io.BytesIO(wav_data[item_id]))
        audio_arrays.append(audio.astype(np.float32))

    results = {}
    for i in range(0, len(ids), UNIT_BATCH_SIZE):
        batch_ids = ids[i : i + UNIT_BATCH_SIZE]
        batch_audio = audio_arrays[i : i + UNIT_BATCH_SIZE]
        units_list = extractor.extract_units_batch(batch_audio)
        for item_id, units in zip(batch_ids, units_list):
            results[item_id] = units.cpu().tolist()

    return results


def process_chunk(chunk: list[dict], extractor: UnitExtractor) -> list[dict]:
    """
    Process a chunk of samples:
    1. Synthesize answer audio (50 concurrent)
    2. Extract units (batch=32)
    3. Return rows with (id, answer_units)
    """
    tts_items = [(item["id"], item["answer_text"]) for item in chunk]
    wav_data = synthesize_batch(tts_items)
    print(f"    TTS: {len(wav_data)}/{len(tts_items)} succeeded")

    if not wav_data:
        return []

    units = extract_units_from_wavs(wav_data, extractor)

    results = []
    for item in chunk:
        if item["id"] in units:
            results.append({"id": item["id"], "answer_units": units[item["id"]]})

    return results


def append_to_parquet(rows: list[dict], output_path: Path):
    """Append rows to parquet file (create if not exists)."""
    table = pa.table(
        {
            "id": [r["id"] for r in rows],
            "answer_units": [r["answer_units"] for r in rows],
        }
    )

    if output_path.exists():
        existing = pq.read_table(output_path)
        table = pa.concat_tables([existing, table])

    pq.write_table(table, output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support: find max ID and start from there
    start_idx = 0
    total_saved = 0
    if OUTPUT_PARQUET.exists():
        existing = pq.read_table(OUTPUT_PARQUET)
        total_saved = len(existing)
        start_idx = max(existing["id"].to_pylist()) + 1
        print(f"Resuming from idx {start_idx} ({total_saved} samples already saved)")

    print("Loading dataset...")
    dataset = load_dataset("yuekai/InstructS2S-200K", split="train")
    print(f"Dataset size: {len(dataset)}")

    print("Loading UnitExtractor...")
    extractor = UnitExtractor(device="cuda")

    chunk = []
    total_processed = total_saved

    for idx in range(start_idx, len(dataset)):
        row = dataset[idx]
        answer_text = row.get("answer", "")
        if not answer_text or len(answer_text) < 5:
            continue

        chunk.append({"id": idx, "answer_text": answer_text})

        if len(chunk) >= CHUNK_SIZE:
            print(
                f"Processing chunk {total_processed // CHUNK_SIZE + 1} (samples {total_processed}-{total_processed + len(chunk) - 1})..."
            )
            results = process_chunk(chunk, extractor)

            if results:
                append_to_parquet(results, OUTPUT_PARQUET)
                total_saved += len(results)

            total_processed += len(chunk)
            print(
                f"  Saved {len(results)} samples. Total: {total_saved}/{total_processed} processed, {idx + 1}/{len(dataset)} seen"
            )
            chunk = []

    if chunk:
        print("Processing final chunk...")
        results = process_chunk(chunk, extractor)
        if results:
            append_to_parquet(results, OUTPUT_PARQUET)
            total_saved += len(results)
        total_processed += len(chunk)

    print(f"\n=== Complete! ===")
    print(f"Total processed: {total_processed}")
    print(f"Total saved: {total_saved}")
    print(f"Output: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
