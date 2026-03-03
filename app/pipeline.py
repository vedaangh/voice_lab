"""
Data generation pipeline.

Takes text-in/text-out pairs, synthesizes both sides with Qwen3-TTS,
extracts speech units from answer audio, and saves a HuggingFace Dataset.

Questions are synthesized with all 9 voices (9x augmentation).
Answers use a fixed assistant voice.
All TTS for a chunk is done in a single batch call.
"""

from itertools import batched

from datasets import Dataset, load_dataset

from app.tts import TTS, Voice, TARGET_SAMPLE_RATE
from app.units import UnitExtractor


def run_pipeline(
    dataset_name: str,
    tts: TTS,
    extractor: UnitExtractor,
    output_dir: str,
    assistant_voice: Voice,
    chunk_size: int = 64,
):
    """
    Generate a speech dataset from text pairs.

    For each (question_text, answer_text) pair and each of 9 speakers,
    emits one row with synthesized question audio, answer text, and
    mHuBERT speech units extracted from the synthesized answer audio.
    """
    dataset = load_dataset(dataset_name, split="train")
    speakers = TTS.list_voices()
    rows: list[dict] = []

    for chunk in batched(dataset, chunk_size):
        chunk = list(chunk)
        n = len(chunk)

        # Build a single TTS batch: answers (n) + questions (n * num_speakers)
        # Layout: [answer_0..answer_{n-1}, q_0_spk0..q_{n-1}_spk0, ..., q_0_spkS..q_{n-1}_spkS]
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

        all_results = tts.synthesize_batch(texts, voices)

        answer_audios = [audio for audio, _sr in all_results[:n]]
        answer_units = extractor.extract_units_batch(answer_audios)

        for s_idx, speaker in enumerate(speakers):
            offset = n + s_idx * n
            for i, item in enumerate(chunk):
                q_audio, _ = all_results[offset + i]
                rows.append({
                    "messages": [],
                    "question_audio": {"array": q_audio, "sampling_rate": TARGET_SAMPLE_RATE},
                    "question_text": item["question_text"],
                    "answer": item["answer"],
                    "answer_units": answer_units[i].tolist(),
                    "voice_id": speaker,
                })

    Dataset.from_list(rows).save_to_disk(output_dir)
