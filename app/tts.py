"""Qwen3-TTS wrapper."""

from dataclasses import dataclass

import librosa
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

SPEAKERS = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]

TARGET_SAMPLE_RATE = 16_000


@dataclass
class Voice:
    speaker: str
    language: str = "English"
    instruct: str = ""


class TTS:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "cuda",
    ):
        self.model = Qwen3TTSModel.from_pretrained(
            model_name, device_map=device, dtype=torch.bfloat16,
        )

    def synthesize(self, text: str, voice: Voice) -> tuple[np.ndarray, int]:
        """Synthesize a single utterance. Returns (audio_16k, 16000)."""
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=voice.language,
            speaker=voice.speaker,
            instruct=voice.instruct,
        )
        audio = _resample(wavs[0], sr)
        return audio, TARGET_SAMPLE_RATE

    def synthesize_batch(
        self, texts: list[str], voices: list[Voice],
    ) -> list[tuple[np.ndarray, int]]:
        """Synthesize a batch with per-item voice control."""
        wavs, sr = self.model.generate_custom_voice(
            text=texts,
            language=[v.language for v in voices],
            speaker=[v.speaker for v in voices],
            instruct=[v.instruct for v in voices],
        )
        return [(_resample(w, sr), TARGET_SAMPLE_RATE) for w in wavs]

    @staticmethod
    def list_voices() -> list[str]:
        return list(SPEAKERS)


def _resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to 16kHz if needed."""
    if orig_sr == TARGET_SAMPLE_RATE:
        return audio.astype(np.float32)
    return librosa.resample(
        audio.astype(np.float32), orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE,
    )
