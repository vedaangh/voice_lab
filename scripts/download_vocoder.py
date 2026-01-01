"""
Download the HiFi-GAN vocoder checkpoint and config from fairseq.
This is a unit-based vocoder trained on LJSpeech with mHuBERT units (K=1000, layer 11).
"""

import urllib.request
from pathlib import Path

VOCODER_DIR = Path(__file__).parent.parent / "vocoder"
VOCODER_CHECKPOINT = VOCODER_DIR / "g_00500000"
VOCODER_CONFIG = VOCODER_DIR / "config.json"

BASE_URL = "https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj"


def download_file(url: str, dest: Path):
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"Done: {dest.stat().st_size / 1e6:.1f} MB")


def main():
    VOCODER_DIR.mkdir(exist_ok=True)

    if not VOCODER_CHECKPOINT.exists():
        download_file(f"{BASE_URL}/g_00500000", VOCODER_CHECKPOINT)
    else:
        print(f"Checkpoint already exists: {VOCODER_CHECKPOINT}")

    if not VOCODER_CONFIG.exists():
        download_file(f"{BASE_URL}/config.json", VOCODER_CONFIG)
    else:
        print(f"Config already exists: {VOCODER_CONFIG}")

    print("\nVocoder ready!")


if __name__ == "__main__":
    main()


