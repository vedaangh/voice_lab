import torch


# Audio preprocessing constants
TARGET_SAMPLE_RATE = 16_000
MAX_AUDIO_SAMPLES = 480_000  # 30 seconds at 16 kHz


# Model dtype configuration
WHISPER_DTYPE = torch.bfloat16
LLM_DTYPE = torch.bfloat16

