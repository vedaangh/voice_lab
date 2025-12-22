"""
Extracts discrete speech units from audio using mHuBERT + K-means clustering.
Uses layer 11 features with K=1000 clusters, then deduplicates consecutive indices.
"""
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from transformers import HubertModel, Wav2Vec2FeatureExtractor

HUBERT_MODEL = "voidful/mhubert-base"
KMEANS_FILENAME = "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin"
TARGET_SAMPLE_RATE = 16_000
HUBERT_LAYER = 11
NUM_CLUSTERS = 1000


class UnitExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL)
        self.hubert = HubertModel.from_pretrained(HUBERT_MODEL).to(device)
        self.hubert.eval()
        
        kmeans_path = hf_hub_download(repo_id=HUBERT_MODEL, filename=KMEANS_FILENAME)
        self.centroids = torch.from_numpy(np.load(kmeans_path)).to(device)

    @torch.no_grad()
    def extract_units(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract discrete units from raw audio waveform.
        
        Args:
            audio: 1D numpy array of audio samples at 16kHz
            
        Returns:
            1D tensor of deduplicated unit indices (0 to NUM_CLUSTERS-1)
        """
        inputs = self.feature_extractor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)
        
        outputs = self.hubert(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[HUBERT_LAYER]
        features = hidden_states.squeeze(0)
        
        distances = torch.cdist(features, self.centroids)
        unit_indices = distances.argmin(dim=-1)
        
        deduplicated = self._deduplicate(unit_indices)
        return deduplicated

    def _deduplicate(self, units: torch.Tensor) -> torch.Tensor:
        """Remove consecutive duplicate indices."""
        if units.numel() == 0:
            return units
        mask = torch.cat([torch.tensor([True], device=units.device), units[1:] != units[:-1]])
        return units[mask]


