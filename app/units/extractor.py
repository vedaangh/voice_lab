"""
Extracts discrete speech units from audio using mHuBERT + K-means clustering.
Uses layer 11 features with K=1000 clusters, then deduplicates consecutive indices.
"""

import torch
import numpy as np
import joblib
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
        kmeans_model = joblib.load(kmeans_path)
        self.centroids = torch.from_numpy(kmeans_model.cluster_centers_).float().to(device)

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

        return self._deduplicate(unit_indices)

    @torch.no_grad()
    def extract_units_batch(self, audio_list: list[np.ndarray]) -> list[torch.Tensor]:
        """
        Extract discrete units from a batch of audio waveforms.

        Args:
            audio_list: List of 1D numpy arrays of audio samples at 16kHz

        Returns:
            List of 1D tensors of deduplicated unit indices (0 to NUM_CLUSTERS-1)
        """
        inputs = self.feature_extractor(
            audio_list,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = (
            inputs.attention_mask.to(self.device) if hasattr(inputs, "attention_mask") else None
        )

        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[HUBERT_LAYER]

        results = []
        for i in range(hidden_states.shape[0]):
            features = hidden_states[i]
            if attention_mask is not None:
                seq_len = attention_mask[i].sum().item()
                feat_len = int(seq_len / 320)
                features = features[:feat_len]

            distances = torch.cdist(features.unsqueeze(0), self.centroids.unsqueeze(0)).squeeze(0)
            unit_indices = distances.argmin(dim=-1)
            results.append(self._deduplicate(unit_indices))

        return results

    def _deduplicate(self, units: torch.Tensor) -> torch.Tensor:
        """Remove consecutive duplicate indices."""
        if units.numel() == 0:
            return units
        mask = torch.cat([torch.tensor([True], device=units.device), units[1:] != units[:-1]])
        return units[mask]
