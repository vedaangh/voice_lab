"""
Speech models: SpeechToTextModel (encoder) and SpeechToSpeechModel (encoder + decoder).
"""

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperConfig, AutoModelForCausalLM, LlamaModel, LlamaConfig

NUM_UNITS = 1000
BLANK_IDX = 1000


class Adapter(nn.Module):
    """
    2-layer MLP adapter with ReLU activation and k-based downsampling.
    Downsamples by concatenating every k frames.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, ds_rate=5):
        super().__init__()
        self.k = ds_rate
        self.fc1 = nn.Linear(input_dim * self.k, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        remainder = seq_len % self.k
        if remainder > 0:
            pad_size = self.k - remainder
            x = nn.functional.pad(x, (0, 0, 0, pad_size), value=0.0)
            seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SpeechDecoder(nn.Module):
    """
    Non-autoregressive transformer decoder following LLaMA architecture.
    Takes LLM hidden states, upsamples by λ, outputs unit logits.
    """

    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        intermediate_dim: int = 4096,
        upsample_rate: int = 25,
        num_units: int = NUM_UNITS,
    ):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        config = LlamaConfig(
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=intermediate_dim,
            _attn_implementation="sdpa",
        )
        self.transformer = LlamaModel(config)
        self.unit_head = nn.Linear(hidden_dim, num_units + 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.repeat_interleave(self.upsample_rate, dim=1)
        hidden = self.transformer(inputs_embeds=x).last_hidden_state
        return self.unit_head(hidden)


class SpeechToTextModel(nn.Module):
    """
    Whisper encoder (frozen) + Adapter (trainable) + Qwen (frozen).
    """

    def __init__(
        self,
        whisper_model_name="openai/whisper-large-v3",
        qwen_model_name="Qwen/Qwen3-4B-Instruct-2507",
        adapter_hidden_dim=2048,
        adapter_ds_rate=5,
        device_map=None,
        load_in_8bit=False,
    ):
        super().__init__()

        config = WhisperConfig.from_pretrained(whisper_model_name)
        whisper_dim = config.d_model

        whisper_model = WhisperModel.from_pretrained(whisper_model_name)
        self.whisper_encoder = whisper_model.encoder

        for param in self.whisper_encoder.parameters():
            param.requires_grad = False

        llm_kwargs = {}
        if device_map is not None:
            llm_kwargs["device_map"] = device_map
        if load_in_8bit:
            llm_kwargs["load_in_8bit"] = True
            llm_kwargs["torch_dtype"] = torch.bfloat16

        self.llm = AutoModelForCausalLM.from_pretrained(qwen_model_name, **llm_kwargs)
        qwen_dim = self.llm.config.hidden_size

        self.adapter = Adapter(
            input_dim=whisper_dim,
            hidden_dim=adapter_hidden_dim,
            output_dim=qwen_dim,
            ds_rate=adapter_ds_rate,
        )

        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        """
        Forward pass through Qwen with pre-constructed embeddings.
        Returns the full CausalLMOutput (use outputs.hidden_states[-1] for last hidden state).
        """
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )


class SpeechToSpeechModel(nn.Module):
    """
    Full speech-to-speech model.
    Frozen: Whisper encoder, Adapter, Qwen (from SpeechToTextModel)
    Trainable: SpeechDecoder
    """

    def __init__(
        self,
        adapter_checkpoint_path: str,
        whisper_model_name: str = "openai/whisper-large-v3",
        qwen_model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
        adapter_hidden_dim: int = 2048,
        adapter_ds_rate: int = 5,
        decoder_hidden_dim: int = 1024,
        decoder_num_heads: int = 16,
        decoder_num_layers: int = 12,
        decoder_intermediate_dim: int = 4096,
        decoder_upsample_rate: int = 25,
        device_map=None,
        load_in_8bit=False,
    ):
        super().__init__()

        self.speech_text_model = SpeechToTextModel(
            whisper_model_name=whisper_model_name,
            qwen_model_name=qwen_model_name,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_ds_rate=adapter_ds_rate,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
        )

        checkpoint = torch.load(adapter_checkpoint_path, map_location="cpu")
        self.speech_text_model.adapter.load_state_dict(checkpoint["adapter_state_dict"])

        for param in self.speech_text_model.parameters():
            param.requires_grad = False

        self.speech_decoder = SpeechDecoder(
            input_dim=self.speech_text_model.llm.config.hidden_size,
            hidden_dim=decoder_hidden_dim,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            intermediate_dim=decoder_intermediate_dim,
            upsample_rate=decoder_upsample_rate,
        )

    def forward(self, inputs_embeds, response_start: int):
        """
        Args:
            inputs_embeds: Pre-constructed embeddings [batch, seq_len, hidden_dim]
            response_start: Index where the response starts (prompt length)
        Returns:
            Unit logits [batch, T, NUM_UNITSstatus
              + 1] where T = response_len * upsample_rate
        """
        with torch.no_grad():
            outputs = self.speech_text_model(inputs_embeds=inputs_embeds)
            hidden_states = outputs.hidden_states[-1].float()

        response_hidden = hidden_states[:, response_start:, :]
        return self.speech_decoder(response_hidden)
