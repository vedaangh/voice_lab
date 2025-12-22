"""
Non-autoregressive transformer decoder following LLaMA architecture.
Takes LLM hidden states, upsamples by λ, outputs unit embeddings.
Uses HuggingFace's LlamaModel.
"""
import torch.nn as nn
from transformers import LlamaModel, LlamaConfig


class SpeechDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        intermediate_dim: int = 4096,
        upsample_rate: int = 25,
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

    def forward(self, x):
        """
        Args:
            x: LLM hidden states [batch, seq_len, input_dim]
        Returns:
            Unit embeddings [batch, seq_len * upsample_rate, hidden_dim]
        """
        x = self.input_proj(x)
        x = x.repeat_interleave(self.upsample_rate, dim=1)
        
        out = self.transformer(inputs_embeds=x)
        return out.last_hidden_state
