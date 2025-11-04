import torch
import torch.nn as nn
from transformers import WhisperModel, AutoModelForCausalLM
from adapter import Adapter


class SpeechToTextModel(nn.Module):
    """
    Combined model: Whisper encoder (frozen) + Adapter (trainable) + Qwen (frozen).
    """
    def __init__(self, whisper_model_name="openai/whisper-large-v3", 
                 qwen_model_name="Qwen/Qwen3-4B-Instruct-2507"):
        super().__init__()
        
        from transformers import WhisperConfig
        
        config = WhisperConfig.from_pretrained(whisper_model_name)
        whisper_dim = config.d_model
        
        whisper_model = WhisperModel.from_pretrained(whisper_model_name)
        self.whisper_encoder = whisper_model.encoder
        
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False
        
        qwen_temp = AutoModelForCausalLM.from_pretrained(qwen_model_name)
        qwen_dim = qwen_temp.config.hidden_size
        
        self.adapter = Adapter(
            input_dim=whisper_dim,
            hidden_dim=2048,
            output_dim=qwen_dim,
            ds_rate=5
        )
        
        self.qwen = qwen_temp
        for param in self.qwen.parameters():
            param.requires_grad = False
    
    def forward(self, audio_features=None, inputs_embeds=None, 
                attention_mask=None, labels=None):
        """
        Forward pass.
        
        Training mode: inputs_embeds provided, forward through Qwen
        Inference mode: audio_features provided, return speech embeddings
        """
        if inputs_embeds is not None:
            return self.qwen(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            with torch.no_grad():
                speech_hidden = self.whisper_encoder(audio_features).last_hidden_state
            speech_embeds = self.adapter(speech_hidden)
            return speech_embeds
