"""
Speech-to-speech inference pipeline.
Takes audio input, generates text response + synthesized speech output.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, WhisperProcessor

from model import SpeechToSpeechModel
from config import MODEL_DTYPE

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                              dilation=dilation[i], padding=get_padding(kernel_size, dilation[i])))
            for i in range(3)
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1,
                              dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in range(3)
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class HiFiGANGenerator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_kernels = len(cfg["resblock_kernel_sizes"])
        self.num_upsamples = len(cfg["upsample_rates"])

        self.conv_pre = weight_norm(Conv1d(
            cfg.get("model_in_dim", 80),
            cfg["upsample_initial_channel"],
            7, 1, padding=3
        ))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg["upsample_rates"], cfg["upsample_kernel_sizes"])):
            self.ups.append(weight_norm(ConvTranspose1d(
                cfg["upsample_initial_channel"] // (2 ** i),
                cfg["upsample_initial_channel"] // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2
            )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg["upsample_initial_channel"] // (2 ** (i + 1))
            for k, d in zip(cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"]):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class VariancePredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg["encoder_embed_dim"]
        hidden_dim = cfg["var_pred_hidden_dim"]
        kernel_size = cfg["var_pred_kernel_size"]
        dropout = cfg["var_pred_dropout"]

        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.ln1(x))
        x = self.conv2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.ln2(x))
        return self.proj(x).squeeze(-1)


class CodeHiFiGANGenerator(HiFiGANGenerator):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dict = nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"])
        self.dur_predictor = None
        if cfg.get("dur_predictor_params"):
            self.dur_predictor = VariancePredictor(cfg["dur_predictor_params"])

    def forward(self, code, dur_prediction=False):
        x = self.dict(code).transpose(1, 2)

        if self.dur_predictor and dur_prediction:
            log_dur_pred = self.dur_predictor(x.transpose(1, 2))
            dur_out = torch.clamp(torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1)
            x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        return super().forward(x)


class CodeHiFiGANVocoder(nn.Module):
    def __init__(self, checkpoint_path: str, cfg: dict):
        super().__init__()
        self.model = CodeHiFiGANGenerator(cfg)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state_dict["generator"])
        self.model.eval()
        self.model.remove_weight_norm()

    def forward(self, code: torch.Tensor, dur_prediction: bool = True) -> torch.Tensor:
        mask = code >= 0
        code = code[mask].unsqueeze(0)
        return self.model(code, dur_prediction=dur_prediction).detach().squeeze()


def ctc_postprocess(tokens: torch.Tensor, blank: int) -> list[int]:
    """
    CTC decoding: collapse consecutive duplicates and remove blanks.
    """
    toks = tokens.squeeze(0).tolist()
    deduplicated = [v for i, v in enumerate(toks) if i == 0 or v != toks[i - 1]]
    return [v for v in deduplicated if v != blank]


def load_vocoder(checkpoint_path: str, config_path: str, device: str) -> CodeHiFiGANVocoder:
    with open(config_path) as f:
        cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(checkpoint_path, cfg)
    return vocoder.to(device)


def run_inference(
    audio_path: str,
    model: SpeechToSpeechModel,
    vocoder: CodeHiFiGANVocoder,
    tokenizer,
    whisper_processor,
    prompt_template_path: str,
    device: str,
    max_new_tokens: int = 256,
) -> tuple[str, np.ndarray]:
    """
    Run speech-to-speech inference on an audio file.
    Returns (text_response, audio_waveform).
    """
    with open(prompt_template_path) as f:
        content = f.read()
    before_text, after_text = content.split("<speech>")

    embed_layer = model.speech_text_model.llm.get_input_embeddings()
    before_tokens = tokenizer(before_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    after_tokens = tokenizer(after_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

    with torch.no_grad():
        before_embeds = embed_layer(before_tokens).to(MODEL_DTYPE)
        after_embeds = embed_layer(after_tokens).to(MODEL_DTYPE)

    audio, sr = sf.read(audio_path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    whisper_inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt")
    audio_features = whisper_inputs.input_features.to(device=device, dtype=MODEL_DTYPE)

    with torch.no_grad():
        speech_hidden = model.speech_text_model.whisper_encoder(audio_features).last_hidden_state
        speech_embeds = model.speech_text_model.adapter(speech_hidden)

    inputs_embeds = torch.cat([
        before_embeds.unsqueeze(0),
        speech_embeds,
        after_embeds.unsqueeze(0),
    ], dim=1)

    prompt_length = inputs_embeds.shape[1]

    with torch.no_grad():
        output_ids = model.speech_text_model.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    generated_ids = output_ids.sequences[0]
    text_response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    response_tokens = generated_ids
    response_embeds = embed_layer(response_tokens).unsqueeze(0).to(MODEL_DTYPE)
    full_embeds = torch.cat([inputs_embeds, response_embeds], dim=1)

    with torch.no_grad():
        model.speech_text_model(inputs_embeds=full_embeds)
        hidden_states = model.speech_text_model.hidden_states

    response_hidden = hidden_states[:, prompt_length:, :]

    with torch.no_grad():
        unit_logits = model.speech_decoder(response_hidden)

    unit_ids = unit_logits.argmax(dim=-1)
    units = ctc_postprocess(unit_ids, blank=1000)

    if len(units) == 0:
        return text_response, np.zeros(16000, dtype=np.float32)

    unit_tensor = torch.LongTensor(units).unsqueeze(0).to(device)

    with torch.no_grad():
        waveform = vocoder(unit_tensor, dur_prediction=True)

    return text_response, waveform.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Speech-to-speech inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to output audio file")
    parser.add_argument("--adapter-checkpoint", type=str, default="checkpoints/best_adapter.pt")
    parser.add_argument("--decoder-checkpoint", type=str, default="checkpoints/20251229_021638/decoder/best_decoder.pt")
    parser.add_argument("--vocoder-checkpoint", type=str, default="vocoder/g_00500000")
    parser.add_argument("--vocoder-config", type=str, default="vocoder/config.json")
    parser.add_argument("--prompt-template", type=str, default="prompt_templates/original.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--decoder-hidden-dim", type=int, default=4096)
    parser.add_argument("--decoder-num-heads", type=int, default=32)
    parser.add_argument("--decoder-num-layers", type=int, default=2)
    parser.add_argument("--decoder-intermediate-dim", type=int, default=4096)
    parser.add_argument("--decoder-upsample-rate", type=int, default=25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

    model = SpeechToSpeechModel(
        adapter_checkpoint_path=args.adapter_checkpoint,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_num_layers=args.decoder_num_layers,
        decoder_intermediate_dim=args.decoder_intermediate_dim,
        decoder_upsample_rate=args.decoder_upsample_rate,
    ).to(dtype=MODEL_DTYPE, device=device)

    decoder_ckpt = torch.load(args.decoder_checkpoint, map_location="cpu")
    model.speech_decoder.load_state_dict(decoder_ckpt["decoder_state_dict"])
    model.eval()

    vocoder = load_vocoder(args.vocoder_checkpoint, args.vocoder_config, device)

    print(f"Processing: {args.audio}")
    text_response, audio_output = run_inference(
        audio_path=args.audio,
        model=model,
        vocoder=vocoder,
        tokenizer=tokenizer,
        whisper_processor=whisper_processor,
        prompt_template_path=args.prompt_template,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\nText response: {text_response}")

    sf.write(args.output, audio_output, 16000)
    print(f"Audio saved to: {args.output}")


if __name__ == "__main__":
    main()

