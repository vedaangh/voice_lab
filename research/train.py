"""
Unified training script for encoder (adapter) and decoder (speech decoder + unit head).
Runs phases sequentially based on config.

Single-GPU:   python train.py
Multi-GPU:    python train.py use_device_map=true   (shards frozen LLM across GPUs)
Quantised:    python train.py load_in_8bit=true      (halves frozen LLM memory)
"""

import os
import math
import shutil
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from transformers import AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration, get_cosine_schedule_with_warmup

from model import SpeechToTextModel, SpeechToSpeechModel, BLANK_IDX
from data_loader import get_dataloaders
from config import MODEL_DTYPE
from utils import prepare_template_embeddings, prepare_batch, encode_speech
from inference import ctc_postprocess, load_vocoder


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_ctc_loss(logits, unit_ids, unit_lengths):
    batch_size, T, _ = logits.shape
    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
    input_lengths = torch.full((batch_size,), T, dtype=torch.long, device=logits.device)
    targets = torch.nn.utils.rnn.pad_sequence(unit_ids, batch_first=True, padding_value=0)
    targets = targets.to(logits.device)
    target_lengths = unit_lengths.to(logits.device)

    valid_mask = target_lengths <= T
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    if not valid_mask.all():
        log_probs = log_probs[:, valid_mask, :]
        input_lengths = input_lengths[valid_mask]
        targets = targets[valid_mask]
        target_lengths = target_lengths[valid_mask]

    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=BLANK_IDX,
        zero_infinity=True,
    )


def sync_checkpoint(local_path: str, checkpoint_bucket: str, rel_key: str):
    """Copy checkpoint to the mounted storage path (cloud-agnostic via SkyPilot mount)."""
    dest = os.path.join(checkpoint_bucket, rel_key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(local_path, dest)
    print(f"Synced {local_path} -> {dest}")


def _needs_manual_placement(cfg) -> bool:
    """True when we should call .to(device) on the model ourselves."""
    return not cfg.use_device_map and not cfg.load_in_8bit


def _place_model(model, cfg, device):
    """
    Place a SpeechToTextModel (or SpeechToSpeechModel) on the correct device(s).

    - Standard:   whole model → single device in MODEL_DTYPE.
    - device_map: LLM already placed by HF; move whisper + adapter to primary GPU.
    - 8bit:       LLM already placed by bitsandbytes; move whisper + adapter.
    """
    if _needs_manual_placement(cfg):
        model = model.to(dtype=MODEL_DTYPE, device=device)
    else:
        # LLM is already on GPU(s). Move the remaining components.
        stm = model if isinstance(model, SpeechToTextModel) else model.speech_text_model
        stm.whisper_encoder = stm.whisper_encoder.to(dtype=MODEL_DTYPE, device=device)
        stm.adapter = stm.adapter.to(dtype=MODEL_DTYPE, device=device)
        if isinstance(model, SpeechToSpeechModel):
            model.speech_decoder = model.speech_decoder.to(dtype=torch.float32, device=device)
    return model


def _get_template_embeds(model, template_path, tokenizer, device):
    """Extract template embeddings from a (possibly sharded) model."""
    if isinstance(model, SpeechToSpeechModel):
        embed_layer = model.speech_text_model.llm.get_input_embeddings()
    else:
        embed_layer = model.llm.get_input_embeddings()
    embed_device = next(embed_layer.parameters()).device
    before_embeds, after_embeds, before_len, after_len = prepare_template_embeddings(
        template_path, tokenizer, embed_layer, embed_device
    )
    return (
        before_embeds.to(device=device, dtype=MODEL_DTYPE),
        after_embeds.to(device=device, dtype=MODEL_DTYPE),
        before_len,
        after_len,
    )


def _is_accumulation_step(step: int, total_steps: int, accum_steps: int) -> bool:
    """True when we should run optimizer.step()."""
    return (step + 1) % accum_steps == 0 or (step + 1) == total_steps


def _run_asr_eval(
    model,
    val_dataset,
    vocoder,
    whisper_asr,
    whisper_processor,
    tokenizer,
    before_embeds,
    after_embeds,
    device,
    epoch,
    num_samples=5,
):
    """
    Run full inference on a few val samples, transcribe with Whisper ASR,
    and print ground-truth vs generated speech transcription.
    """
    model.eval()
    model.speech_text_model.llm.config.use_cache = True

    embed_layer = model.speech_text_model.llm.get_input_embeddings()
    embed_device = next(embed_layer.parameters()).device

    print(f"\n=== ASR Eval (Epoch {epoch + 1}) ===")
    eval_rows = []

    indices = list(range(min(num_samples, len(val_dataset))))
    for idx in indices:
        sample = val_dataset[idx]

        # 1. Encode question audio through Whisper + adapter
        audio = sample["question_audio"]["array"]
        audio = np.pad(audio, (0, max(0, 480_000 - len(audio))))[:480_000]
        whisper_inputs = whisper_processor(
            audio, sampling_rate=16_000, return_tensors="pt"
        )
        audio_features = whisper_inputs.input_features.to(device=device, dtype=MODEL_DTYPE)

        with torch.no_grad():
            speech_embeds = encode_speech(model.speech_text_model, audio_features, MODEL_DTYPE)

        # 2. Build prompt and generate text
        inputs_embeds = torch.cat([
            before_embeds.unsqueeze(0),
            speech_embeds,
            after_embeds.unsqueeze(0),
        ], dim=1)
        prompt_length = inputs_embeds.shape[1]

        with torch.no_grad():
            output_ids = model.speech_text_model.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_ids = output_ids.sequences[0] if hasattr(output_ids, "sequences") else output_ids[0]
        llm_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 3. Get hidden states for decoder
        response_embeds = embed_layer(generated_ids).unsqueeze(0).to(device=device, dtype=MODEL_DTYPE)
        full_embeds = torch.cat([inputs_embeds, response_embeds], dim=1)

        with torch.no_grad():
            model.speech_text_model(inputs_embeds=full_embeds)
            hidden_states = model.speech_text_model.hidden_states.float()
        response_hidden = hidden_states[:, prompt_length:, :]

        # 4. Speech decoder → CTC → units
        decoder_device = next(model.speech_decoder.parameters()).device
        with torch.no_grad():
            unit_logits = model.speech_decoder(response_hidden.to(decoder_device))
        unit_ids = unit_logits.argmax(dim=-1)
        units = ctc_postprocess(unit_ids, blank=BLANK_IDX)

        # 5. Vocoder → waveform
        if len(units) == 0:
            asr_text = "<no units produced>"
        else:
            unit_tensor = torch.LongTensor(units).unsqueeze(0).to(device)
            with torch.no_grad():
                waveform = vocoder(unit_tensor, dur_prediction=True).cpu().numpy()

            # 6. Whisper ASR transcription (on CPU to save GPU memory)
            asr_inputs = whisper_processor(
                waveform, sampling_rate=16_000, return_tensors="pt"
            )
            asr_features = asr_inputs.input_features.to(whisper_asr.device)
            with torch.no_grad():
                asr_ids = whisper_asr.generate(asr_features, max_new_tokens=128)
            asr_text = whisper_processor.batch_decode(asr_ids, skip_special_tokens=True)[0]

        ground_truth = sample["answer"]
        print(f"[{idx + 1}] Ground truth: \"{ground_truth}\"")
        print(f"     LLM text:     \"{llm_text}\"")
        print(f"     ASR of speech: \"{asr_text}\"")
        print(f"     Units: {len(units)} (post-CTC)")

        eval_rows.append([epoch + 1, idx, ground_truth, llm_text, asr_text, len(units)])

    # Log to wandb as a table
    table = wandb.Table(
        columns=["epoch", "sample", "ground_truth", "llm_text", "asr_text", "num_units"],
        data=eval_rows,
    )
    wandb.log({"decoder/eval_table": table})

    model.speech_text_model.llm.config.use_cache = False
    print("=" * 50)


def train_encoder(
    cfg,
    device,
    tokenizer,
    template_path,
    checkpoint_dir,
):
    """Train the adapter (encoder phase)."""
    print("\n" + "=" * 50)
    print("PHASE 1: Encoder Training (Adapter)")
    print("=" * 50)

    model = SpeechToTextModel(
        whisper_model_name=cfg.whisper_name,
        qwen_model_name=cfg.llm_name,
        adapter_hidden_dim=cfg.adapter_hidden_dim,
        adapter_ds_rate=cfg.adapter_ds_rate,
        device_map="auto" if cfg.use_device_map else None,
        load_in_8bit=cfg.load_in_8bit,
    )
    model = _place_model(model, cfg, device)
    model.llm.config.use_cache = False

    before_embeds, after_embeds, before_len, after_len = _get_template_embeds(
        model, template_path, tokenizer, device
    )

    optimizer = AdamW(model.adapter.parameters(), lr=cfg.encoder_learning_rate)

    train_dataloader, val_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        max_answer_tokens=cfg.max_answer_tokens,
    )

    accum = cfg.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accum)
    num_training_steps = num_update_steps_per_epoch * cfg.encoder_num_epochs
    num_warmup_steps = max(1, int(num_training_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if cfg.adapter_checkpoint and cfg.adapter_resume:
        print(f"Resuming encoder from: {cfg.adapter_checkpoint}")
        checkpoint = torch.load(cfg.adapter_checkpoint, map_location=device)
        model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, val_loss: {best_val_loss:.4f}")

    encoder_dir = os.path.join(checkpoint_dir, "encoder")
    os.makedirs(encoder_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg.encoder_num_epochs):
        print(f"\nEncoder Epoch {epoch + 1}/{cfg.encoder_num_epochs}")

        model.eval()
        model.adapter.train()
        total_train_loss = 0
        num_batches = len(train_dataloader)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            inputs = prepare_batch(
                batch,
                model,
                tokenizer,
                before_embeds,
                after_embeds,
                before_len,
                after_len,
                device,
                MODEL_DTYPE,
            )

            outputs = model(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )

            loss = outputs.loss / accum
            loss.backward()

            total_train_loss += outputs.loss.detach().item()

            if _is_accumulation_step(step, num_batches, accum):
                torch.nn.utils.clip_grad_norm_(
                    model.adapter.parameters(), max_norm=cfg.gradient_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = total_train_loss / num_batches

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                inputs = prepare_batch(
                    batch,
                    model,
                    tokenizer,
                    before_embeds,
                    after_embeds,
                    before_len,
                    after_len,
                    device,
                    MODEL_DTYPE,
                )

                outputs = model(
                    inputs_embeds=inputs["inputs_embeds"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )

                total_val_loss += outputs.loss.item()

        val_loss = total_val_loss / len(val_dataloader)

        print(f"Encoder - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        wandb.log(
            {
                "encoder/train_loss": train_loss,
                "encoder/val_loss": val_loss,
                "encoder/learning_rate": scheduler.get_last_lr()[0],
                "encoder/epoch": epoch,
            }
        )

        checkpoint_data = {
            "epoch": epoch,
            "adapter_state_dict": model.adapter.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "before_len": before_len,
            "after_len": after_len,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(encoder_dir, "best_adapter.pt")
            torch.save(checkpoint_data, best_path)
            print(f"Saved best adapter with val loss: {val_loss:.4f}")
            if cfg.checkpoint_bucket:
                sync_checkpoint(best_path, cfg.checkpoint_bucket, f"{os.path.basename(checkpoint_dir)}/encoder/best_adapter.pt")

        last_path = os.path.join(encoder_dir, "last_adapter.pt")
        torch.save(checkpoint_data, last_path)
        if cfg.checkpoint_bucket:
            sync_checkpoint(last_path, cfg.checkpoint_bucket, f"{os.path.basename(checkpoint_dir)}/encoder/last_adapter.pt")

    del model
    torch.cuda.empty_cache()

    return os.path.join(encoder_dir, "best_adapter.pt")


def train_decoder(
    cfg,
    device,
    tokenizer,
    template_path,
    checkpoint_dir,
    adapter_path,
):
    """Train the speech decoder (decoder phase)."""
    print("\n" + "=" * 50)
    print("PHASE 2: Decoder Training (Speech Decoder)")
    print("=" * 50)

    model = SpeechToSpeechModel(
        adapter_checkpoint_path=adapter_path,
        whisper_model_name=cfg.whisper_name,
        qwen_model_name=cfg.llm_name,
        adapter_hidden_dim=cfg.adapter_hidden_dim,
        adapter_ds_rate=cfg.adapter_ds_rate,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_num_heads=cfg.decoder_num_heads,
        decoder_num_layers=cfg.decoder_num_layers,
        decoder_intermediate_dim=cfg.decoder_intermediate_dim,
        decoder_upsample_rate=cfg.decoder_upsample_rate,
        device_map="auto" if cfg.use_device_map else None,
        load_in_8bit=cfg.load_in_8bit,
    )

    if _needs_manual_placement(cfg):
        model.to(device)
        model.speech_text_model.to(dtype=MODEL_DTYPE)
        model.speech_decoder.to(dtype=torch.float32)
    else:
        model = _place_model(model, cfg, device)

    model.speech_text_model.llm.config.use_cache = False

    before_embeds, after_embeds, before_len, after_len = _get_template_embeds(
        model, template_path, tokenizer, device
    )

    trainable_params = list(model.speech_decoder.parameters())
    optimizer = AdamW(trainable_params, lr=cfg.decoder_learning_rate)

    train_dataloader, val_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        whisper_name=cfg.whisper_name,
        max_answer_tokens=cfg.max_answer_tokens,
    )

    accum = cfg.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accum)
    num_training_steps = num_update_steps_per_epoch * cfg.decoder_num_epochs
    num_warmup_steps = max(1, int(num_training_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if cfg.decoder_checkpoint and cfg.decoder_resume:
        print(f"Resuming decoder from: {cfg.decoder_checkpoint}")
        checkpoint = torch.load(cfg.decoder_checkpoint, map_location=device)
        model.speech_decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, val_loss: {best_val_loss:.4f}")

    decoder_dir = os.path.join(checkpoint_dir, "decoder")
    os.makedirs(decoder_dir, exist_ok=True)

    # Load eval components (vocoder on GPU, Whisper ASR on CPU)
    vocoder, whisper_asr, eval_whisper_processor = None, None, None
    eval_samples = getattr(cfg, "eval_samples_per_epoch", 0)
    if eval_samples > 0:
        original_cwd = hydra.utils.get_original_cwd()
        voc_ckpt = os.path.join(original_cwd, cfg.vocoder_checkpoint)
        voc_cfg = os.path.join(original_cwd, cfg.vocoder_config)
        if os.path.exists(voc_ckpt) and os.path.exists(voc_cfg):
            print("Loading vocoder for ASR eval...")
            vocoder = load_vocoder(voc_ckpt, voc_cfg, device)
            print("Loading Whisper ASR model (CPU) for eval...")
            whisper_asr = WhisperForConditionalGeneration.from_pretrained(
                cfg.whisper_name
            ).to("cpu")
            whisper_asr.eval()
            eval_whisper_processor = WhisperProcessor.from_pretrained(cfg.whisper_name)
        else:
            print(f"Warning: vocoder not found at {voc_ckpt}, skipping ASR eval. "
                  f"Run: python scripts/download_vocoder.py")
            eval_samples = 0

    # Keep a reference to the raw val dataset for eval sampling
    val_dataset = val_dataloader.dataset

    for epoch in range(start_epoch, cfg.decoder_num_epochs):
        print(f"\nDecoder Epoch {epoch + 1}/{cfg.decoder_num_epochs}")

        model.train()
        for param in model.speech_text_model.parameters():
            param.requires_grad = False

        total_train_loss = 0
        num_batches = len(train_dataloader)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            inputs = prepare_batch(
                batch,
                model.speech_text_model,
                tokenizer,
                before_embeds,
                after_embeds,
                before_len,
                after_len,
                device,
                MODEL_DTYPE,
            )

            logits = model(inputs["inputs_embeds"], response_start=inputs["prompt_len"])
            loss = compute_ctc_loss(logits.float(), batch["unit_ids"], batch["unit_lengths"])
            scaled_loss = loss / accum
            scaled_loss.backward()

            total_train_loss += loss.detach().item()

            if _is_accumulation_step(step, num_batches, accum):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = total_train_loss / num_batches

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                inputs = prepare_batch(
                    batch,
                    model.speech_text_model,
                    tokenizer,
                    before_embeds,
                    after_embeds,
                    before_len,
                    after_len,
                    device,
                    MODEL_DTYPE,
                )

                logits = model(inputs["inputs_embeds"], response_start=inputs["prompt_len"])
                loss = compute_ctc_loss(logits.float(), batch["unit_ids"], batch["unit_lengths"])

                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_dataloader)

        print(f"Decoder - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        wandb.log(
            {
                "decoder/train_loss": train_loss,
                "decoder/val_loss": val_loss,
                "decoder/learning_rate": scheduler.get_last_lr()[0],
                "decoder/epoch": epoch,
            }
        )

        # ASR eval: run full inference on a few val samples
        if eval_samples > 0 and vocoder is not None:
            _run_asr_eval(
                model=model,
                val_dataset=val_dataset,
                vocoder=vocoder,
                whisper_asr=whisper_asr,
                whisper_processor=eval_whisper_processor,
                tokenizer=tokenizer,
                before_embeds=before_embeds,
                after_embeds=after_embeds,
                device=device,
                epoch=epoch,
                num_samples=eval_samples,
            )

        checkpoint_data = {
            "epoch": epoch,
            "decoder_state_dict": model.speech_decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(decoder_dir, "best_decoder.pt")
            torch.save(checkpoint_data, best_path)
            print(f"Saved best decoder with val loss: {val_loss:.4f}")
            if cfg.checkpoint_bucket:
                sync_checkpoint(best_path, cfg.checkpoint_bucket, f"{os.path.basename(checkpoint_dir)}/decoder/best_decoder.pt")

        last_path = os.path.join(decoder_dir, "last_decoder.pt")
        torch.save(checkpoint_data, last_path)
        if cfg.checkpoint_bucket:
            sync_checkpoint(last_path, cfg.checkpoint_bucket, f"{os.path.basename(checkpoint_dir)}/decoder/last_decoder.pt")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(cfg.seed)

    original_cwd = hydra.utils.get_original_cwd()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(original_cwd, "checkpoints", run_timestamp)
    template_path = os.path.join(original_cwd, "prompt_templates", cfg.prompt_template)

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    wandb.init(
        project=cfg.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"train_{run_timestamp}",
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)

    adapter_path = None

    if cfg.train_encoder:
        if cfg.adapter_checkpoint and not cfg.adapter_resume:
            print(f"Skipping encoder training, using existing adapter: {cfg.adapter_checkpoint}")
            adapter_path = cfg.adapter_checkpoint
            if not os.path.isabs(adapter_path):
                adapter_path = os.path.join(original_cwd, adapter_path)
        else:
            adapter_path = train_encoder(
                cfg,
                device,
                tokenizer,
                template_path,
                checkpoint_dir,
            )
    elif cfg.adapter_checkpoint:
        adapter_path = cfg.adapter_checkpoint
        if not os.path.isabs(adapter_path):
            adapter_path = os.path.join(original_cwd, adapter_path)
        print(f"Using existing adapter checkpoint: {adapter_path}")

    if cfg.train_decoder:
        if adapter_path is None:
            raise ValueError(
                "Decoder training requires adapter_checkpoint when train_encoder=false"
            )
        train_decoder(
            cfg,
            device,
            tokenizer,
            template_path,
            checkpoint_dir,
            adapter_path,
        )

    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
