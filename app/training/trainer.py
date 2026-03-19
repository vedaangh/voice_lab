"""
Training loop for encoder (adapter) and decoder (speech decoder).
Adapted from research/train.py — no Hydra, uses Pydantic config, writes to Modal volumes.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, WhisperProcessor, get_cosine_schedule_with_warmup

from app.training.config import MODEL_DTYPE, TrainingConfig
from app.training.data import (
    MAX_AUDIO_SAMPLES,
    TARGET_SAMPLE_RATE,
    _pad_or_trim,
    get_dataloaders,
    load_parquet_dataset,
)
from app.training.models import BLANK_IDX, SpeechToSpeechModel, SpeechToTextModel
from app.training.utils import prepare_batch, prepare_template_embeddings

MAX_CACHEABLE_SAMPLES = 50_000


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


def precompute_whisper_features(dataset, whisper_processor, whisper_encoder, device, dtype, batch_size=32):
    """Pre-compute Whisper encoder outputs for all samples. Returns dict[int, Tensor] on CPU."""
    cache = {}
    whisper_encoder.eval()
    print(f"Pre-computing Whisper features for {len(dataset)} samples...")

    for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Caching Whisper"):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch_audio = []
        for i in range(start_idx, end_idx):
            audio = _pad_or_trim(dataset[i]["question_audio"]["array"])
            batch_audio.append(audio)

        inputs = whisper_processor(
            batch_audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        with torch.no_grad():
            features = whisper_encoder(
                inputs.input_features.to(device=device, dtype=dtype)
            ).last_hidden_state.cpu()

        for j, i in enumerate(range(start_idx, end_idx)):
            cache[i] = features[j]

    print(f"Cached {len(cache)} Whisper feature tensors.")
    return cache


def train_encoder(
    cfg: TrainingConfig,
    device,
    tokenizer,
    before_embeds,
    after_embeds,
    before_len,
    after_len,
    checkpoint_dir,
    whisper_cache=None,
):
    """Phase 1: Train the adapter (encoder phase)."""
    print("\n" + "=" * 50)
    print("PHASE 1: Encoder Training (Adapter)")
    print("=" * 50)

    model = SpeechToTextModel(
        whisper_model_name=cfg.whisper_name,
        qwen_model_name=cfg.llm_name,
        adapter_hidden_dim=cfg.adapter_hidden_dim,
        adapter_ds_rate=cfg.adapter_ds_rate,
    ).to(dtype=MODEL_DTYPE, device=device)

    model.llm.config.use_cache = False

    optimizer = AdamW(model.adapter.parameters(), lr=cfg.encoder_learning_rate)

    data_dir = os.path.join("/data/output", cfg.dataset_id)
    train_dataloader, val_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        whisper_name=cfg.whisper_name,
        max_answer_tokens=cfg.max_answer_tokens,
        whisper_cache=whisper_cache,
    )

    num_training_steps = len(train_dataloader) * cfg.encoder_num_epochs
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
        ckpt = torch.load(cfg.adapter_checkpoint, map_location=device)
        model.adapter.load_state_dict(ckpt["adapter_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, val_loss: {best_val_loss:.4f}")

    encoder_dir = os.path.join(checkpoint_dir, "encoder")
    os.makedirs(encoder_dir, exist_ok=True)

    pre_encoded = whisper_cache is not None

    for epoch in range(start_epoch, cfg.encoder_num_epochs):
        print(f"\nEncoder Epoch {epoch + 1}/{cfg.encoder_num_epochs}")

        model.eval()
        model.adapter.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            inputs = prepare_batch(
                batch, model, tokenizer,
                before_embeds, after_embeds, before_len, after_len,
                device, MODEL_DTYPE, pre_encoded=pre_encoded,
            )

            outputs = model(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                inputs = prepare_batch(
                    batch, model, tokenizer,
                    before_embeds, after_embeds, before_len, after_len,
                    device, MODEL_DTYPE, pre_encoded=pre_encoded,
                )

                outputs = model(
                    inputs_embeds=inputs["inputs_embeds"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )
                total_val_loss += outputs.loss.item()

        val_loss = total_val_loss / len(val_dataloader)

        print(f"Encoder - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        _wandb_log({
            "encoder/train_loss": train_loss,
            "encoder/val_loss": val_loss,
            "encoder/learning_rate": scheduler.get_last_lr()[0],
            "encoder/epoch": epoch,
        })

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

        last_path = os.path.join(encoder_dir, "last_adapter.pt")
        torch.save(checkpoint_data, last_path)

    del model
    torch.cuda.empty_cache()

    return os.path.join(encoder_dir, "best_adapter.pt")


def train_decoder(
    cfg: TrainingConfig,
    device,
    tokenizer,
    before_embeds,
    after_embeds,
    before_len,
    after_len,
    checkpoint_dir,
    adapter_path,
    whisper_cache=None,
):
    """Phase 2: Train the speech decoder."""
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
    ).to(device)
    model.speech_text_model.to(dtype=MODEL_DTYPE)
    model.speech_decoder.to(dtype=torch.float32)

    model.speech_text_model.llm.config.use_cache = False

    trainable_params = list(model.speech_decoder.parameters())
    optimizer = AdamW(trainable_params, lr=cfg.decoder_learning_rate)

    data_dir = os.path.join("/data/output", cfg.dataset_id)
    train_dataloader, val_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        whisper_name=cfg.whisper_name,
        max_answer_tokens=cfg.max_answer_tokens,
        whisper_cache=whisper_cache,
    )

    num_training_steps = len(train_dataloader) * cfg.decoder_num_epochs
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
        ckpt = torch.load(cfg.decoder_checkpoint, map_location=device)
        model.speech_decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, val_loss: {best_val_loss:.4f}")

    decoder_dir = os.path.join(checkpoint_dir, "decoder")
    os.makedirs(decoder_dir, exist_ok=True)

    pre_encoded = whisper_cache is not None

    for epoch in range(start_epoch, cfg.decoder_num_epochs):
        print(f"\nDecoder Epoch {epoch + 1}/{cfg.decoder_num_epochs}")

        model.train()
        for param in model.speech_text_model.parameters():
            param.requires_grad = False

        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
            inputs = prepare_batch(
                batch, model.speech_text_model, tokenizer,
                before_embeds, after_embeds, before_len, after_len,
                device, MODEL_DTYPE, pre_encoded=pre_encoded,
            )

            logits = model(inputs["inputs_embeds"], response_start=inputs["prompt_len"])
            loss = compute_ctc_loss(logits.float(), batch["unit_ids"], batch["unit_lengths"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                inputs = prepare_batch(
                    batch, model.speech_text_model, tokenizer,
                    before_embeds, after_embeds, before_len, after_len,
                    device, MODEL_DTYPE, pre_encoded=pre_encoded,
                )

                logits = model(inputs["inputs_embeds"], response_start=inputs["prompt_len"])
                loss = compute_ctc_loss(logits.float(), batch["unit_ids"], batch["unit_lengths"])
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_dataloader)

        print(f"Decoder - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        _wandb_log({
            "decoder/train_loss": train_loss,
            "decoder/val_loss": val_loss,
            "decoder/learning_rate": scheduler.get_last_lr()[0],
            "decoder/epoch": epoch,
        })

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

        last_path = os.path.join(decoder_dir, "last_decoder.pt")
        torch.save(checkpoint_data, last_path)

    return os.path.join(decoder_dir, "best_decoder.pt"), best_val_loss


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

_wandb_active = False


def _wandb_init(project: str, config: dict, name: str):
    global _wandb_active
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set — skipping wandb logging.")
        return
    try:
        import wandb
        wandb.init(project=project, config=config, name=name)
        _wandb_active = True
    except Exception as e:
        print(f"wandb init failed: {e}")


def _wandb_log(data: dict):
    if not _wandb_active:
        return
    import wandb
    wandb.log(data)


def _wandb_finish():
    if not _wandb_active:
        return
    import wandb
    wandb.finish()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_training(config: TrainingConfig) -> dict:
    """Run the full training pipeline. Called by the Modal TrainingWorker."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = os.path.join("/checkpoints/runs", config.job_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        f.write(config.model_dump_json(indent=2))

    template_path = os.path.join(
        os.path.dirname(__file__), "prompt_templates", config.prompt_template
    )

    _wandb_init(
        project=config.wandb_project,
        config=config.model_dump(),
        name=f"train_{config.job_id}",
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.llm_name)

    print("Preparing template embeddings...")
    temp_model = SpeechToTextModel(
        whisper_model_name=config.whisper_name,
        qwen_model_name=config.llm_name,
        adapter_hidden_dim=config.adapter_hidden_dim,
        adapter_ds_rate=config.adapter_ds_rate,
    ).to(dtype=MODEL_DTYPE, device=device)
    embed_layer = temp_model.llm.get_input_embeddings()
    before_embeds, after_embeds, before_len, after_len = prepare_template_embeddings(
        template_path=template_path,
        tokenizer=tokenizer,
        embed_layer=embed_layer,
        device=device,
    )

    # Whisper feature caching
    whisper_cache = None
    if config.cache_whisper_features:
        data_dir = os.path.join("/data/output", config.dataset_id)
        dataset = load_parquet_dataset(data_dir)
        if len(dataset) <= MAX_CACHEABLE_SAMPLES:
            whisper_processor = WhisperProcessor.from_pretrained(config.whisper_name)
            whisper_cache = precompute_whisper_features(
                dataset, whisper_processor, temp_model.whisper_encoder, device, MODEL_DTYPE,
            )
        else:
            print(f"Dataset too large for caching ({len(dataset)} > {MAX_CACHEABLE_SAMPLES}), skipping.")

    del temp_model
    torch.cuda.empty_cache()

    adapter_path = None
    encoder_best_val_loss = None
    decoder_best_val_loss = None

    if config.train_encoder:
        if config.adapter_checkpoint and not config.adapter_resume:
            print(f"Skipping encoder training, using existing adapter: {config.adapter_checkpoint}")
            adapter_path = config.adapter_checkpoint
        else:
            adapter_path = train_encoder(
                config, device, tokenizer,
                before_embeds, after_embeds, before_len, after_len,
                checkpoint_dir, whisper_cache,
            )
    elif config.adapter_checkpoint:
        adapter_path = config.adapter_checkpoint
        print(f"Using existing adapter checkpoint: {adapter_path}")

    if config.train_decoder:
        if adapter_path is None:
            raise ValueError(
                "Decoder training requires adapter_checkpoint when train_encoder=false"
            )
        _, decoder_best_val_loss = train_decoder(
            config, device, tokenizer,
            before_embeds, after_embeds, before_len, after_len,
            checkpoint_dir, adapter_path, whisper_cache,
        )

    _wandb_finish()
    print("\nTraining complete!")

    return {
        "job_id": config.job_id,
        "status": "complete",
        "adapter_checkpoint": os.path.join(checkpoint_dir, "encoder", "best_adapter.pt") if config.train_encoder else adapter_path,
        "decoder_checkpoint": os.path.join(checkpoint_dir, "decoder", "best_decoder.pt") if config.train_decoder else None,
        "encoder_best_val_loss": encoder_best_val_loss,
        "decoder_best_val_loss": decoder_best_val_loss,
    }
