"""
Unified training script for encoder (adapter) and decoder (speech decoder + unit head).
Runs phases sequentially based on config.
"""

import os
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

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from model import SpeechToTextModel, SpeechToSpeechModel, BLANK_IDX
from data_loader import get_dataloaders
from config import MODEL_DTYPE
from utils import prepare_template_embeddings, prepare_batch


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


def train_encoder(
    cfg,
    device,
    tokenizer,
    before_embeds,
    after_embeds,
    before_len,
    after_len,
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
    ).to(dtype=MODEL_DTYPE, device=device)

    model.llm.config.use_cache = False

    optimizer = AdamW(model.adapter.parameters(), lr=cfg.encoder_learning_rate)

    train_dataloader, val_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        max_answer_tokens=cfg.max_answer_tokens,
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

        for batch in tqdm(train_dataloader, desc="Training"):
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
    before_embeds,
    after_embeds,
    before_len,
    after_len,
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
    ).to(device)
    model.speech_text_model.to(dtype=MODEL_DTYPE)
    model.speech_decoder.to(dtype=torch.float32)

    model.speech_text_model.llm.config.use_cache = False

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
        checkpoint = torch.load(cfg.decoder_checkpoint, map_location=device)
        model.speech_decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, val_loss: {best_val_loss:.4f}")

    decoder_dir = os.path.join(checkpoint_dir, "decoder")
    os.makedirs(decoder_dir, exist_ok=True)

    for epoch in range(start_epoch, cfg.decoder_num_epochs):
        print(f"\nDecoder Epoch {epoch + 1}/{cfg.decoder_num_epochs}")

        model.train()
        for param in model.speech_text_model.parameters():
            param.requires_grad = False

        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc="Training"):
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

    print("Preparing template embeddings...")
    temp_model = SpeechToTextModel(
        whisper_model_name=cfg.whisper_name,
        qwen_model_name=cfg.llm_name,
        adapter_hidden_dim=cfg.adapter_hidden_dim,
        adapter_ds_rate=cfg.adapter_ds_rate,
    ).to(device)
    embed_layer = temp_model.llm.get_input_embeddings()
    before_embeds, after_embeds, before_len, after_len = prepare_template_embeddings(
        template_path=template_path, tokenizer=tokenizer, embed_layer=embed_layer, device=device
    )
    del temp_model
    torch.cuda.empty_cache()

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
                before_embeds,
                after_embeds,
                before_len,
                after_len,
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
            before_embeds,
            after_embeds,
            before_len,
            after_len,
            checkpoint_dir,
            adapter_path,
        )

    wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
