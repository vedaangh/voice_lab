import os
import random
from datetime import datetime
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from data_loader import get_dataloaders
from config import WHISPER_DTYPE, LLM_DTYPE
from model import SpeechToTextModel
from utils import prepare_template_embeddings


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def disable_tokenizer_parallelism():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def prepare_batch(batch, model, tokenizer, before_embeds, after_embeds,
                  before_len, after_len, device):
    """
    Prepare batch: process audio, tokenize text, combine embeddings.
    Handles variable-length speech sequences.
    
    Returns:
        inputs_embeds: (batch, total_seq_len, hidden_dim)
        labels: (batch, total_seq_len) - -100 for prompt/speech, token IDs for response
        attention_mask: (batch, total_seq_len) - 1s for real tokens, 0s for padding
    """
    batch_size = len(batch['answer_input_ids'])
    audio_features = batch['input_features'].to(device=device, dtype=WHISPER_DTYPE)

    with torch.no_grad():
        speech_hidden = model.whisper_encoder(audio_features).last_hidden_state

    speech_embeds = model.adapter(speech_hidden)
    speech_len = speech_embeds.shape[1]

    speech_mask = torch.ones(batch_size, speech_len, dtype=torch.long, device=device)

    answer_id_list = batch['answer_input_ids']
    max_answer_len = max((ids.shape[0] for ids in answer_id_list), default=0)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    response_ids = torch.full(
        (batch_size, max_answer_len),
        pad_token_id,
        dtype=torch.long,
        device=device
    )
    response_mask = torch.zeros(batch_size, max_answer_len, dtype=torch.long, device=device)
    for idx, ids in enumerate(answer_id_list):
        if ids.numel() == 0:
            continue
        length = ids.shape[0]
        response_ids[idx, :length] = ids.to(device)
        response_mask[idx, :length] = 1
    
    embed_layer = model.qwen.get_input_embeddings()
    response_embeds = embed_layer(response_ids).to(dtype=LLM_DTYPE)
    
    before_embeds_batch = before_embeds.unsqueeze(0).expand(batch_size, -1, -1)
    after_embeds_batch = after_embeds.unsqueeze(0).expand(batch_size, -1, -1)
    
    inputs_embeds = torch.cat([
        before_embeds_batch,
        speech_embeds,
        after_embeds_batch,
        response_embeds
    ], dim=1)
    
    prompt_len = before_len + speech_len + after_len
    prompt_labels = torch.full(
        (batch_size, prompt_len),
        -100,
        dtype=torch.long,
        device=device
    )
    response_labels = response_ids.clone()
    response_labels[response_mask == 0] = -100
    labels = torch.cat([prompt_labels, response_labels], dim=1)
    
    before_mask = torch.ones(batch_size, before_len, dtype=torch.long, device=device)
    after_mask = torch.ones(batch_size, after_len, dtype=torch.long, device=device)
    prompt_mask = torch.cat([before_mask, speech_mask, after_mask], dim=1)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    
    return {
        'inputs_embeds': inputs_embeds,
        'labels': labels,
        'attention_mask': attention_mask
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    disable_tokenizer_parallelism()
    set_seed(cfg.seed)
    
    original_cwd = hydra.utils.get_original_cwd()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(original_cwd, "checkpoints", "encoder", run_timestamp)
    template_path = os.path.join(original_cwd, "prompt_templates/original.yaml")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    wandb.init(
        project=cfg.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    print("Loading models...")
    model = SpeechToTextModel(
        whisper_model_name=cfg.whisper_name,
        qwen_model_name=cfg.llm_name,
        adapter_hidden_dim=cfg.adapter_hidden_dim,
        adapter_ds_rate=cfg.adapter_ds_rate,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
    if hasattr(model.qwen, "config"):
        model.qwen.config.use_cache = False
    
    model.qwen = model.qwen.to(dtype=LLM_DTYPE)
    model.whisper_encoder = model.whisper_encoder.to(dtype=WHISPER_DTYPE)
    model.adapter = model.adapter.to(dtype=LLM_DTYPE)
    
    print("Preparing template...")
    embed_layer = model.qwen.get_input_embeddings()
    before_embeds, after_embeds, before_len, after_len = prepare_template_embeddings(
        template_path=template_path,
        tokenizer=tokenizer,
        embed_layer=embed_layer,
        device=device
    )
    
    dummy_audio = torch.randn(1, 128, 3000, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        dummy_output = model.whisper_encoder(dummy_audio)
        dummy_adapter_output = model.adapter(dummy_output.last_hidden_state)
    speech_len = dummy_adapter_output.shape[1]
    print(f"Speech sequence length after adapter: {speech_len}")
    
    optimizer = AdamW(model.adapter.parameters(), lr=cfg.learning_rate)
    
    print("Loading dataset...")
    train_dataloader, val_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=cfg.batch_size,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )

    num_training_steps = len(train_dataloader) * cfg.num_epochs
    num_warmup_steps = max(1, int(num_training_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if cfg.resume and cfg.resume_path:
        resume_path = cfg.resume_path
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(original_cwd, resume_path)
        
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.adapter.load_state_dict(checkpoint['adapter_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"Resumed from epoch {start_epoch}, val_loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: resume_path '{resume_path}' not found, starting fresh")
    
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.num_epochs}")
        
        model.eval()
        model.adapter.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            inputs = prepare_batch(
                batch, model, tokenizer,
                before_embeds, after_embeds,
                before_len, after_len, device
            )
            
            outputs = model(
                audio_features=None,
                inputs_embeds=inputs['inputs_embeds'],
                attention_mask=inputs['attention_mask'],
                labels=inputs['labels']
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
                    before_embeds, after_embeds,
                    before_len, after_len, device
                )
                
                outputs = model(
                    audio_features=None,
                    inputs_embeds=inputs['inputs_embeds'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['labels']
                )
                
                total_val_loss += outputs.loss.item()
        
        val_loss = total_val_loss / len(val_dataloader)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch,
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'adapter_state_dict': model.adapter.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'before_len': before_len,
                'after_len': after_len,
            }, os.path.join(checkpoint_dir, "best_adapter.pt"))
            print(f"Saved best adapter with val loss: {val_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'adapter_state_dict': model.adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, "last_adapter.pt"))
    
    wandb.finish()


if __name__ == "__main__":
    main()
