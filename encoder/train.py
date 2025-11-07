import torch
from torch.optim import AdamW
from tqdm import tqdm
import os

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from data_loader import get_dataloaders
from config import WHISPER_DTYPE, LLM_DTYPE
from model import SpeechToTextModel
from utils import prepare_template_embeddings


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


def main():
    disable_tokenizer_parallelism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4
    
    # TODO: Increase batch_size or implement gradient accumulation for more stable training
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompt_templates/original.yaml")
    checkpoint_dir = "checkpoints"
    
    print("Loading models...")
    model = SpeechToTextModel().to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
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
    
    optimizer = AdamW(model.adapter.parameters(), lr=learning_rate)
    
    print("Loading dataset...")
    train_dataloader, val_dataloader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        val_ratio=0.01,
        seed=42,
    )

    num_training_steps = len(train_dataloader) * num_epochs
    warmup_ratio = 0.05
    num_warmup_steps = max(1, int(num_training_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
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
            
            # TODO: Add gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.adapter.parameters(), max_norm=1.0)
            
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


if __name__ == "__main__":
    main()