import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import os

from transformers import AutoTokenizer
from dataset import get_dataloaders
from model import SpeechToTextModel
from utils import prepare_template_embeddings


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
    batch_size = len(batch['output_text'])
    audio_features = batch['input_features'].to(device)
    speech_lengths = batch['speech_lengths'].to(device)
    
    with torch.no_grad():
        speech_hidden = model.whisper_encoder(audio_features).last_hidden_state
    
    speech_embeds = model.adapter(speech_hidden)
    
    whisper_lengths = (speech_lengths + 1) // 2
    adapter_lengths = whisper_lengths // 5
    
    speech_embeds_list = []
    for i in range(batch_size):
        actual_len = adapter_lengths[i].item()
        speech_embeds_list.append(speech_embeds[i, :actual_len])
    
    max_speech_len = max(len(emb) for emb in speech_embeds_list)
    padded_speech_embeds = []
    for emb in speech_embeds_list:
        if emb.shape[0] < max_speech_len:
            pad_size = max_speech_len - emb.shape[0]
            padded = F.pad(emb.unsqueeze(0), (0, 0, 0, pad_size), value=0.0).squeeze(0)
        else:
            padded = emb
        padded_speech_embeds.append(padded)
    
    speech_embeds = torch.stack(padded_speech_embeds, dim=0)
    speech_len = speech_embeds.shape[1]
    
    response_tokens = tokenizer(
        batch['output_text'],
        return_tensors='pt',
        padding=True,
        add_special_tokens=False
    ).to(device)
    response_ids = response_tokens['input_ids']
    response_mask = response_tokens['attention_mask']
    
    embed_layer = model.qwen.get_input_embeddings()
    response_embeds = embed_layer(response_ids)
    
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
    labels = torch.cat([prompt_labels, response_ids], dim=1)
    
    prompt_mask = torch.ones(batch_size, prompt_len, dtype=torch.long, device=device)
    attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
    
    return {
        'inputs_embeds': inputs_embeds,
        'labels': labels,
        'attention_mask': attention_mask
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-4
    
    # TODO: Increase batch_size or implement gradient accumulation for more stable training
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompt_templates/original.yaml")
    checkpoint_dir = "checkpoints"
    
    print("Loading models...")
    model = SpeechToTextModel().to(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    
    model.qwen = model.qwen.to(dtype=torch.bfloat16)
    model.whisper_encoder = model.whisper_encoder.to(dtype=torch.bfloat16)
    model.adapter = model.adapter.to(dtype=torch.bfloat16)
    
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
    
    # TODO: Add learning rate scheduler (e.g., cosine annealing with warmup) for better convergence
    
    print("Loading dataset...")
    train_dataloader, val_dataloader = get_dataloaders(
        batch_size=batch_size,
        cache_dir='data/preprocessed'
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
                'val_loss': val_loss,
                'before_len': before_len,
                'after_len': after_len,
                'speech_len': speech_len,
            }, os.path.join(checkpoint_dir, "best_adapter.pt"))
            print(f"Saved best adapter with val loss: {val_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'adapter_state_dict': model.adapter.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, "last_adapter.pt"))


if __name__ == "__main__":
    main()