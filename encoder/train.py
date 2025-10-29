import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
encoder = model.get_encoder()

for param in encoder.parameters():
    param.requires_grad = False

encoder = encoder.to(device)

optimizer = AdamW(filter(lambda p: p.requires_grad, encoder.parameters()), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    encoder.train()
    total_train_loss = 0
    
    for inputs, labels in tqdm(train_dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = encoder(inputs)
        
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    train_loss = total_train_loss / len(train_dataloader)
    
    encoder.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = encoder(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_val_loss += loss.item()
    
    val_loss = total_val_loss / len(val_dataloader)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, "checkpoints/best_model.pt")
        print(f"Saved best model with val loss: {val_loss:.4f}")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, "checkpoints/last_model.pt")
