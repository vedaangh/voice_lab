import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch using cross entropy loss.
    Expects dataloader to return (inputs, labels) where labels are token ids.
    """
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Validate the model and return average loss.
    """
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    learning_rate=1e-4,
    device="cuda",
    save_dir="checkpoints"
):
    """
    Main training loop with checkpointing.
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_loss = validate(model, val_dataloader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f"{save_dir}/best_model.pt")
            print(f"Saved best model with val loss: {val_loss:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, f"{save_dir}/last_model.pt")

