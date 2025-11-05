import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os


class S2SDataset(Dataset):
    """
    Dataset for InstructS2S-200K that loads preprocessed audio features from cache.
    Returns input audio mel-spectrogram and output text only.
    
    Expects preprocessed cache created by preprocess.py.
    """
    def __init__(self, cache_dir, split='train', processor=None):
        self.cache_dir = cache_dir
        self.split = split
        
        metadata_path = os.path.join(cache_dir, 'metadata.pt')
        metadata = torch.load(metadata_path)
        
        if split == 'train':
            self.indices = metadata['train_indices']
        else:
            self.indices = metadata['val_indices']
        
        self.processor = processor
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        cache_path = os.path.join(self.cache_dir, f'{self.split}_{actual_idx}.pt')
        
        data = torch.load(cache_path)
        
        return {
            'input_features': data['input_features'],
            'output_text': data['output_text'],
            'speech_length': data['speech_length']
        }


def collate_fn(batch):
    """
    Collate function for batching with variable-length padding.
    Pads audio features to longest in batch.
    """
    input_features_list = [item['input_features'] for item in batch]
    output_texts = [item['output_text'] for item in batch]
    speech_lengths = torch.tensor([item['speech_length'] for item in batch], dtype=torch.long)
    
    max_len = max(f.shape[1] for f in input_features_list)
    padded_features = []
    for features in input_features_list:
        if features.shape[1] < max_len:
            pad_amount = max_len - features.shape[1]
            padded = F.pad(features, (0, pad_amount), value=0.0)
        else:
            padded = features
        padded_features.append(padded)
    
    input_features = torch.stack(padded_features, dim=0)
    
    return {
        'input_features': input_features,
        'output_text': output_texts,
        'speech_lengths': speech_lengths
    }


def get_dataloaders(batch_size=16, num_workers=4, cache_dir='data/preprocessed'):
    """
    Create train and validation dataloaders from preprocessed cache.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        cache_dir: Directory containing preprocessed cache files
    
    Returns:
        train_dataloader, val_dataloader
    """
    train_dataset = S2SDataset(cache_dir, split='train')
    val_dataset = S2SDataset(cache_dir, split='val')
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    print("Loading dataloaders...")
    train_dataloader, val_dataloader = get_dataloaders(batch_size=2, cache_dir='data/preprocessed')
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    
    print("\nFetching first batch...")
    batch = next(iter(train_dataloader))
    print(f"Input features shape: {batch['input_features'].shape}")
    print(f"Speech lengths: {batch['speech_lengths']}")
    print(f"Output text: {batch['output_text'][0][:100]}...")


