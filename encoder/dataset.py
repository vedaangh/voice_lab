import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import WhisperProcessor
import librosa
import os


class S2SDataset(Dataset):
    """
    Dataset for InstructS2S-200K that loads and preprocesses audio.
    Returns input audio mel-spectrogram and output text only.
    
    Full dataset download: ~127GB (yuekai/InstructS2S-200K version with all audio)
    Lite version: ~491MB (ICTNLP/InstructS2S-200K version)
    Total samples: 200,000 conversation turns
    """
    def __init__(self, hf_dataset, processor, cache_dir):
        self.data = hf_dataset
        self.processor = processor
        self.cache_dir = cache_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        input_audio_path = os.path.join(self.cache_dir, sample['input_speech'])
        input_audio, _ = librosa.load(input_audio_path, sr=16000)
        
        input_features = self.processor(input_audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        
        if input_features.shape[1] < 3000:
            pad_amount = 3000 - input_features.shape[1]
            input_features = F.pad(input_features, (0, pad_amount), value=0.0)
        else:
            input_features = input_features[:, :3000]
        
        return {
            'input_features': input_features,
            'output_text': sample['output_text']
        }


def collate_fn(batch):
    """
    Collate function for batching.
    Audio is already padded to 3000 frames.
    """
    input_features = torch.stack([item['input_features'] for item in batch])
    output_texts = [item['output_text'] for item in batch]
    
    return {
        'input_features': input_features,
        'output_text': output_texts
    }


def get_dataloaders(batch_size=16, num_workers=4, val_split=0.05, use_full_audio=True):
    """
    Create train and validation dataloaders with audio preprocessing.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation
        use_full_audio: If True, downloads full dataset with audio (127GB from yuekai version)
                       If False, uses ICTNLP version (491MB, may have limited audio)
    
    Returns:
        train_dataloader, val_dataloader
    """
    if use_full_audio:
        dataset = load_dataset('yuekai/InstructS2S-200K', split='train', trust_remote_code=True)
    else:
        dataset = load_dataset('ICTNLP/InstructS2S-200K', split='train')
    
    dataset = dataset.filter(lambda x: x == 1, input_columns=['round'])
    
    cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
    dataset_cache = os.path.join(cache_dir, 'downloads/extracted')
    
    for root, dirs, files in os.walk(cache_dir):
        if 'instruct_en_0-1-user.wav' in files:
            dataset_cache = root.split('/wav/')[0]
            break
    
    def extract_first_pair(example):
        return {
            'input_speech': example['question_audio']['path'],
            'output_text': example['answer']
        }
    
    dataset = dataset.map(extract_first_pair, remove_columns=['id', 'round', 'question', 'speech_token', 'question_audio'])
    split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    train_dataset = S2SDataset(split_dataset['train'], processor, dataset_cache)
    val_dataset = S2SDataset(split_dataset['test'], processor, dataset_cache)
    
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
    train_dataloader, val_dataloader = get_dataloaders(batch_size=2)
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches: {len(val_dataloader)}")
    
    print("\nFetching first batch...")
    batch = next(iter(train_dataloader))
    print(f"Input features shape: {batch['input_features'].shape}")
    print(f"Output text: {batch['output_text'][0][:100]}...")
