import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import WhisperProcessor
import librosa
import os


class S2SDataset(Dataset):
    """
    Dataset for InstructS2S-200K that loads and preprocesses audio.
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
        output_audio_path = os.path.join(self.cache_dir, sample['output_speech'])
        
        input_audio, _ = librosa.load(input_audio_path, sr=16000)
        output_audio, _ = librosa.load(output_audio_path, sr=16000)
        
        input_features = self.processor(input_audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        output_features = self.processor(output_audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        
        return {
            'input_features': input_features,
            'output_features': output_features,
            'input_text': sample['input_text'],
            'output_text': sample['output_text'],
            'output_unit': sample['output_unit']
        }


def collate_fn(batch):
    """
    Collate function to pad variable-length sequences.
    """
    input_features = [item['input_features'] for item in batch]
    output_features = [item['output_features'] for item in batch]
    
    input_features = torch.nn.utils.rnn.pad_sequence(
        [f.transpose(0, 1) for f in input_features],
        batch_first=True,
        padding_value=0.0
    ).transpose(1, 2)
    
    output_features = torch.nn.utils.rnn.pad_sequence(
        [f.transpose(0, 1) for f in output_features],
        batch_first=True,
        padding_value=0.0
    ).transpose(1, 2)
    
    return {
        'input_features': input_features,
        'output_features': output_features,
        'input_text': [item['input_text'] for item in batch],
        'output_text': [item['output_text'] for item in batch],
        'output_unit': [item['output_unit'] for item in batch]
    }


def get_dataloaders(batch_size=16, num_workers=4, val_split=0.05):
    """
    Create train and validation dataloaders with audio preprocessing.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation
    
    Returns:
        train_dataloader, val_dataloader
    """
    dataset = load_dataset('ICTNLP/InstructS2S-200K', split='train')
    
    cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
    dataset_cache = os.path.join(cache_dir, 'downloads/extracted')
    
    for root, dirs, files in os.walk(cache_dir):
        if 'instruct_en_0-1-user.wav' in files:
            dataset_cache = root.split('/wav/')[0]
            break
    
    def extract_first_pair(example):
        human_turn = example['conversation'][0]
        gpt_turn = example['conversation'][1]
        return {
            'input_speech': human_turn['speech'],
            'input_text': human_turn['text'],
            'input_unit': human_turn['unit'],
            'output_speech': gpt_turn['speech'],
            'output_text': gpt_turn['text'],
            'output_unit': gpt_turn['unit']
        }
    
    dataset = dataset.map(extract_first_pair, remove_columns=['id', 'conversation'])
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
    print(f"Output features shape: {batch['output_features'].shape}")
    print(f"Input text: {batch['input_text'][0][:100]}...")
    print(f"Output text: {batch['output_text'][0][:100]}...")
