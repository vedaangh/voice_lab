import torch
from datasets import load_dataset
import os
from tqdm import tqdm
from multiprocessing import Pool


def process_sample(args):
    idx, sample, output_dir, split = args
    
    try:
        from transformers import WhisperProcessor
        import torchaudio.functional as audio_F
        
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        
        audio_decoder = sample['question_audio']
        
        audio_samples = audio_decoder.get_all_samples()
        audio_tensor = audio_samples.data
        native_sr = audio_samples.sample_rate
        
        if native_sr != 16000:
            audio_tensor = audio_F.resample(audio_tensor, native_sr, 16000)
        
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0)
        
        input_audio = audio_tensor.squeeze().numpy()
        
        input_features = processor(input_audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        
        speech_length = input_features.shape[1]
        
        data = {
            'input_features': input_features,
            'output_text': sample['answer'],
            'speech_length': speech_length
        }
        
        cache_path = os.path.join(output_dir, f'{split}_{idx}.pt')
        torch.save(data, cache_path)
        
        return idx, True
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return idx, False


def preprocess_dataset(
    dataset_name='yuekai/InstructS2S-200K',
    output_dir='data/preprocessed',
    num_workers=8,
    use_full_audio=True,
    val_split=0.05
):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset: {dataset_name}")
    if use_full_audio:
        dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
    else:
        dataset = load_dataset('ICTNLP/InstructS2S-200K', split='train')
    
    print("Filtering to round==1...")
    dataset = dataset.filter(lambda x: x == 1, input_columns=['round'])
    
    print("Splitting train/val...")
    split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
    
    train_data = split_dataset['train']
    val_data = split_dataset['test']
    
    print(f"Preprocessing {len(train_data)} training samples...")
    train_args = [(idx, train_data[idx], output_dir, 'train') 
                  for idx in range(len(train_data))]
    
    train_indices = []
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for idx, success in tqdm(pool.imap(process_sample, train_args), total=len(train_data)):
                if success:
                    train_indices.append(idx)
    else:
        for args in tqdm(train_args):
            idx, success = process_sample(args)
            if success:
                train_indices.append(idx)
    
    print(f"Preprocessing {len(val_data)} validation samples...")
    val_args = [(idx, val_data[idx], output_dir, 'val') 
                for idx in range(len(val_data))]
    
    val_indices = []
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for idx, success in tqdm(pool.imap(process_sample, val_args), total=len(val_data)):
                if success:
                    val_indices.append(idx)
    else:
        for args in tqdm(val_args):
            idx, success = process_sample(args)
            if success:
                val_indices.append(idx)
    
    metadata = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'train_split_idx': len(train_data)
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.pt')
    torch.save(metadata, metadata_path)
    
    print(f"Preprocessing complete!")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"Cache directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yuekai/InstructS2S-200K')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_full_audio', action='store_true', default=True)
    parser.add_argument('--no_full_audio', dest='use_full_audio', action='store_false')
    parser.add_argument('--val_split', type=float, default=0.05)
    
    args = parser.parse_args()
    
    preprocess_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        use_full_audio=args.use_full_audio,
        val_split=args.val_split
    )

