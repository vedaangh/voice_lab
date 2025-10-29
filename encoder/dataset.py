from datasets import load_dataset


def load_s2s_200k(val_split=0.05):
    """
    Load the InstructS2S-200K dataset and split into train/val.
    Only keeps the first input-output pair from each conversation.
    
    Args:
        val_split: Fraction of data to use for validation
    
    Returns:
        train_dataset, val_dataset
        Each sample has input (human) and output (gpt) with speech_path, text, unit
    """
    dataset = load_dataset('ICTNLP/InstructS2S-200K', split='train')
    
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
    
    return split_dataset['train'], split_dataset['test']


if __name__ == "__main__":
    train_dataset, val_dataset = load_s2s_200k(val_split=0.05)
    
    print("Dataset loaded successfully!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print("\nFirst sample (input-output pair):")
    sample = train_dataset[0]
    print(f"  Input speech: {sample['input_speech']}")
    print(f"  Input text: {sample['input_text']}")
    print(f"  Output speech: {sample['output_speech']}")
    print(f"  Output text: {sample['output_text']}")
    print(f"  Output unit length: {len(sample['output_unit']) if sample['output_unit'] else 0}")
    print("\nDataset features:")
    print(train_dataset.features)

