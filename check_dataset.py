from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset('yuekai/InstructS2S-200K', split='train', trust_remote_code=True)

print("\nDataset info:")
print(dataset)

print("\nColumn names:")
print(dataset.column_names)

print("\nDataset features:")
print(dataset.features)

print("\n" + "="*80)
print("EXAMINING FIRST 5 EXAMPLES")
print("="*80)

for i in range(5):
    print(f"\n--- Example {i} ---")
    example = dataset._data.slice(i, 1).to_pydict()
    for key in ['id', 'round', 'question', 'answer']:
        value = example[key][0]
        print(f"{key}: {value}")
    
    speech_token = example['speech_token'][0]
    if isinstance(speech_token, str):
        print(f"speech_token (length {len(speech_token)}): {speech_token[:100]}...")
    else:
        print(f"speech_token: {type(speech_token)}")
    
    print(f"question_audio: {type(example['question_audio'][0])}")

