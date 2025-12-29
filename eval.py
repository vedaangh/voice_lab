"""
Simple evaluation script - runs 10 examples through the speech-to-text model.
Prints the ground truth answer and the model's generated response.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoTokenizer, WhisperProcessor
from datasets import load_dataset, Audio
from model import SpeechToTextModel
from config import MODEL_DTYPE

ADAPTER_PATH = "checkpoints/20251223_114443/encoder/best_adapter.pt"
NUM_EXAMPLES = 10
MAX_NEW_TOKENS = 200

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    
    model = SpeechToTextModel().to(dtype=MODEL_DTYPE, device=device)
    
    checkpoint = torch.load(ADAPTER_PATH, map_location="cpu")
    model.adapter.load_state_dict(checkpoint["adapter_state_dict"])
    model.eval()
    
    # Load prompt template
    with open("prompt_templates/original.yaml", "r") as f:
        content = f.read()
    before_text, after_text = content.split("<speech>")
    
    embed_layer = model.llm.get_input_embeddings()]
    before_tokens = tokenizer(before_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    after_tokens = tokenizer(after_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    
    with torch.no_grad():
        before_embeds = embed_layer(before_tokens).to(MODEL_DTYPE)
        after_embeds = embed_layer(after_tokens).to(MODEL_DTYPE)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("yuekai/InstructS2S-200K", split="train")
    dataset = dataset.cast_column("question_audio", Audio(sampling_rate=16000))
    
    print(f"\n{'='*60}")
    print(f"Running {NUM_EXAMPLES} examples")
    print(f"{'='*60}\n")
    
    for i in range(NUM_EXAMPLES):
        sample = dataset[i]
        
        # Process audio
        audio = sample["question_audio"]["array"]
        whisper_inputs = whisper_processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        audio_features = whisper_inputs.input_features.to(device=device, dtype=MODEL_DTYPE)
        
        # Get speech embeddings
        with torch.no_grad():
            speech_hidden = model.whisper_encoder(audio_features).last_hidden_state
            speech_embeds = model.adapter(speech_hidden)
        
        # Build input embeddings
        inputs_embeds = torch.cat([
            before_embeds.unsqueeze(0),
            speech_embeds,
            after_embeds.unsqueeze(0),
        ], dim=1)
        
        # Generate
        with torch.no_grad():
            outputs = model.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ground_truth = sample["answer"]
        
        print(f"Example {i+1}")
        print(f"-" * 40)
        print(f"Ground Truth: {ground_truth}")
        print(f"Generated:    {generated_text}")
        print(f"\n")


if __name__ == "__main__":
    main()

