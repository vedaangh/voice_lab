"""
Setup script for Qwen 3.5-4B model using Hugging Face transformers.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_qwen_model(model_name="Qwen/Qwen3‑4B‑Instruct‑2507"):
    """
    Load and configure Qwen 3.5-4B model for inference.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        tuple: (model, tokenizer)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    print(f"Model loaded on: {device}")
    print(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """
    Generate a response from the model given a prompt.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        str: Generated text
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


if __name__ == "__main__":
    model, tokenizer = setup_qwen_model()
    
    test_prompt = "What is the capital of France?"
    print(f"\nPrompt: {test_prompt}")
    print(f"\nResponse: {generate_response(model, tokenizer, test_prompt)}")


