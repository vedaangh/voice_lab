import torch
from transformers import AutoTokenizer


def prepare_template_embeddings(template_path, tokenizer, embed_layer, device):
    """
    Load template, tokenize, and get embeddings in one step.
    
    Returns:
        before_embeds: (before_len, hidden_dim)
        after_embeds: (after_len, hidden_dim)
        before_len: int
        after_len: int
    """
    with open(template_path, 'r') as f:
        content = f.read()
    
    before_text, after_text = content.split('<speech>')
    
    before_tokens = tokenizer(
        before_text,
        return_tensors='pt',
        add_special_tokens=False
    )['input_ids'][0]
    
    after_tokens = tokenizer(
        after_text,
        return_tensors='pt',
        add_special_tokens=False
    )['input_ids'][0]
    
    before_tokens = before_tokens.to(device)
    after_tokens = after_tokens.to(device)
    
    with torch.no_grad():
        before_embeds = embed_layer(before_tokens)
        after_embeds = embed_layer(after_tokens)
    
    return before_embeds, after_embeds, len(before_tokens), len(after_tokens)
