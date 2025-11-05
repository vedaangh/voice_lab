import torch


def _load_template_segments(template_path):
    """Load the prompt template and split it into text before and after <speech>."""
    with open(template_path, 'r') as file:
        content = file.read()

    return content.split('<speech>')


def get_template_token_ids(template_path, tokenizer, device):
    """Return token ids for the fixed prompt segments surrounding the speech placeholder."""
    before_text, after_text = _load_template_segments(template_path)

    before_tokens = tokenizer(
        before_text,
        return_tensors='pt',
        add_special_tokens=False
    )['input_ids'][0].to(device)

    after_tokens = tokenizer(
        after_text,
        return_tensors='pt',
        add_special_tokens=False
    )['input_ids'][0].to(device)

    return before_tokens, after_tokens


def prepare_template_embeddings(template_path, tokenizer, embed_layer, device):
    """Convert the fixed prompt template segments into embeddings once per run."""
    before_tokens, after_tokens = get_template_token_ids(
        template_path=template_path,
        tokenizer=tokenizer,
        device=device
    )

    with torch.no_grad():
        before_embeds = embed_layer(before_tokens)
        after_embeds = embed_layer(after_tokens)

    return before_embeds, after_embeds, len(before_tokens), len(after_tokens)
