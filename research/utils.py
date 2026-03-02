"""
Utility functions for preparing batches and embeddings.
"""

import torch


def prepare_template_embeddings(template_path, tokenizer, embed_layer, device):
    """Load prompt template, tokenize, and convert to embeddings."""
    with open(template_path, "r") as file:
        content = file.read()

    before_text, after_text = content.split("<speech>")

    before_tokens, after_tokens = [
        tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
        for text in [before_text, after_text]
    ]

    with torch.no_grad():
        before_embeds, after_embeds = [
            embed_layer(tokens) for tokens in [before_tokens, after_tokens]
        ]

    return before_embeds, after_embeds, len(before_tokens), len(after_tokens)


def encode_speech(model, audio_features, dtype):
    """Process audio through Whisper encoder and adapter."""
    with torch.no_grad():
        speech_hidden = model.whisper_encoder(audio_features.to(dtype=dtype)).last_hidden_state
    return model.adapter(speech_hidden)


def prepare_response_embeds(batch, tokenizer, embed_layer, device, dtype):
    """Pad, embed response tokens, return embeddings, ids, and mask."""
    response_ids = torch.nn.utils.rnn.pad_sequence(
        batch["answer_input_ids"], batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(device)
    response_mask = (response_ids != tokenizer.pad_token_id).long()
    response_embeds = embed_layer(response_ids).to(dtype=dtype)
    return response_embeds, response_ids, response_mask


def prepare_batch(
    batch,
    model,
    tokenizer,
    before_embeds,
    after_embeds,
    before_len,
    after_len,
    device,
    dtype,
):
    """
    Prepare batch: process audio, tokenize text, combine embeddings.

    Returns:
        inputs_embeds: (batch, total_seq_len, hidden_dim)
        labels: (batch, total_seq_len) - -100 for prompt/speech, token IDs for response
        attention_mask: (batch, total_seq_len) - 1s for real tokens, 0s for padding
    """
    batch_size = len(batch["answer_input_ids"])
    audio_features = batch["input_features"].to(device=device)

    speech_embeds = encode_speech(model, audio_features, dtype)
    speech_len = speech_embeds.shape[1]

    embed_layer = model.llm.get_input_embeddings()
    response_embeds, response_ids, response_mask = prepare_response_embeds(
        batch, tokenizer, embed_layer, device, dtype
    )

    before_embeds_batch, after_embeds_batch = [
        embed.unsqueeze(0).expand(batch_size, -1, -1).to(dtype=dtype)
        for embed in [before_embeds, after_embeds]
    ]

    inputs_embeds = torch.cat(
        [before_embeds_batch, speech_embeds, after_embeds_batch, response_embeds], dim=1
    )

    prompt_len = before_len + speech_len + after_len
    prompt_labels = torch.full((batch_size, prompt_len), -100, dtype=torch.long, device=device)
    response_labels = response_ids.clone()
    response_labels[response_mask == 0] = -100

    labels = torch.cat([prompt_labels, response_labels], dim=1)

    masks = [
        torch.ones(batch_size, length, dtype=torch.long, device=device)
        for length in [before_len, speech_len, after_len]
    ]
    masks.append(response_mask)

    attention_mask = torch.cat(masks, dim=1)

    return {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "attention_mask": attention_mask,
        "prompt_len": prompt_len,
    }
