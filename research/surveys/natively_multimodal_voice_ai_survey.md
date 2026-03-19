# Natively Multimodal LLMs and Voice AI Systems: Technical Survey

**Date**: March 2026
**Scope**: Architecture, integration patterns, latency, benchmarks, and conversation stability for systems that integrate audio understanding and/or generation directly into language models.

---

## 1. GPT-4o (OpenAI)

### Architecture & Audio Integration

GPT-4o is OpenAI's first truly natively multimodal model. Unlike its predecessor pipeline (Whisper ASR -> GPT-4 -> TTS), GPT-4o is a **single end-to-end neural network** that accepts and produces text, audio, and images. All modalities are tokenized and processed within the same transformer backbone; audio is processed in the model's latent space rather than being transcribed to text first. This means the model can perceive paralinguistic features—tone, emotion, pacing, emphasis, background sounds—that are lost in text transcription.

### End-to-End vs. Cascade

Fully end-to-end. The previous Voice Mode in ChatGPT used a three-model cascade: Whisper (speech-to-text) -> GPT-3.5/4 (reasoning) -> TTS (text-to-speech). GPT-4o collapses this into a single forward pass, eliminating information loss at each handoff.

### Latency & Streaming

- Average audio response latency: **~320 milliseconds**, comparable to human conversational turn-taking latency (~300ms).
- Previous cascade Voice Mode had latencies of 2.8s (GPT-3.5) to 5.4s (GPT-4).
- The Realtime API uses WebSocket connections for streaming bidirectional audio.
- Supports server-side voice activity detection (VAD) and turn management.

### Performance on Benchmarks

- Matches GPT-4 Turbo on text reasoning (English), significantly surpasses it on multilingual and code tasks.
- Sets new state-of-the-art on multilingual speech recognition benchmarks.
- Strong on M3Exam and multilingual MMLU.
- Audio perception enables emotion detection, speaker identification, and ambient sound understanding—tasks impossible for cascade systems.

### Multiturn Conversation Stability

Good stability in short-to-medium conversations. The Realtime API manages session state over WebSocket. Some reported issues with voice consistency drifting in very long conversations (>30 minutes) and occasional "hallucinated" audio artifacts when context becomes very long.

### Speech Quality / Naturalness

High naturalness. The model generates speech with appropriate prosody, emotion, pacing, and can even laugh, sing, whisper, and express surprise. Six preset voices available. Because generation is end-to-end, it can dynamically adjust speech style based on context without explicit conditioning.

### Known Issues

- Audio output can sometimes be overly dramatic or emotional.
- Long conversation sessions can show latency creep.
- Safety filtering can sometimes produce abrupt tone changes.
- The model architecture is entirely proprietary; no published paper with architectural details.

---

## 2. Gemini (Google DeepMind)

### Architecture & Audio Integration

Gemini (1.0, 1.5, 2.0, 2.5 family) is a natively multimodal model that processes text, images, video, and audio within a **single transformer decoder**. Audio is tokenized at a rate of **32 tokens per second** (1 minute of audio = 1,920 tokens). The input audio is downsampled to 16 Kbps and multi-channel audio is mixed to mono.

Gemini 1.0 used audio features derived from a Universal Speech Model (USM) encoder. Gemini 1.5 Pro is a **sparse Mixture-of-Experts (MoE)** architecture that can process up to **10 million tokens** of context, enabling ingestion of hours of audio in a single prompt (up to 9.5 hours). Gemini 1.5 achieves >99% retrieval accuracy across its long context window, including for audio content.

### End-to-End vs. Cascade

For audio *understanding* (ASR, audio QA, emotion detection), Gemini is fully end-to-end—audio tokens are processed natively by the same model.

For audio *generation*, Gemini 2.5 Flash TTS and Pro TTS models provide dedicated text-to-speech. The **Live API** (introduced with Gemini 2.0) enables real-time bidirectional audio streaming with native audio output, barge-in support, and affective dialog adaptation. This Live API processes raw 16-bit PCM audio at 16kHz input / 24kHz output via WebSocket.

### Latency & Streaming

- The Live API is designed for "sub-second native audio streaming" (Gemini 2.5 Flash Live).
- Standard audio API does NOT support real-time transcription—the Live API is the dedicated path for real-time voice.
- Supports 70+ languages for multilingual voice.
- Ephemeral tokens for production deployment security.

### Performance on Benchmarks

- Gemini 1.0 Ultra: SOTA on 30 of 32 benchmarks at launch, including first model to exceed human-expert performance on MMLU.
- Gemini 1.5 Pro: SOTA on long-context ASR, long-document QA, long-video QA.
- Demonstrated ability to learn to translate to Kalamang (an extremely low-resource language) from grammar manuals provided in-context.
- Specific ASR WER numbers not publicly released in detail, but competitive with or exceeding Whisper on many languages.

### Multiturn Conversation Stability

The Live API supports stateful WebSocket sessions with dedicated session management. The massive context window (up to 10M tokens for 1.5 Pro) provides strong multiturn stability. Proactive audio features allow the model to manage response timing naturally.

### Speech Quality / Naturalness

Gemini 2.5 TTS models are described as "high-fidelity speech synthesis." The Live API supports "affective dialog" that adapts response style and tone to match user expression. However, compared to GPT-4o's Advanced Voice Mode, early user reports suggest Gemini's voice output can sound slightly more synthetic in extended conversations.

### Known Issues

- Standard Gemini audio API explicitly does NOT support real-time streaming—only the Live API does.
- Limited public documentation on exact audio architecture internals.
- Audio generation quality varies by language.

---

## 3. Qwen-Audio / Qwen2-Audio (Alibaba)

### Architecture & Audio Integration

Qwen2-Audio uses a modular architecture: an **audio encoder** (based on Whisper-large-v3 architecture) connected to the **Qwen-7B language model** as the backbone LLM. The audio encoder extracts continuous feature representations from raw audio, which are projected into the LLM's embedding space through a learned projection layer.

The model operates in two modes:

- **Voice Chat Mode**: Audio-only input where the user's instructions are embedded within the speech. No separate ASR module needed.
- **Audio Analysis Mode**: Paired audio + text inputs, where the user provides explicit text instructions alongside audio for analysis tasks.

### End-to-End vs. Cascade

End-to-end for audio *understanding*—audio goes directly into the model without ASR intermediary. However, Qwen2-Audio is an **understanding-only** model; it produces text output, not speech. Speech generation requires a separate TTS system.

### Training Pipeline

Three-stage training:

1. **Multi-task pretraining**: Audio-language alignment across diverse tasks.
2. **Supervised finetuning**: Task-specific instruction tuning.
3. **Direct Preference Optimization (DPO)**: Aligning outputs with human preferences.

### Performance on Benchmarks

Qwen2-Audio-7B-Instruct significantly surpasses both prior SOTA and the original Qwen-Audio across:

- **LibriSpeech** (English ASR)
- **Common Voice 15** (multilingual ASR)
- **FLEURS** (multilingual speech)
- **Aishell2** (Chinese ASR)
- **CoVoST2** (speech translation)
- **MELD** (emotion recognition)
- **VocalSound** (sound classification)
- **AIR-Benchmark** (audio information retrieval)

Supports 8+ languages for transcription and translation.

### Multiturn Conversation

Explicitly supports multi-turn conversational reasoning with audio context. The model can maintain context across turns when discussing audio content, though specific stability metrics for very long conversations are not published.

### Speech Quality / Naturalness

N/A for output (text-only output). For input understanding, robust to mixed audio containing overlapping speech and music, environmental noise, and diverse acoustic conditions.

### Known Issues

- Understanding-only model; requires external TTS for voice responses.
- Performance on very long audio clips (>30 seconds) may degrade since the encoder has finite context.
- Musical analysis capabilities (key, tempo detection) are approximate.

---

## 4. Qwen2.5-Omni (Alibaba)

### Architecture & Audio Integration

Qwen2.5-Omni (7B parameters) introduces the **Thinker-Talker architecture**:

- **Thinker**: A full Transformer decoder LLM with attached encoders for audio and images/video. Processes all multimodal inputs (text, images, audio, video) and generates text token sequences. Acts as the "reasoning brain."
- **Talker**: A **dual-track autoregressive Transformer decoder** that takes the Thinker's hidden representations and generates speech audio tokens in streaming fashion. The Talker does NOT independently reason—it converts the Thinker's plans into speech.

**TMRoPE (Time-aligned Multimodal RoPE)**: A novel positional embedding that synchronizes video frame timestamps with audio timestamps in the interleaved input sequence, ensuring temporal coherence across modalities.

**Audio Codec**: Uses a **Sliding-window DiT** (Diffusion Transformer) for audio decoding, which restricts the receptive field to reduce initial package delay for streaming output.

Block-wise processing of audio and video inputs enables **streaming perception** of multimedia content.

### End-to-End vs. Cascade

Fully end-to-end for both understanding and generation. The model accepts audio/video/image/text and produces both text AND speech output in a single architecture. The Thinker and Talker are jointly trained.

### Latency & Streaming

- Designed for real-time streaming: "chunked input and immediate output."
- Sliding-window DiT specifically designed to minimize first-token latency for speech output.
- Exact latency numbers not published, but the architecture is explicitly optimized for streaming interaction.

### Performance on Benchmarks

- **Omni-Bench**: State-of-the-art results for omni-modal understanding.
- **Speech instruction following**: Performance on MMLU and GSM8K via speech input **matches text input performance**—a critical milestone showing minimal degradation from audio interface.
- Outperforms Qwen2-Audio on all audio understanding tasks.
- Competitive with Qwen2.5-VL (similar size) on vision tasks.
- Strong on Common Voice (ASR), CoVoST2 (translation), MMAU (audio understanding), MMMU/MMStar (image reasoning), MVBench (video understanding).
- Streaming Talker "outperforms most existing streaming and non-streaming alternatives in robustness and naturalness."

### Multiturn Conversation Stability

The Thinker maintains full LLM context, so multiturn stability is inherited from the base Qwen2.5 architecture. The Talker generates speech conditioned on the Thinker's hidden states per-turn, avoiding accumulation of audio generation errors across turns.

### Speech Quality / Naturalness

The Talker produces natural speech with claims of state-of-the-art robustness and naturalness among streaming systems. The dual-track design means text and audio are generated concurrently without one blocking the other.

### Known Issues

- 7B parameter model—smaller than GPT-4o, so reasoning depth is more limited.
- Sliding-window DiT trades off some global coherence for streaming latency.
- Publicly available weights, but not yet widely stress-tested in production.

---

## 5. CSM / Conversational Speech Model (Sesame AI)

### Architecture & Audio Integration

CSM uses a **dual autoregressive transformer** design:

1. **Backbone Transformer** (Llama-3.2-1B): Processes interleaved text and audio tokens from the conversation history. It predicts the **zeroth codebook level** of the RVQ (Residual Vector Quantization) representation—the coarsest, most semantic layer.
2. **Decoder Transformer** (smaller): Takes the backbone's output and autoregressively generates the remaining RVQ codebook levels (acoustic detail levels) to produce full-fidelity audio tokens.

**Mimi Codec**: Audio is tokenized using the Mimi codec (from Kyutai), which operates at **12.5 Hz producing 32 discrete codes per 80 milliseconds**. RVQ decomposes audio into hierarchical levels: semantic tokens capture linguistic content while acoustic tokens preserve speaker identity, prosody, and naturalness.

**Compute Amortization**: The decoder is trained on only 1/16th of audio frames while maintaining full zeroth-codebook training, reducing memory without perceptible quality loss.

### End-to-End vs. Cascade

CSM is a **speech generation model only**—it is NOT a multimodal LLM. It cannot understand or reason about audio. In practice, it sits downstream of a text LLM: the LLM generates text responses, and CSM converts them to natural conversational speech with appropriate prosody. So the full pipeline is cascade (ASR -> LLM -> CSM), but CSM itself is end-to-end from text+context to speech.

### Latency & Streaming

- On RTX 4090: ~2.1x realtime factor (bfloat16 compiled), meaning it generates speech faster than realtime.
- ~4.4GB VRAM in bfloat16/float16.
- Supports CUDA graph compilation via `torch.compile()` with static cache for consistent performance.
- Supports batched inference for multiple prompts.

### Speech Quality / Naturalness

This is CSM's primary differentiator. Sesame's research focus is "crossing the uncanny valley of voice." Key quality features:

- **Context-dependent prosody**: CSM conditions on conversation history (prior text + audio turns), so it generates speech with prosody appropriate to the conversational moment—not just the text content.
- **One-to-many problem**: The model addresses the fact that identical text can be spoken many valid ways; context determines which is appropriate.
- **Emotional responsiveness**: Adjusts tone based on the user's emotional state from prior turns.
- **Natural timing**: Produces appropriate pauses, hesitations, and rhythm.

Traditional metrics (WER, speaker similarity) have "saturated"—Sesame introduced novel phonetic benchmarks including homograph disambiguation and pronunciation consistency tests. Subjective studies show generated speech matches human naturalness even without explicit contextual cues.

### Multiturn Conversation Stability

Explicitly designed for conversational context. The model takes a structured conversation history with speaker IDs, text, and audio for each prior turn. Quality improves with more context ("CSM sounds best when provided with context"). This design inherently supports multiturn stability—each generation is grounded in the full conversation history.

### Known Issues

- **Not a multimodal LLM**: Cannot understand audio, generate text, or reason. Requires a separate LLM.
- **English only**: Limited non-English capacity due to training data contamination only.
- **Voice control**: Base model generates varied voices; specific voice consistency requires fine-tuning.
- **1B parameters**: Relatively small; larger models may improve quality further.

---

## 6. Hume EVI (Empathic Voice Interface)

### Architecture & Audio Integration

Hume EVI is a **commercially deployed speech-to-speech AI system** with emotional intelligence. Key architectural components:

- **Expression Measurement Model**: Analyzes voice (and optionally facial expressions) to detect **600+ emotional and vocal characteristic tags** in real-time. This is Hume's core differentiator—a proprietary model trained on large-scale human expression data.
- **Language Model**: EVI wraps an LLM (configurable—users can bring their own) that receives both the text transcription AND the expression measurement vector as input, enabling emotionally-aware reasoning.
- **Voice Synthesis**: Generates speech output with expressive prosody conditioned on the emotional context of the conversation.

EVI is a **cascade system** that integrates multiple specialized models (expression measurement, ASR, LLM, TTS) rather than a single end-to-end model, but the expression measurement signal flows through the entire pipeline, differentiating it from naive cascades.

### End-to-End vs. Cascade

Cascade architecture, but with emotion as a through-line signal. The expression measurement runs in parallel with ASR, and both feed into the LLM, which then drives emotionally-conditioned TTS.

### Latency & Streaming

- **~250ms speech LLM latency** (as of EVI 2/3).
- WebSocket-based streaming API.
- SDKs in TypeScript, Python, .NET, Swift.
- Integrations with Twilio, LiveKit, Agora, VAPI for production telephony.

### Performance on Benchmarks

- Ranked **#1 in naturalness and expressivity** in comparative evaluations (per Hume's claims).
- EVI 3 (latest) emphasizes instruction-following capabilities and personalization.
- No standard academic benchmarks published (proprietary system).

### Multiturn Conversation Stability

The emotion measurement model provides continuous signal across turns, enabling the system to track emotional arcs over a conversation. The LLM receives this signal each turn, so the system can respond to emotional escalation, calming, etc. Conversation stability is generally good for the intended use cases (coaching, companionship, customer service).

### Speech Quality / Naturalness

High expressivity—the system generates speech with dynamic emotional range. Can adjust warmth, concern, excitement, and calmness in voice. The "fully personalized" mode allows developers to shape the voice personality.

### Known Issues

- Proprietary and closed-source; no published papers on internal architecture.
- Cascade latency is additive across stages, though 250ms total is competitive.
- Emotion detection can misclassify in noisy environments.
- Expression tags (600+) may be overly granular for some applications.
- Cost: commercial API pricing.

---

## 7. Sesame / Dia (Nari Labs)

### Architecture & Audio Integration

Dia is a **1.6B parameter text-to-speech model** inspired by SoundStorm, Parakeet, and the Descript Audio Codec (DAC). It generates highly realistic **multi-speaker dialogue** from text transcripts in a single pass.

Key features:

- **Multi-speaker synthesis**: Uses `[S1]` and `[S2]` speaker tags to generate distinct voices in dialogue.
- **DAC codec**: Uses the Descript Audio Codec for audio tokenization/detokenization.
- **Nonverbal communication**: Supports ~20 nonverbal tags: `(laughs)`, `(sighs)`, `(clears throat)`, `(gasps)`, `(coughs)`, `(singing)`, `(mumbles)`, `(groans)`, `(sniffs)`, `(claps)`, `(screams)`, `(inhales)`, `(exhales)`, `(applause)`, `(burps)`, `(humming)`, `(sneezes)`, `(chuckle)`, `(whistles)`.
- **Voice cloning**: 5-10 second audio prompts enable speaker consistency through voice cloning.

Note: Dia is from **Nari Labs**, which is a separate entity from Sesame AI Labs (which makes CSM). They are sometimes confused but are different organizations.

### End-to-End vs. Cascade

Dia is an end-to-end text-to-speech model for dialogue. It does not perform audio understanding or text reasoning. Like CSM, it would sit downstream of an LLM in a full conversational AI pipeline.

### Latency & Streaming

On RTX 4090:


| Precision | Realtime Factor (compiled) | VRAM   |
| --------- | -------------------------- | ------ |
| bfloat16  | 2.1x realtime              | ~4.4GB |
| float16   | 2.2x realtime              | ~4.4GB |
| float32   | 1.0x realtime              | ~7.9GB |


Initial run includes DAC download overhead.

### Speech Quality / Naturalness

Strong for dialogue generation. The nonverbal communication tags enable natural-sounding conversations with laughter, sighs, and other paralinguistic elements. Voice cloning maintains speaker identity across an entire dialogue.

### Known Issues

- Overuse of nonverbal tags can produce artifacts.
- English only.
- Not an understanding model—purely generative TTS.
- Voice cloning quality depends on prompt quality (5-10 seconds optimal).

---

## 8. SALMONN (Tsinghua / ByteDance)

### Architecture & Audio Integration

SALMONN (Speech Audio Language Music Open Neural Network) uses a distinctive **dual encoder** architecture:

1. **Whisper-Large-v2 Encoder** (speech encoder): Trained for ASR/translation; output features capture speech content and some background noise information. Operates at 50Hz frame rate.
2. **BEATs Encoder** (audio encoder): Self-supervised model trained for non-speech audio semantics via iterative tokenize-mask-predict training. Also 50Hz frame rate.

The two encoder outputs are **concatenated frame-by-frame** along the feature dimension (Eq. 1 in the paper), creating a dual representation that captures both speech and general audio.

**Window-level Q-Former**: A modified Q-Former that operates on **L=17 frame windows (~0.33 seconds each)** rather than the full sequence. Uses **N=1 trainable query per window**, producing a variable number of textual tokens proportional to audio length (88 tokens for 30 seconds of audio). This preserves temporal resolution and monotonic alignment—critical for speech recognition.

**LLM Backbone**: Vicuna-13B (frozen) with **LoRA adapters** (rank 8, scaling factor 4.0) on query and value matrices in self-attention layers. Only Q-Former + LoRA are trained (~33M parameters, 0.24% of total model).

### Training Method (3 Stages)

1. **Pre-training**: Large-scale ASR (LibriSpeech 960h + GigaSpeech 220h) + audio captioning (WavCaps 2800h + AudioCaps + Clotho) to learn audio-text alignment.
2. **Instruction Tuning**: 12 supervised tasks across ~4400 hours, ~2.3M samples: ASR, translation (En2Zh), audio captioning, phone recognition, emotion recognition, music captioning, overlapped speech recognition, speaker verification, gender recognition, speech/audio/music QA.
3. **Activation Tuning** (novel): A few-shot self-supervised stage using just 12 story samples (12 training steps) to overcome "task overfitting"—where the model becomes biased toward ASR/captioning and loses emergent cross-modal abilities. This works by regularizing the intrinsic conditional language model P(Y|X) so it doesn't collapse to only ASR-like outputs.

### End-to-End vs. Cascade

End-to-end for audio understanding. Audio goes directly through dual encoders -> Q-Former -> LLM, with no intermediate ASR step. Text-only output (no speech generation).

### Performance on Benchmarks (from paper Table 3)

**Level 1 (trained tasks, with activation tuning)**:

- ASR: 2.1% / 4.9% / 10.0% WER (LibriSpeech clean/other, GigaSpeech) — near Whisper-Large-v2 quality
- En2Zh translation: 33.1 BLEU4
- Audio captioning: 24.0 METEOR, 40.3 SPIDEr
- Phone recognition: 4.2% PER
- Emotion recognition: 0.69 accuracy
- Music captioning: 5.5 BLEU4, 21.8 RougeL
- Speaker verification: 0.94 accuracy

**Level 2 (untrained tasks)**:

- En2De: 18.6 BLEU4; En2Ja: 22.7 BLEU4 (zero-shot translation)
- Keyword extraction: 0.32 accuracy
- Spoken QA: 0.41 accuracy (98% following rate after activation tuning)
- Slot filling: 0.41 accuracy (99% following rate)

**Level 3 (novel tasks)**:

- Audio storytelling: 82.57 diversity score (100% FR after activation tuning; was 0% before)
- Speech-audio co-reasoning: 0.50 accuracy (73% FR; was 4% before)

### Multiturn Conversation Stability

Not explicitly designed for multiturn dialogue. The model processes one audio input + one text instruction at a time. Multi-turn would require external conversation management. The "task overfitting" issue (where the model ignores instructions and defaults to ASR) is a form of instability that activation tuning resolves.

### Speech Quality / Naturalness

N/A (text-only output). For input, handles general audio including mixed speech + audio events + music.

### Known Issues

- Task overfitting: Without activation tuning, the model ignores complex instructions and defaults to transcription. Activation tuning is critical.
- Understanding-only model; no speech generation.
- 0.33-second Q-Former windows may miss very fast acoustic events.
- Repeat rate: QA-based activation tuning leads to 4.6% repetition rate vs. 0.1% for story-based.

---

## 9. Pengi / AudioPaLM

### 9a. Pengi (Microsoft, NeurIPS 2023)

**Architecture**: Pengi frames ALL audio tasks as text-generation problems. Three components:

- **Audio Encoder**: HTSAT transformer backbone from CLAP. Processes audio at 44.1kHz, converted to log Mel spectrograms (64 bins), truncated to 7 seconds. NOT frozen—fine-tuned during training.
- **Text Encoder**: Frozen CLIP-based encoder for text prompts.
- **Mapping Networks**: Two trainable networks (m1, m2) convert single embeddings into sequences of k embeddings each, producing 80 prefix tokens total (40 audio + 40 text).
- **Language Model**: Frozen GPT-2 base (124M parameters), autoregressively generates output conditioned on the audio-text prefix.

**Training**: Single captioning objective on 3.4M audio-text pairs across 8 task templates. Cross-entropy loss, Adam optimizer, 60 epochs, batch 384, 20 V100 GPUs.

**Benchmark Results (22 tasks)**:

- AudioCaps captioning: 0.4667 SPIDEr (+6.6% over prior SOTA)
- Clotho captioning: 0.2709 SPIDEr (+26% over prior SOTA)
- ClothoAQA: 64.5% accuracy
- ESC50 sound classification: 92% (exceeds 81% human baseline)
- FSD50K: 0.4676 mAP
- UrbanSound8K: 71.85% accuracy

**Limitations**: Frozen GPT-2 base is very small (124M). 7-second audio truncation limits long-form understanding. Text-to-audio retrieval significantly underperforms contrastive models (CLAP).

### 9b. AudioPaLM (Google, 2023)

**Architecture**: Fuses PaLM-2 (8B) with AudioLM into a unified model with a **shared text-audio vocabulary**.

- **Initialization**: Starts from pretrained PaLM-2 8B checkpoint. Embedding matrix expanded from t to (t+a) tokens, where a = audio vocabulary size. New audio embeddings initialized to zero; text embeddings retain pretrained values.
- **Audio Tokenization**: Self-supervised speech models (w2v-BERT multilingual or USM) extract semantic tokens at **25Hz with vocabulary size 1024**. Uses AudioLM's hierarchical tokenization (semantic -> coarse acoustic -> fine acoustic).
- **Unified Vocabulary**: Text (SentencePiece) and audio tokens share the same sequence space. Mixed text-audio sequences can be input/output.
- **Task Specification**: Textual prefixes like `[ASR French]` or `[S2ST English French]` specify the task. Can chain tasks: output English text, then French text, then French audio tokens.
- **Audio Decoding**: Two methods—AudioLM stages 2-3 (autoregressive) or **SoundStorm** (non-autoregressive, two orders of magnitude faster, +1.3 BLEU on S2ST).

**Training**: Joint training on interleaved speech-text tasks: ASR, AST, S2ST, TTS, MT. Data includes CoVoST2/CVSS, VoxPopuli, CommonVoice, YouTube ASR, WMT/TED, plus synthetic data from PaLM-2 translations.

**Critical Finding**: Fine-tuning from PaLM-2 checkpoint achieves 18.4 BLEU vs. 6.9 BLEU training from scratch on CoVoST2 AST—confirming that text pretraining massively benefits speech tasks.

**Benchmark Results**:

- CoVoST2 AST: **37.8 BLEU** (vs. Whisper's 29.1)
- CVSS S2ST: **32.5 ASR-BLEU** with SoundStorm
- VoxPopuli ASR: **9.8 WER** (vs. Whisper's 13.6)
- Zero-shot AST (FLEURS): 28.6 BLEU observed pairs, 20.7 BLEU for ASR-only languages
- Voice transfer: 4.0 similarity MOS (vs. Translatotron 2's 3.51)
- Adding ASR to AST training improved BLEU from 16.0 to 18.5 (multi-task benefit)

**Voice Preservation**: 3-second voice samples (audio tokens + SoundStream tokens) condition the model to preserve speaker identity across languages.

**Limitations**: No public model release. 8B parameters max. Audio generation quality dependent on SoundStorm/AudioLM decoding. Not designed for real-time conversation.

---

## 10. Thinker-Talker Architecture Pattern

### Pattern Description

The Thinker-Talker pattern separates the **reasoning/understanding** component (Thinker) from the **speech generation** component (Talker) within a single trained system. This is distinct from cascade systems because the Talker has access to the Thinker's internal hidden representations, not just its text output.

### Instances of the Pattern

#### Qwen2.5-Omni

- **Thinker**: Full Transformer decoder LLM with audio/image encoders. Generates text tokens and hidden states.
- **Talker**: Dual-track autoregressive Transformer decoder. Receives hidden representations from the Thinker (not just text) and generates speech audio tokens via Sliding-window DiT.
- **Key property**: Text and speech are generated **concurrently** from the same reasoning pass.

#### SALMONN-Omni (ByteDance, extension of SALMONN)

- Extends the original SALMONN for **full-duplex** speech interaction (simultaneous listening and speaking).
- Uses a thinking module for reasoning about audio input and a talking module for generating speech output.
- Designed for streaming interaction, though detailed architecture documentation is limited compared to Qwen2.5-Omni.

### Advantages of Thinker-Talker over Cascade

1. **Richer information flow**: The Talker receives full hidden states, not just discrete text tokens. This preserves information about emphasis, uncertainty, emotion that the Thinker inferred but might not express in text.
2. **Concurrent generation**: Text and speech can stream simultaneously, reducing perceived latency.
3. **Joint training**: End-to-end training optimizes both components together, avoiding mismatch between independently trained ASR/LLM/TTS.
4. **Streaming-native**: The architecture naturally supports chunked processing and incremental output.

### Advantages over Fully-Unified Models (like GPT-4o)

1. **Modularity**: Can upgrade the Thinker (better reasoning) or Talker (better speech) independently.
2. **Interpretability**: Can inspect intermediate text output from the Thinker.
3. **Efficiency**: The Talker can be much smaller than the Thinker since it doesn't need to reason.

### Disadvantages

1. **Added complexity**: Two transformer stacks to train and serve.
2. **Latency overhead**: Information must flow from Thinker to Talker (though this can be pipelined).
3. **Potential bottleneck**: If the Thinker's hidden representations don't capture all speech-relevant information, the Talker is limited.

---

## Comparative Summary Table


| System          | Type                      | Audio In           | Audio Out                | End-to-End         | Latency       | Open Weights  |
| --------------- | ------------------------- | ------------------ | ------------------------ | ------------------ | ------------- | ------------- |
| GPT-4o          | Unified native multimodal | Yes (native)       | Yes (native)             | Yes                | ~320ms        | No            |
| Gemini          | Unified native multimodal | Yes (native)       | Yes (Live API)           | Yes                | Sub-second    | No            |
| Qwen2-Audio     | Audio understanding LLM   | Yes (encoder)      | No                       | Understanding only | N/A           | Yes (7B)      |
| Qwen2.5-Omni    | Thinker-Talker omni       | Yes (native)       | Yes (Talker)             | Yes                | Streaming     | Yes (7B)      |
| CSM (Sesame)    | Speech generation         | Context audio      | Yes (RVQ)                | Generation only    | >2x RT        | Yes (1B)      |
| Hume EVI        | Emotionally-aware cascade | Yes (expression)   | Yes (expressive TTS)     | No (cascade)       | ~250ms        | No            |
| Dia (Nari Labs) | Dialogue TTS              | Prompt audio       | Yes (DAC)                | Generation only    | >2x RT        | Yes (1.6B)    |
| SALMONN         | Audio understanding LLM   | Yes (dual encoder) | No                       | Understanding only | N/A           | Yes (13B)     |
| Pengi           | Audio-language model      | Yes (HTSAT)        | No                       | Understanding only | N/A           | Yes (124M LM) |
| AudioPaLM       | Unified audio-text LLM    | Yes (USM/w2v-BERT) | Yes (AudioLM/SoundStorm) | Yes                | Not real-time | No            |


---

## Key Architectural Trends

1. **Native multimodality is winning**: GPT-4o and Gemini show that training a single model on all modalities from scratch outperforms cascade pipelines on quality, latency, and emergent capabilities.
2. **Thinker-Talker as the open-source path**: Since training a GPT-4o-scale unified model requires enormous resources, the Thinker-Talker pattern (Qwen2.5-Omni) offers a practical decomposition that preserves most benefits of end-to-end training while enabling modular development.
3. **Audio tokenization is critical**: All systems rely on converting audio to discrete tokens—whether via learned codecs (Mimi at 12.5Hz, DAC), self-supervised models (w2v-BERT at 25Hz), or model-internal tokenization (Gemini at 32 tokens/sec). The codec choice determines the quality/efficiency tradeoff.
4. **Context is everything for speech quality**: Both CSM and Dia demonstrate that conditioning on conversation history dramatically improves prosody naturalness—a finding that applies across all voice AI systems.
5. **Long-conversation degradation remains unsolved**: Every system shows some form of quality degradation in very long conversations, whether through context window limitations, attention dilution, or generation drift. This is an active research frontier.

