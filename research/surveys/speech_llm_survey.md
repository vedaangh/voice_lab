# End-to-End Speech LLMs: Comprehensive Survey

*Compiled March 2026*

---

## Table of Contents

1. [SpeechGPT](#1-speechgpt)
2. [SpeechGPT-Gen](#2-speechgpt-gen)
3. [LLaMA-Omni](#3-llama-omni)
4. [LLaMA-Omni 2](#4-llama-omni-2)
5. [Mini-Omni](#5-mini-omni)
6. [Mini-Omni 2](#6-mini-omni-2)
7. [VITA](#7-vita)
8. [Freeze-Omni](#8-freeze-omni)
9. [GLM-4-Voice](#9-glm-4-voice)
10. [Ichigo](#10-ichigo)
11. [IntrinsicVoice](#11-intrinsicvoice)
12. [OmniFlatten](#12-omniflatten)
13. [SLAM-Omni](#13-slam-omni)
14. [Westlake-Omni](#14-westlake-omni)
15. [Cross-Model Comparison Tables](#15-cross-model-comparison-tables)
16. [VoiceBench Results](#16-voicebench-results)

---

## 1. SpeechGPT

| Field | Details |
|-------|---------|
| **Full Title** | SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities |
| **Authors** | Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, Xipeng Qiu |
| **Date** | May 18, 2023 |
| **ArXiv ID** | 2305.11000 |
| **Venue** | Work in progress (Fudan University) |

### Architecture

- **Speech Input Encoding:** Discrete speech units via HuBERT (Hidden-Unit BERT) with k-means clustering. Speech waveforms are converted into a sequence of discrete tokens from a learned codebook. These discrete units are added to the LLM vocabulary as new tokens.
- **LLM Backbone:** LLaMA-based (7B/13B parameters). The LLM vocabulary is expanded with discrete speech unit tokens so the model can process both text and speech in a unified token space.
- **Adapter/Connection:** Direct vocabulary expansion -- speech tokens are embedded alongside text tokens in a shared embedding space (early fusion via discrete tokenization). No separate adapter module.
- **Speech Output Generation:** The LLM autoregressively generates discrete speech unit tokens, which are then converted to waveforms via a unit-based HiFi-GAN vocoder.

### Training Strategy (Three Stages)

1. **Stage 1 -- Modality-Adaptation Pre-training:** The LLM is pre-trained on large-scale speech data (unpaired) to learn speech discrete unit distributions, expanding its vocabulary to include speech tokens.
2. **Stage 2 -- Cross-Modal Instruction Fine-tuning:** Trained on the SpeechInstruct dataset with paired cross-modal (speech<->text) instruction-following tasks.
3. **Stage 3 -- Chain-of-Modality Instruction Fine-tuning:** A novel training paradigm where the model learns to first generate a text (chain-of-thought-like) intermediate response, then produce the speech output. This leverages the LLM's strong text reasoning to guide speech generation.

### Training Data

- **SpeechInstruct:** Large-scale cross-modal speech instruction dataset constructed by the authors, containing speech-text paired instruction-following examples.

### Key Contributions

- First multimodal LLM capable of perceiving and generating speech through intrinsic cross-modal conversational abilities (not a cascade pipeline).
- Chain-of-Modality instruction fine-tuning: generates text internally before producing speech, improving response quality.
- Demonstrated that discrete speech representations can unify speech and text modalities within a single LLM.

### Evaluation

- Demonstrated cross-modal conversational abilities on spoken QA, speech continuation, and instruction following tasks.
- Qualitative demos rather than large-scale quantitative benchmarks in the initial release.

### Streaming / Real-time

- **No.** SpeechGPT generates the full text chain-of-modality response before producing speech, introducing significant latency (reported ~1+ second).

### Multi-turn Conversation

- Supported in principle through the shared discrete token space, but limited by context length constraints.

---

## 2. SpeechGPT-Gen

| Field | Details |
|-------|---------|
| **Full Title** | SpeechGPT-Gen: Scaling Chain-of-Information Speech Generation |
| **Authors** | Dong Zhang, Xin Zhang, Jun Zhan, Shimin Li, Yaqian Zhou, Xipeng Qiu |
| **Date** | January 24, 2024 |
| **ArXiv ID** | 2401.13527 |
| **Venue** | Fudan University |

### Architecture

- **Speech Tokenizer:** SpeechTokenizer with Residual Vector Quantization (RVQ), 8 hierarchical codebooks:
  - Layer 1 (q1): Semantic tokens (discrete) -- captures linguistic/semantic content
  - Layers 2-7 (v2:7): Perceptual information (continuous vectors, summed) -- captures speaker timbre, prosody, acoustic details
  - Combined (v1:8) reconstructs complete speech via decoder
- **LLM Backbone:** LLaMA2-7B-Chat (8B total parameters with extensions)
- **Speech Output Generation:** Two-stage Chain-of-Information Generation (CoIG):
  - **Stage 1 (Semantic):** Autoregressive LLM generates RVQ-1 semantic tokens
  - **Stage 2 (Perceptual):** Non-autoregressive flow-matching model generates perceptual layers (v2:8)
  - Two variants: *Explicit Chain* (standard Gaussian prior) and *Implicit Chain* (semantic-injected prior, mu=v1, which is more efficient)

### Key Contributions

- Chain-of-Information Generation (CoIG): Decouples semantic and perceptual information for more efficient and higher-quality speech synthesis.
- Demonstrates that infusing semantic information into the flow-matching prior distribution significantly improves generation quality.
- Scales to 8B parameters for speech generation.

### Training Data

- Semantic LLM: Multilingual LibriSpeech, GigaSpeech, CommonVoice, LibriSpeech
- Flow matching: Multilingual LibriSpeech
- Training: 77,000 steps (semantic), 1,000 steps (perceptual) on A100 GPUs

### Evaluation Results

| Task | WER | Speaker Similarity | Quality MOS | Similarity MOS |
|------|-----|-------------------|-------------|----------------|
| Zero-shot TTS (LibriSpeech test-clean) | 2.4% | 0.66 | 3.69 | 3.51 |
| Voice Conversion (VCTK, 109 speakers) | 3.1% | 0.86 | 3.72 | 3.54 |
| Speech-to-Speech Dialogue | -- | 0.87 | -- | -- |
| S2S Dialogue ChatGPT Score | 3.61 | -- | -- | -- |

### Streaming / Real-time

- **No.** The two-stage generation process (semantic then perceptual) is not designed for streaming.

### Multi-turn Conversation

- Supported through the LLM backbone's conversational capabilities.

---

## 3. LLaMA-Omni

| Field | Details |
|-------|---------|
| **Full Title** | LLaMA-Omni: Seamless Speech Interaction with Large Language Models |
| **Authors** | Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, Yang Feng |
| **Date** | September 10, 2024 |
| **ArXiv ID** | 2409.06666 |
| **Venue** | ICLR 2025 |

### Architecture

- **Speech Input Encoding:** Whisper-large-v3 encoder (frozen throughout training). Extracts continuous speech representations from raw audio.
- **Speech Adaptor:** Downsampling by factor k=5 (concatenating consecutive frames) followed by a 2-layer MLP perceptron: `Linear(ReLU(Linear(DownSample(H))))`. Trainable.
- **LLM Backbone:** Llama-3.1-8B-Instruct
- **Speech Output Generation:** Non-autoregressive streaming speech decoder using Connectionist Temporal Classification (CTC).
  - 2 Transformer layers with 4096 hidden dimensions
  - Predicts discrete speech unit alignment sequences from LLM hidden states
  - A unit-based HiFi-GAN vocoder synthesizes waveforms from the predicted discrete units
  - Streaming: speech generation can begin before the full text response is complete

### Training Strategy (Two Stages)

1. **Stage 1:** Freeze speech encoder; train speech adaptor + LLM jointly using cross-entropy loss on text response generation. 3 epochs.
2. **Stage 2:** Freeze encoder + adaptor + LLM; train only the speech decoder with CTC loss to predict speech units from LLM hidden states.

**Total training time:** ~65 hours on 4 NVIDIA L40 GPUs (~3 days).

### Training Data: InstructS2S-200K

Constructed in three steps:
1. Rewrite 200K text instructions using Llama-3-70B (add fillers, convert symbols to spoken form)
2. Generate speech-appropriate responses (concise, no formatting/markdown)
3. Synthesize speech: instructions with CosyVoice-300M-SFT (random voices), responses with VITS
- Sources: 50K Alpaca + 150K UltraChat instructions

### Evaluation Results

| Metric | Value |
|--------|-------|
| ChatGPT Score (S2T Instruction Following) | 3.99 |
| ChatGPT Score (S2S Instruction Following) | 3.47 |
| ASR-WER | 10.82% |
| UTMOS (speech naturalness) | 3.93 |
| Minimum latency (streaming) | 226ms |

### Streaming / Real-time

- **Yes.** The non-autoregressive CTC decoder enables streaming speech generation with 226ms latency. Speech synthesis begins as the LLM generates text tokens.

### Multi-turn Conversation

- Initially single-turn. The InstructS2S-200K dataset was later extended (May 2025) to include multi-turn conversations with diversified input speech timbres.

---

## 4. LLaMA-Omni 2

| Field | Details |
|-------|---------|
| **Full Title** | LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis |
| **Authors** | Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, Yang Feng |
| **Date** | May 5, 2025 |
| **ArXiv ID** | 2505.02625 |
| **Venue** | Preprint |

### Architecture

- **Speech Input Encoding:** Whisper-large-v3 encoder
- **Speech Adaptor:** Downsampling (5x reduction) + feed-forward network with 2048 intermediate dimension
- **LLM Backbone:** Qwen2.5 series (0.5B, 1.5B, 3B, 7B, 14B variants)
- **Speech Tokenizer:** Finite Scalar Quantization (FSQ) module inserted into the encoder of SenseVoice-Large ASR model. Produces tokens at 25 tokens/second with vocabulary size 6561.
- **Speech Decoder:** Autoregressive streaming TTS language model initialized from Qwen2.5-0.5B.
  - Uses a "Read-Write" strategy: after the LLM generates R=3 text tokens, the TTS model produces W=10 speech tokens per chunk
  - Enables simultaneous text and speech generation in a streaming fashion

### Key Differences from LLaMA-Omni v1

- Replaces non-autoregressive CTC decoder with autoregressive streaming decoder
- Significantly improved speech naturalness (UTMOS 4.19 vs 3.93)
- Built on Qwen2.5 instead of Llama-3.1
- Native multi-turn conversation support
- Multiple model sizes (0.5B to 14B)

### Training Data

- 200K multi-turn speech-to-speech dialogues
- Synthesized from Alpaca and UltraChat datasets using Llama-3.3-70B for text generation
- Voice synthesis via Fish-Speech-1.5 and CosyVoice2-0.5B
- Multi-turn: turn counts sampled from Poisson(lambda=2), clamped 1-5, with varied voices for instructions and consistent voice for responses

### Evaluation Results

| Model | S2T Acc | S2S Acc | ChatGPT Score | ASR-WER | UTMOS | Latency (ms) |
|-------|---------|---------|---------------|---------|-------|-------------|
| LLaMA-Omni2-0.5B | 54.3% | 42.3% | 3.51 | 4.93 | 4.18 | 377.11 |
| LLaMA-Omni2-3B | 66.3% | 55.7% | 4.22 | 3.12 | 4.19 | 472.11 |
| LLaMA-Omni2-7B | 70.3% | 60.7% | 4.28 | 3.26 | 4.19 | 582.91 |
| LLaMA-Omni2-14B | 73.0% | 62.7% | 4.56 | 3.89 | 4.20 | 663.32 |
| GLM-4-Voice (9B) | 64.7% | 50.7% | 4.16 | 9.02 | 3.48 | 1562.81 |
| LLaMA-Omni (8B) | -- | -- | 3.47 | 10.82 | 3.93 | 346 |

### Streaming / Real-time

- **Yes.** Autoregressive streaming with Read-Write strategy. Latency ranges from ~377ms (0.5B) to ~663ms (14B).

### Multi-turn Conversation

- **Yes.** Natively supported. Training data includes multi-turn dialogues with 1-5 turns.

---

## 5. Mini-Omni

| Field | Details |
|-------|---------|
| **Full Title** | Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming |
| **Authors** | Zhifei Xie, Changqiao Wu |
| **Date** | August 29, 2024 |
| **ArXiv ID** | 2408.16725 |
| **Venue** | Preprint |

### Architecture

- **Speech Input Encoding:** Whisper-small encoder, connected via a 2-layer MLP adapter for feature alignment.
- **LLM Backbone:** Qwen2-0.5B (24 transformer blocks, 896 hidden dimension)
- **Speech Output Generation:** SNAC (Sub-band Neural Audio Codec) with 7 token layers in complementary relationships.
  - 8 parallel language model heads generate 8 tokens (including 1 text + 7 audio layers) per step
  - Text-Instructed Delay Parallel Decoding: text tokens generated first with N-token padding, audio tokens generated conditioned on corresponding text
  - Batch-parallel inference: input expanded to batch size 2 -- one generates text+audio (text discarded), other generates text-only (used to condition first sample's audio)

### Training Strategy ("Any Model Can Talk" -- Three Stages)

1. **Stage 1 -- Modality Alignment:** Freeze core model, train adapters on ASR/TTS data
2. **Stage 2 -- Adaptation Training:** Freeze adapters, train LLM on audio-input/text-response tasks
3. **Stage 3 -- Multi-modal Fine-tuning:** Unfreeze all weights for comprehensive training including audio output

### Training Data

- **VoiceAssistant-400K:** Over 400,000 entries generated by GPT-4o, specifically designed for speech assistant supervised fine-tuning. Avoids code/symbols, uses voice-assistant-appropriate language.
- Training: 8 A100 GPUs, batch size 192, 40,000 steps per epoch

### Evaluation Results

| Benchmark | Mini-Omni | Whisper-small | VITA |
|-----------|-----------|---------------|------|
| LibriSpeech test-clean WER | 4.5% | 3.4% | 8.14% |
| LibriSpeech test-other WER | 9.7% | 7.6% | 18.41% |

### Key Contributions

- First fully end-to-end, open-source model for real-time speech interaction
- "Any Model Can Talk" method that enables speech output capability for existing text LLMs
- Batch-parallel inference strategy to transfer text capabilities to audio modality
- Demonstrates that a 0.5B model can achieve real-time speech interaction

### Streaming / Real-time

- **Yes.** Parallel decoding generates hundreds of audio tokens per second with minimal first-token delay.

### Multi-turn Conversation

- Limited. Primarily designed for single-turn interactions due to small model size.

---

## 6. Mini-Omni 2

| Field | Details |
|-------|---------|
| **Full Title** | Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities |
| **Authors** | Zhifei Xie, Changqiao Wu |
| **Date** | October 15, 2024 |
| **ArXiv ID** | 2410.11190 |
| **Venue** | Preprint |

### Architecture

- **Speech/Audio Encoder:** Whisper-small model (same as Mini-Omni)
- **Vision Encoder:** CLIP ViT-B/32, producing feature sequences of length 50, processed through single-layer LlamaMLP adapter
- **LLM Backbone:** Qwen2-0.5B (ported via LitGPT framework), vocabulary expanded to 181,120 tokens
- **Speech Output Generation:** SNAC tokenizer with 7 complementary token layers. Text-Instruct Delay Parallel Decoding algorithm generates 8 tokens per step (1 text + 7 audio) with one-step delays between layers.

### Training Strategy (Three Stages)

1. **Stage 1 -- Multimodal Encoder Adaptation:** Train linear adapter layers connecting encoders to LLM
2. **Stage 2 -- Modality Alignment:** Freeze adapters, train LLM weights to transfer text QA capabilities to vision and audio inputs (text-only responses)
3. **Stage 3 -- Post-training:** Retrain all tasks with audio response generation enabled; integrate interruption mechanism

### Command-Based Interruption Mechanism

- Model receives noise-mixed audio containing "Stop Omni" phrases with random voice timbres
- Generates state tokens: "irq" (interrupt) or "n-irq" (no interrupt)
- On "irq" output: halts generation and initiates listening mode
- Semantic-level interruption (not just voice activity detection)

### Training Data

- ~3.4M total samples: 638 hours LibriTTS/VCTK/MLS (ASR), 1.5M Open-Orca (text QA), 1.5M Moss-002-sft (audio QA), 800K ALLaVA-4V (visual QA), plus RLHF data

### Evaluation Results

| Benchmark | Mini-Omni2 | Mini-Omni |
|-----------|------------|-----------|
| LibriSpeech test-clean WER | 4.8% | 4.5% |
| LibriSpeech test-other WER | 9.8% | 9.7% |

### Streaming / Real-time

- **Yes.** Real-time conversation with streaming audio generation and command-based interruption.

### Multi-turn Conversation

- Supports duplex interaction with interruption capability. Multi-modal input (vision + audio).

---

## 7. VITA

| Field | Details |
|-------|---------|
| **Full Title** | VITA: Towards Open-Source Interactive Omni Multimodal LLM |
| **Authors** | Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, et al. (19 authors) |
| **Date** | August 9, 2024 |
| **ArXiv ID** | 2408.05211 |
| **Venue** | Preprint |

### Architecture

- **Speech/Audio Encoder:** Custom encoder processing Mel Filter Bank features through 4x CNN downsampling layers followed by 24 transformer layers, totaling 341M parameters. Each 2 seconds of audio yields 25 tokens. Connected to LLM via 2-layer MLP.
- **Vision Encoder:** InternViT-300M-448px with dynamic patching for high-resolution images. Videos uniformly sampled (4-16 frames).
- **LLM Backbone:** Mixtral 8x7B (sparse mixture of experts). Vocabulary expanded from 32,000 to 51,747 tokens for Chinese proficiency.
- **Speech Output Generation:** External TTS tool (GPT-SoVITS) converts LLM text output to speech. **Not end-to-end for output** -- acknowledged by authors as "quite time-consuming" with end-to-end TTS identified as future work.

### Training Strategy

1. **Stage 1:** LLM bilingual tuning with 55M corpus
2. **Stage 2:** Multimodal alignment (visual + audio encoders; 2,749.9K entries across image/video/text)
3. **Stage 3:** Multimodal instruction tuning with state tokens (<1>=query audio, <2>=noisy audio, <3>=text query)

### Duplex Interaction

Two VITA models deployed simultaneously: one generates responses while the other monitors environmental audio. Upon detecting new query audio (via state token filtering), generation pauses, models swap roles -- enabling audio interruption.

### Evaluation Results

| Benchmark | Score |
|-----------|-------|
| C-EVAL (Chinese) | 56.68% |
| MMLU | 70.98% |
| GSM8K | 75.66% |
| WenetSpeech CER (test_net) | 12.15% |
| LibriSpeech WER (test-clean) | 8.14% |

### Streaming / Real-time

- **Partial.** The duplex architecture enables interruption handling, but speech output relies on external TTS, adding significant latency.

### Multi-turn Conversation

- **Yes.** Supports multi-turn multimodal conversations across video, image, text, and audio.

---

## 8. Freeze-Omni

| Field | Details |
|-------|---------|
| **Full Title** | Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM |
| **Authors** | Xiong Wang, Yangze Li, Chaoyou Fu, Yunhang Shen, Lei Xie, Ke Li, Xing Sun, Long Ma |
| **Date** | November 1, 2024 (v5: December 8, 2024) |
| **ArXiv ID** | 2411.00774 |
| **Venue** | Preprint |

### Architecture

- **Speech Encoder:** Custom encoder with multi-layer convolution (4x downsampling) + 24 transformer layers (hidden size 1024), ~350M parameters. Adapter uses multi-convolution with 2x downsampling. Output at 12.5Hz frame rate. Processes mel-filter bank features (25ms window, 10ms shift).
- **LLM Backbone:** Qwen2-7B-Instruct (**frozen throughout entire training**). This is the core innovation.
- **Speech Decoder:** Two components:
  - **NAR (Non-AutoRegressive) Decoder:** Models semantic features from LLM output during prefill stage. 4-layer Llama decoder, hidden size 896, ~120M parameters.
  - **AR (AutoRegressive) Decoder:** Generates speech tokens sequentially after NAR prefill. Same architecture.
- **Codec:** TiCodec with single 1024-size codebook, 40Hz speech token frequency, 24kHz output sample rate.

### Training Strategy (Three Stages for Input + Three Stages for Output)

**Speech Input:**
1. CTC-based ASR training on 110,000h paired data
2. Connect encoder to frozen LLM via trainable adapter; train on transcript labels
3. Add trainable prompt embeddings; train on 60,000 multi-round text Q&A with synthesized speech inputs

**Speech Output:**
1. Train codec on 3,000h speech data
2. Train NAR/AR decoders on text-speech pairs using teacher forcing
3. Add NAR prefix decoder for LLM hidden states; fine-tune alignment

**Key principle:** LLM parameters never change. All adaptation happens in encoder, adapter, and decoder.

### Evaluation Results

| Metric | Score |
|--------|-------|
| Aishell-1 CER | 2.15% |
| LibriSpeech test-clean WER | 3.24% |
| LibriSpeech test-other WER | 7.68% |
| Web Questions accuracy | 44.73% |
| LLaMA Questions accuracy | 72.0% |
| Audio Trivia QA accuracy | 53.88% |
| Speech decoder CER | 1.69-1.99% |
| Average latency | 745ms |
| End-to-end real-world latency | ~1.2s |

### Streaming / Real-time

- **Yes.** Chunk-wise streaming input processing. Duplex dialogue via multi-task training with chunk-level state prediction.

### Multi-turn Conversation

- **Yes.** Trained on 60,000 multi-round Q&A data. Duplex enables natural conversation flow.

### Key Contributions

- Demonstrates that speech capabilities can be added to an LLM without modifying any LLM parameters, completely avoiding catastrophic forgetting.
- Speech modality intelligence matches text modality intelligence.
- "Model as a Server" strategy for multi-user concurrent serving.

---

## 9. GLM-4-Voice

| Field | Details |
|-------|---------|
| **Full Title** | GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot |
| **Authors** | Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, Jie Tang |
| **Date** | December 3, 2024 |
| **ArXiv ID** | 2412.02612 |
| **Venue** | Preprint (Tsinghua University / Zhipu AI) |

### Architecture

- **Speech Tokenizer:** Ultra-low bitrate (175 bps) single-codebook speech tokenizer at 12.5Hz frame rate. Built by adding a Vector Quantization (VQ) bottleneck into a Whisper encoder, then supervised training on ASR data. Produces discrete tokens from continuous speech.
- **LLM Backbone:** GLM-4-9B (9 billion parameters), a pre-trained text language model.
- **Speech Decoder:** Flow Matching architecture based on CosyVoice, supporting streaming inference. Minimum 10 speech tokens needed to start generation (only ~0.8 seconds of speech context required for streaming).
- **Speech Generation:** The LLM generates interleaved text and speech tokens. Text tokens are generated for reasoning/content, speech tokens for acoustic realization. The decoder converts speech tokens to waveforms.

### Training Strategy

1. **Speech-text interleaved data synthesis:** Novel method to synthesize interleaved speech-text training data from existing text corpora using a text-to-token model.
2. **Continued pre-training:** 1 trillion tokens combining unsupervised speech data, interleaved speech-text data, and supervised speech-text data.
3. **Fine-tuning:** On high-quality conversational speech data for instruction following and voice control (emotion, intonation, speech rate, dialect).

### Training Data

- Millions of hours of audio
- Hundreds of billions of tokens of interleaved text-speech data
- Supports both Chinese and English

### Evaluation Results (from LLaMA-Omni2 comparison)

| Metric | GLM-4-Voice |
|--------|------------|
| S2T Accuracy (LLaMA Questions) | 64.7% |
| S2S Accuracy | 50.7% |
| ChatGPT Score | 4.16 |
| ASR-WER | 9.02% |
| UTMOS | 3.48 |
| Latency | 1562.81ms |

### Key Contributions

- End-to-end speech-to-speech model that can control vocal nuances (emotion, intonation, speech rate, dialect) via user instructions.
- Ultra-low bitrate speech tokenizer enables efficient discrete speech modeling.
- Massive-scale pre-training on interleaved speech-text data.

### Streaming / Real-time

- **Yes.** Streaming inference supported via the Flow Matching decoder. Outputs text and speech modalities alternately.

### Multi-turn Conversation

- **Yes.** Designed as a conversational chatbot with multi-turn capability.

---

## 10. Ichigo

| Field | Details |
|-------|---------|
| **Full Title** | Ichigo: Mixed-Modal Early-Fusion Realtime Voice Assistant |
| **Authors** | Alan Dao (Gia Tuan Dao), Dinh Bach Vu, Huy Hoang Ha |
| **Date** | October 20, 2024 |
| **ArXiv ID** | 2410.15316 |
| **Venue** | Preprint |

### Architecture

- **Speech Tokenizer:** WhisperVQ (from WhisperSpeech). 512 tokens with codebook dimension 64. Processes audio at 16kHz, 25Hz frame rate. Converts audio to log-mel spectrograms, processed through Whisper encoder, then quantized to discrete tokens. Token format: `<|sound_dddd|>`.
- **LLM Backbone:** Llama-3.1-8B-Instruct (pre-trained on 15 trillion text tokens). Vocabulary expanded with 512 new speech tokens plus `<|sound_start|>` and `<|sound_end|>` delimiters. New embeddings initialized by averaging existing vocabulary.
- **Connection:** Tokenized early fusion -- speech quantized to discrete tokens in the same format as text tokens. Single uniform transformer processes both modalities. **No separate encoder, adapter, or domain-specific decoder needed.**
- **Speech Output:** Text-only output. Ichigo focuses on speech-to-text understanding; it does not generate speech output.

### Training Strategy

- **Pre-training:** 16,000 hours multilingual ASR data (8 languages): 10,000h English (MLS), 6,000h other languages
- **Post-training (3 stages):**
  1. Instruction fine-tuning: 70% speech instruction, 20% transcription, 10% text-only
  2. Enhancement fine-tuning: multi-turn conversations, refusal scenarios (0.5% refusal data)
  3. Synthetic noise augmentation with randomized tokens
- Total: 2.2M curated instruction samples -> 1.3M speech-text pairs

### Evaluation Results

| Benchmark | Ichigo | Comparison |
|-----------|--------|------------|
| OpenHermes-Audio | 67.8 | Cascaded system: 63.0 |
| ALPACA-Audio | 67.2 | Cascaded: 70.8 |
| MMLU (5-shot) | 63.79 | Original Llama: 69.4 (8.4% degradation) |
| GPQA (0-shot) | 29.69 | Original: 30.4 |
| GSM-8K (8-shot CoT) | 75.28 | Original: 84.5 |

Outperforms Qwen2-Audio by 23 points on OpenHermes-Audio.

### Latency

- First token: 111.52 +/- 7.73 ms (~4x faster than cascaded Whisper+Llama at 453ms, ~3x faster than Qwen2-Audio at 317ms)
- VRAM: 19 GB (vs 32 GB for Qwen2-Audio)

### Streaming / Real-time

- **Yes** for speech understanding (111ms to first token). **No** for speech generation (text output only).

### Multi-turn Conversation

- **Yes.** Handles 4-5 conversation turns within 4096 token max context length.

### Key Contributions

- Demonstrates that early fusion via discrete tokenization can match or exceed adapter-based approaches.
- Minimal text capability degradation (<10%) while adding speech understanding.
- Fully open-source, designed for small research teams.

---

## 11. IntrinsicVoice

| Field | Details |
|-------|---------|
| **Full Title** | IntrinsicVoice: Empowering LLMs with Intrinsic Real-time Voice Interaction Abilities |
| **Authors** | Xin Zhang, Xiang Lyu, Zhihao Du, Qian Chen, Dong Zhang, Hangrui Hu, Chaohong Tan, Tianyu Zhao, Yuxuan Wang, Bin Zhang, Heng Lu, Yaqian Zhou, Xipeng Qiu |
| **Date** | October 9, 2024 |
| **ArXiv ID** | 2410.08035 |
| **Venue** | Preprint |

### Architecture

- **Speech Input Encoding:** HuBERT (mhubert-base-25hz) at 25Hz with 500 clusters via k-means quantization. Converts speech waveforms to discrete token sequences.
- **LLM Backbone:** Qwen2-7B-Instruct, extended with speech tokens (`<sosp>`, `<eosp>`, `<speech>`).
- **GroupFormer:** Novel architecture that augments the LLM. When the LLM predicts a `<speech>` token, the GroupModel (a smaller non-autoregressive transformer encoder) receives the LLM's hidden state + G learnable queries and predicts an entire group of G speech tokens simultaneously. This reduces speech output to ~5 tokens/second (matching ~4.2 text tokens/second in LibriSpeech).
- **Speech Output:** HiFi-GAN vocoder converts discrete speech tokens to waveforms. Uses non-causal convolution, enabling streaming once N_offset tokens are decoded.

### Training Strategy

- Four cross-modal tasks from speech quadruples: speech-to-speech, speech-to-text, text-to-speech, text-to-text
- Combined loss: L_LLM (autoregressive text) + L_G (grouped speech token prediction)
- 8 A100 GPUs, DeepSpeed ZeRO-2, 4 epochs, batch size 256, learning rate 1.5e-4, max sequence length 1200

### Training Data: IntrinsicVoice-500k

- ~500k dialogue turns from multiple sources:
  - 89k multi-turn speech QA pairs (345k turns) from Moss-002-sft-data and CoQA
  - 87k single-turn from SQuAD
  - 54k Spoken-Alpaca-GPT samples
  - 20k hours from Multilingual LibriSpeech (ASR/TTS)
  - 350k text QA from Guanaco_Belle_Merge_v1.0

### Evaluation Results

| Metric | IntrinsicVoice | SpeechGPT |
|--------|---------------|-----------|
| S2S ChatGPT Score | 3.05 | 2.83 |
| S2T ChatGPT Score | 3.22 | 3.04 |
| T2S ChatGPT Score | 3.39 | 3.41 |
| T2T ChatGPT Score | 3.87 | 3.72 |
| UTMOS | 3.75 | 3.73-3.81 |
| Latency | <100ms | 1.07-1.21s |

### Streaming / Real-time

- **Yes.** Achieves <100ms latency (approximately 1/10th of SpeechGPT). GroupFormer enables efficient streaming by reducing speech sequence length to match text.

### Multi-turn Conversation

- **Yes.** Trained on multi-turn dialogue data (345k turns from 89k conversations).

### Key Contributions

- GroupFormer architecture solves the fundamental length mismatch between speech and text sequences.
- Achieves real-time voice interaction (<100ms) while maintaining speech quality.
- Cross-modal training on all four task combinations (S2S, S2T, T2S, T2T).

---

## 12. OmniFlatten

| Field | Details |
|-------|---------|
| **Full Title** | OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation |
| **Authors** | Qinglin Zhang, Luyao Cheng, Chong Deng, Qian Chen, Wen Wang, Siqi Zheng, Jiaqing Liu, Hai Yu, Chaohong Tan, Zhihao Du, Shiliang Zhang |
| **Date** | October 23, 2024 |
| **ArXiv ID** | 2410.17799 |
| **Venue** | Work in progress |

### Architecture

- **Speech Tokenizer:** CosyVoice speech tokenizer with VQ layer. Single codebook of 4096 codes, discretizing audio into speech tokens.
- **LLM Backbone:** Qwen2-0.5B
- **Speech Output:** CosyVoice Optimal-transport Conditional Flow Matching (OT-CFM) model converts speech tokens to mel spectrograms, then HiFi-GAN vocoder produces final audio.
- **Flattening Operation:** Core innovation. Interleaves speech and text token sequences into a unified flat sequence:
  - Half-duplex: User Speech -> User Text -> Assistant Text -> Assistant Speech
  - Full-duplex: Chunked with text chunks of 2 tokens and speech chunks of 10 tokens, interleaved as: input speech, output text, output speech

### Training Strategy (Three Stages)

1. **Stage 1 -- Modality Alignment:** SFT on ASR and TTS tasks using ~100K hours paired speech-text data (30% open-source from Aishell-3, LibriTTS, TED-LIUM, VoxPopuli, LibriSpeech, MLS, WenetSpeech; 70% proprietary)
2. **Stage 2 -- Half-Duplex Dialogue:** Turn-based dialogue training with all four streams (user/assistant speech and text)
3. **Stage 3 -- Full-Duplex Dialogue:** Progressive training:
   - First on 3 streams (removing user text)
   - Then on 2 streams (removing assistant text)
   - Training data: 2,000 hours synthesized full-duplex conversation from 390K filtered multi-turn text dialogues using CosyVoice

### Evaluation Results

| Metric | Score |
|--------|-------|
| LibriSpeech test-clean WER | 7.91% |
| LibriSpeech test-other WER | 19.21% |
| WenetSpeech test-meeting CER | 26.1% |
| WenetSpeech test-net CER | 19.0% |
| LibriTTS TTS WER | 4.51% |
| English chat LLM score (text) | 4.88/10 |
| English chat LLM score (speech) | 3.92/10 |
| Chinese chat LLM score (text) | 5.6/10 |
| Chinese chat LLM score (speech) | 5.15/10 |
| Assistant turn-taking accuracy @25 | 71.7% (vs Moshi: 55.1%) |
| Assistant response latency | 193ms (vs Moshi: 553ms) |
| User turn-taking latency | 287ms (vs Moshi: 753ms) |

### Streaming / Real-time

- **Yes.** Full-duplex conversation with simultaneous bidirectional communication. Handles turn-taking, interruptions, and backchannels.

### Multi-turn Conversation

- **Yes.** Trained on 390K multi-turn text dialogues converted to speech.

### Key Contributions

- Demonstrates that a text LLM can be adapted to full-duplex speech-text dialogue without any architectural modifications, using only a flattening operation.
- Significantly outperforms Moshi on turn-taking latency (193ms vs 553ms).
- Progressive full-duplex training from half-duplex.

---

## 13. SLAM-Omni

| Field | Details |
|-------|---------|
| **Full Title** | SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training |
| **Authors** | Wenxi Chen, Ziyang Ma, Ruiqi Yan, Yuzhe Liang, Xiquan Li, Ruiyang Xu, Zhikang Niu, Yanqiao Zhu, Yifan Yang, Zhanxun Liu, Kai Yu, Yuxuan Hu, Jinyu Li, Yan Lu, Shujie Liu, Xie Chen |
| **Date** | December 20, 2024 |
| **ArXiv ID** | 2412.15649 |
| **Venue** | Preprint (Shanghai Jiao Tong University / Microsoft) |

### Architecture

- **Speech Encoding:** Grouped speech semantic tokens. Speech is encoded into semantic tokens, which are then grouped (group size = 3) to reduce audio token sequence length, improving training and inference speed.
- **LLM Backbone:** Not explicitly named in public documentation; the framework (SLAM-LLM) is codec-agnostic and supports multiple backbones.
- **Speech Decoder/Vocoder:** CosyVoice-based vocoder. Speaker information is decoupled from semantic tokens and routed to the vocoder, enabling zero-shot timbre control.
- **Historical Text Prompting:** Dialogue history is compressed into text prompts rather than retaining full audio history, enabling efficient multi-round interactions.

### Training Strategy

- **Single-stage training:** First spoken dialogue system to achieve competitive performance with a single training stage (no separate ASR/TTS pre-training required).
- Training time: Only 15 hours on 4 GPUs with limited data.

### Training Data

- VoiceAssistant-400K (400K single-round English)
- UltraChat-300K (300K multi-round English)
- Belle_1.4M (1.4M multi-round Chinese)

### Key Contributions

- Zero-shot timbre control: Can generate speech in any target voice without fine-tuning, by decoupling speaker identity to the vocoder.
- Single-stage training efficiency: Eliminates complex multi-stage pipelines.
- Grouped semantic tokens for sequence length reduction.
- Historical text prompting for efficient multi-round dialogue.

### Evaluation Results

- Outperforms prior models of similar scale (specific numbers available in arXiv paper).
- Competitive with multi-stage trained systems despite single-stage approach.

### Streaming / Real-time

- **Yes.** Designed for end-to-end real-time spoken dialogue.

### Multi-turn Conversation

- **Yes.** Multi-round and multilingual (Chinese and English) dialogue supported via historical text prompting.

---

## 14. Westlake-Omni

| Field | Details |
|-------|---------|
| **Full Title** | Westlake-Omni: Open-Source Chinese Emotional Speech Interaction Large Language Model with Unified Discrete Sequence Modeling |
| **Authors** | Xinchen AI (organization; individual authors not publicly listed) |
| **Date** | ~Late 2024 / Early 2025 (no formal paper date) |
| **ArXiv ID** | None (no published arxiv paper as of March 2026) |
| **Venue** | Open-source project (GitHub + HuggingFace) |

### Architecture

- **Speech Encoding:** Discrete representations to unify speech and text processing. Specific encoder architecture not documented publicly.
- **LLM Backbone:** Qwen2 (specific size not disclosed).
- **Speech Decoder:** VQGAN-based (Vector Quantized Generative Adversarial Network). Weights provided separately.
- **Approach:** Unified discrete sequence modeling -- both speech and text are represented as discrete token sequences processed by the same LLM.
- **Codebase:** Built upon Fish Speech architecture.

### Key Capabilities

- Native Chinese emotional speech interaction
- Low-latency simultaneous text and speech generation
- Trained on high-quality Chinese emotional speech dataset (size undisclosed)

### Evaluation Results

- No published benchmark results.

### Streaming / Real-time

- **Yes.** Low-latency speech interaction with concurrent text and speech generation.

### Multi-turn Conversation

- Not explicitly documented.

### Notes

- This is an open-source project without a formal academic paper. It represents a practical implementation focused specifically on Chinese emotional speech, using the Qwen2 backbone with VQGAN for speech synthesis.
- License: Code and VQGAN weights under CC-BY-NC-SA-4.0; LLM weights under Apache 2.0.
- GitHub: https://github.com/xinchen-ai/Westlake-Omni
- HuggingFace: https://huggingface.co/xinchen-ai/Westlake-Omni

---

## 15. Cross-Model Comparison Tables

### Architecture Comparison

| Model | Speech Encoder | LLM Backbone | Speech Decoder | Adapter Type |
|-------|---------------|-------------|----------------|-------------|
| SpeechGPT | HuBERT k-means (discrete units) | LLaMA 7B/13B | Unit-based HiFi-GAN vocoder | Vocabulary expansion (early fusion) |
| SpeechGPT-Gen | SpeechTokenizer RVQ-1 | LLaMA2-7B-Chat | Flow matching (perceptual) + vocoder | Vocabulary expansion |
| LLaMA-Omni | Whisper-large-v3 (frozen) | Llama-3.1-8B-Instruct | Non-AR CTC decoder + HiFi-GAN | 2-layer MLP with 5x downsampling |
| LLaMA-Omni 2 | Whisper-large-v3 | Qwen2.5 (0.5B-14B) | AR streaming TTS (Qwen2.5-0.5B init) | FFN with 5x downsampling |
| Mini-Omni | Whisper-small | Qwen2-0.5B | SNAC 7-layer codec, 8 parallel heads | 2-layer MLP |
| Mini-Omni 2 | Whisper-small + CLIP ViT-B/32 | Qwen2-0.5B | SNAC 7-layer codec | Linear + LlamaMLP adapters |
| VITA | Custom CNN+Transformer (341M) | Mixtral 8x7B | External TTS (GPT-SoVITS) | 2-layer MLP |
| Freeze-Omni | Custom Conv+Transformer (350M) | Qwen2-7B (frozen) | NAR+AR decoder + TiCodec | Multi-conv adapter |
| GLM-4-Voice | Whisper+VQ tokenizer (175bps) | GLM-4-9B | Flow Matching (CosyVoice-based) | Integrated (shared vocab) |
| Ichigo | WhisperVQ (512 tokens) | Llama-3.1-8B-Instruct | None (text output only) | Vocabulary expansion (early fusion) |
| IntrinsicVoice | HuBERT (mhubert-25hz, 500 clusters) | Qwen2-7B-Instruct | GroupFormer + HiFi-GAN | Vocabulary expansion + GroupFormer |
| OmniFlatten | CosyVoice tokenizer (4096 codes) | Qwen2-0.5B | OT-CFM + HiFi-GAN | Flattening (no adapter) |
| SLAM-Omni | Grouped semantic tokens | SLAM-LLM framework | CosyVoice vocoder | Semantic token grouping |
| Westlake-Omni | Discrete representations | Qwen2 | VQGAN | Unified discrete vocab |

### Capability Comparison

| Model | Streaming | Full Duplex | Multi-turn | Emotion Control | Timbre Control | Multilingual |
|-------|-----------|-------------|------------|-----------------|----------------|-------------|
| SpeechGPT | No | No | Limited | No | No | No |
| SpeechGPT-Gen | No | No | Yes | No | Yes (voice conversion) | Yes |
| LLaMA-Omni | Yes (226ms) | No | Yes (later) | No | No | No |
| LLaMA-Omni 2 | Yes (377-663ms) | No | Yes | No | No | No |
| Mini-Omni | Yes | No | Limited | No | No | No |
| Mini-Omni 2 | Yes | Partial (interrupt) | Yes | No | No | No |
| VITA | Partial | Partial (dual model) | Yes | No | No | Yes (CN/EN) |
| Freeze-Omni | Yes (745ms) | Yes | Yes | No | No | Yes (CN/EN) |
| GLM-4-Voice | Yes | No | Yes | Yes | Yes | Yes (CN/EN) |
| Ichigo | Yes (111ms) | No | Yes | No | No | Yes (8 langs) |
| IntrinsicVoice | Yes (<100ms) | No | Yes | No | No | No |
| OmniFlatten | Yes (193ms) | Yes | Yes | No | No | Yes (CN/EN) |
| SLAM-Omni | Yes | No | Yes | No | Yes (zero-shot) | Yes (CN/EN) |
| Westlake-Omni | Yes | No | Unknown | Yes (Chinese) | No | Chinese |

### Scale and Training Efficiency

| Model | Parameters | Training GPUs | Training Time | Training Data Size |
|-------|-----------|--------------|--------------|-------------------|
| SpeechGPT | 7-13B | -- | -- | SpeechInstruct (undisclosed size) |
| SpeechGPT-Gen | 8B | A100s | 77K steps | MultiLingual LS, GigaSpeech, etc. |
| LLaMA-Omni | 8B | 4x L40 | ~65 hours | 200K pairs |
| LLaMA-Omni 2 | 0.5B-14B | -- | -- | 200K multi-turn dialogues |
| Mini-Omni | 0.5B | 8x A100 | 40K steps/epoch | 400K (VoiceAssistant) |
| Mini-Omni 2 | 0.5B | -- | -- | ~3.4M samples |
| VITA | ~47B (MoE) | -- | -- | 55M text + 2.7M multimodal |
| Freeze-Omni | 7B (frozen) + ~470M trainable | 8 GPUs | -- | 110Kh ASR + 3Kh TTS + 60K QA |
| GLM-4-Voice | 9B | -- | -- | 1T tokens (millions of hours) |
| Ichigo | 8B | -- | -- | 16Kh + 1.3M pairs |
| IntrinsicVoice | 7B | 8x A100 | 4 epochs | 500K turns + 20Kh audio |
| OmniFlatten | 0.5B | -- | -- | ~100Kh + 2Kh duplex |
| SLAM-Omni | -- | 4 GPUs | 15 hours | 400K + 300K + 1.4M |
| Westlake-Omni | Unknown | -- | -- | Chinese emotional speech (undisclosed) |

---

## 16. VoiceBench Results

VoiceBench (arXiv: 2410.17196) is a benchmark for evaluating LLM-based voice assistants across general knowledge (AlpacaEval, CommonEval, SD-QA), instruction following (IFEval), and safety (AdvBench).

| Model | Text Score | Speech Score |
|-------|-----------|-------------|
| Naive Pipeline (ASR+LLM+TTS) | 86.25 | 81.88 |
| DiVA | 85.51 | 67.73 |
| LLaMA-Omni | 76.29 | 41.83 |
| VITA | 76.08 | 37.62 |
| Qwen2-Audio | 67.78 | 60.45 |
| Mini-Omni2 | 45.70 | 33.67 |
| Mini-Omni | 43.75 | 28.80 |

**Key finding:** The naive cascade pipeline (ASR -> text LLM -> TTS) significantly outperforms all end-to-end models on spoken instructions by >10 percentage points, highlighting the gap that remains for end-to-end speech LLMs.

---

## Summary of Key Architectural Patterns

### Speech Input Approaches
1. **Continuous encoder + adapter** (most common): Whisper/HuBERT encoder -> MLP/linear adapter -> LLM. Used by LLaMA-Omni, Mini-Omni, VITA, Freeze-Omni.
2. **Discrete tokenization + vocabulary expansion** (early fusion): Speech quantized to discrete tokens added to LLM vocabulary. Used by SpeechGPT, Ichigo, GLM-4-Voice.
3. **Grouped tokens**: Tokens grouped to reduce sequence length. Used by IntrinsicVoice (GroupFormer), SLAM-Omni.

### Speech Output Approaches
1. **Non-autoregressive CTC**: Predicts alignment of discrete units in parallel. Used by LLaMA-Omni v1.
2. **Autoregressive token generation**: LLM or separate decoder generates speech tokens one by one. Used by SpeechGPT, LLaMA-Omni 2, GLM-4-Voice.
3. **Parallel multi-layer codec decoding**: Multiple codec layers decoded simultaneously. Used by Mini-Omni (SNAC 7-layer).
4. **Two-stage semantic + perceptual**: Semantic tokens first, then acoustic details via flow matching. Used by SpeechGPT-Gen, GLM-4-Voice.
5. **GroupFormer**: Non-autoregressive group prediction from LLM hidden states. Used by IntrinsicVoice.
6. **External TTS**: Not truly end-to-end. Used by VITA.

### LLM Freezing Strategies
- **Fully frozen LLM**: Freeze-Omni (only trains encoder/adapter/decoder)
- **Partially frozen**: LLaMA-Omni Stage 2 (freezes LLM for decoder training)
- **Fully fine-tuned**: Most others (Mini-Omni, SpeechGPT, etc.)
