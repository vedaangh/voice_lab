# Speech Tokenization and Neural Audio Codecs for Voice LLMs
## Comprehensive Research Survey

---

## Table of Contents
1. [Overview and Taxonomy](#1-overview-and-taxonomy)
2. [HuBERT (Meta)](#2-hubert-meta)
3. [EnCodec (Meta)](#3-encodec-meta)
4. [SoundStream (Google)](#4-soundstream-google)
5. [Mimi (Kyutai / Moshi)](#5-mimi-kyutai--moshi)
6. [WavTokenizer](#6-wavtokenizer)
7. [SpeechTokenizer](#7-speechtokenizer)
8. [DAC (Descript Audio Codec)](#8-dac-descript-audio-codec)
9. [Whisper as Encoder (OpenAI)](#9-whisper-as-encoder-openai)
10. [CosyVoice Codec Approach](#10-cosyvoice-codec-approach)
11. [SNAC (Multi-Scale Neural Audio Codec)](#11-snac-multi-scale-neural-audio-codec)
12. [Vocos (Vocoder)](#12-vocos)
13. [The Key Debate: Semantic vs Codec vs Continuous](#13-the-key-debate-semantic-vs-codec-vs-continuous)
14. [Summary Comparison Table](#14-summary-comparison-table)
15. [Which Voice LLMs Use Which Approach](#15-which-voice-llms-use-which-approach)

---

## 1. Overview and Taxonomy

Speech tokenization is the foundational representation choice underpinning all speech LLMs. The design space divides into three major paradigms:

**A. Semantic Tokens** — Derived from self-supervised speech models (HuBERT, w2v-BERT, WavLM), discretized via K-means clustering. Capture linguistic/phonetic content but lose speaker identity, prosody, and acoustic fidelity.

**B. Acoustic/Codec Tokens** — Produced by neural audio codecs (EnCodec, SoundStream, DAC, SNAC) using Residual Vector Quantization (RVQ). Capture full audio signal including speaker identity, prosody, and acoustic environment, but at the cost of higher token counts and less clean semantic separation.

**C. Continuous Features** — Raw encoder outputs (from Whisper, HuBERT, etc.) projected into LLM embedding space via linear layers or adapters. Preserve maximum information but cannot be generated autoregressively and require specialized architectures.

**D. Hybrid/Disentangled** — SpeechTokenizer, Mimi, and CosyVoice attempt to unify semantic and acoustic information in a single codec framework, typically by distilling semantic models into the first RVQ level.

---

## 2. HuBERT (Meta)

**Paper**: "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units" (Hsu et al., 2021)

### Architecture
- **Type**: Self-supervised speech representation model (not a codec)
- **Backbone**: Transformer encoder (same architecture family as wav2vec 2.0)
- **Model sizes**:
  - BASE: ~90M parameters, 12 Transformer layers
  - LARGE: ~300M parameters, 24 Transformer layers
  - X-LARGE: ~1B parameters, 48 Transformer layers
- **Input**: Raw 16 kHz waveform, processed through convolutional feature extractor
- **Output**: Continuous frame-level representations (one per ~20ms)

### Self-Supervised Training
- Uses masked prediction of hidden units (BERT-style masking applied to speech)
- **Iterative clustering approach**:
  - Iteration 0: Extract MFCC features, apply K-means with 100 clusters to get initial pseudo-labels
  - Iteration 1: Extract features from 6th layer of trained BASE model, K-means with 500 clusters
  - Iteration 2: Extract from 9th layer, K-means with 500 clusters
- Training objective: predict cluster assignments only for masked positions
- Key insight: the model does not require high-quality cluster labels; consistency of clustering matters more than intrinsic quality

### Discrete Token Generation
- **Not inherent** — HuBERT produces continuous representations
- **Discretization via K-means**: Representations from specific layers are clustered offline
  - Typically 100 to 2000 K-means clusters (codebook sizes)
  - Most common: 500 or 1000 clusters from middle layers
- **Token rate**: 50 Hz (one token per 20ms frame)
- A single stream of tokens per utterance (no RVQ hierarchy)

### Information Captured
- **Primarily semantic/linguistic**: Phonetic content, word identity, linguistic structure
- **Partially prosodic**: Some prosodic information survives in certain layers
- **Loses**: Fine speaker identity, acoustic environment, high-frequency details, exact pitch contour
- Upper layers are more semantic; lower layers retain more acoustic detail

### Bitrate
- With 500 clusters at 50 Hz: ~450 bps (50 tokens/sec x ~9 bits/token)
- With 1000 clusters: ~500 bps
- Extremely low bitrate, but only encodes semantic content

### Reconstruction Quality
- **Cannot reconstruct audio alone** — needs a separate vocoder/decoder
- Resynthesized speech (via unit-based HiFi-GAN) sounds intelligible but loses speaker identity
- Typical pipeline: HuBERT tokens -> duration model -> unit HiFi-GAN -> waveform
- Word Error Rate on reconstructed speech is competitive, but speaker similarity is poor

### Downstream Usage in Voice LLMs
- **AudioLM** (Google): Uses HuBERT-like semantic tokens (from w2v-BERT) as the first stage of a hierarchical generation pipeline, followed by SoundStream acoustic tokens
- **SpeechGPT**: Converts speech to HuBERT discrete tokens (K-means), expands LLM vocabulary to include speech tokens, trains with three-stage pipeline
- **GSLM** (Meta): Generative Spoken Language Model that operates entirely on HuBERT K-means tokens for speech-to-speech generation
- **TWIST**: Uses HuBERT tokens as the speech vocabulary for a text-free spoken language model
- **pGSLM**: Extension of GSLM adding prosody through separate pitch/duration streams alongside HuBERT tokens

---

## 3. EnCodec (Meta)

**Paper**: "High Fidelity Neural Audio Compression" (Defossez et al., 2022)

### Architecture
- **Type**: End-to-end neural audio codec with encoder-decoder and RVQ
- **Encoder**: 1D convolutional network with residual blocks
  - Initial convolution: C=128 channels (hidden_size), kernel_size=7
  - 4 downsampling blocks with stride ratios [8, 5, 4, 2] (total downsampling = 320x)
  - Each block: residual convolution layers with dilation growth rate of 2
  - 2 LSTM layers at the bottleneck for temporal modeling
  - Causal convolutions for streaming
- **Decoder**: Mirror of encoder with transposed convolutions for upsampling
  - Upsampling ratios [2, 4, 5, 8] (reverse of encoder)
  - 2 LSTM layers, residual blocks
- **Quantizer**: Residual Vector Quantization (RVQ)
  - Codebook size: **1024 entries** per codebook
  - Codebook dimension: matches hidden_size (128 by default)
  - Variable number of codebooks depending on target bandwidth
- **Discriminator**: Multi-Scale STFT (MS-STFT) discriminator
- **Loss balancer**: Novel mechanism to stabilize multi-loss training

### RVQ and Bandwidth Mapping (24 kHz model)
| Target Bandwidth | Number of Codebooks | Tokens/sec | Bits/sec |
|-------------------|--------------------:|------------|----------|
| 1.5 kbps          | 2                   | 150        | 1,500    |
| 3 kbps            | 4                   | 300        | 3,000    |
| 6 kbps            | 8                   | 600        | 6,000    |
| 12 kbps           | 16                  | 1,200      | 12,000   |
| 24 kbps           | 32                  | 2,400      | 24,000   |

- Frame rate: **75 Hz** for the 24 kHz model (24000 / 320 = 75 frames/sec)
- Each codebook contributes 10 bits per frame (log2(1024) = 10)
- At 6 kbps (8 codebooks): 8 x 10 x 75 = 6000 bits/sec

### Model Variants
- **24 kHz monophonic** (causal, streaming): 1.5-24 kbps
- **48 kHz stereophonic** (non-causal): 3-24 kbps, codebook counts halved due to doubled frame rate

### Information Captured
- **All information**: Speaker identity, prosody, acoustic environment, linguistic content, background noise
- **RVQ hierarchy**: First codebooks capture coarse structure (pitch, energy, broad spectral shape); later codebooks capture fine details (harmonics, noise texture, high-frequency content)
- No explicit semantic-acoustic disentanglement — all information is mixed across codebooks

### Compression Ratio
- At 24 kHz, 16-bit audio: raw = 384 kbps
- At 6 kbps: ~64x compression
- At 1.5 kbps: ~256x compression

### Reconstruction Quality
- At 6 kbps: high quality for speech, near-transparent for most listeners
- At 3 kbps: good quality, some artifacts in music
- At 1.5 kbps: acceptable for speech, noticeable degradation
- Outperforms Opus at comparable bitrates in MUSHRA evaluations
- Optional Transformer-based entropy coding can further compress by ~40%

### Downstream Usage in Voice LLMs
- **VALL-E** (Microsoft): Uses EnCodec at 6 kbps (8 codebooks). AR model generates first codebook tokens from text + 3s prompt; NAR model generates remaining 7 codebooks. Treats TTS as codec language modeling.
- **MusicGen** (Meta): Uses EnCodec tokens for music generation with delay pattern for parallel codebook modeling
- **AudioGen** (Meta): Sound effect generation using EnCodec
- **VoiceCraft**: Speech editing using EnCodec tokens with masked generation
- **SpeechX** (Microsoft): Versatile speech generation using EnCodec

---

## 4. SoundStream (Google)

**Paper**: "SoundStream: An End-to-End Neural Audio Codec" (Zeghidour et al., 2021)

### Architecture
- **Type**: End-to-end neural audio codec (predecessor/parallel to EnCodec)
- **Encoder**: Fully convolutional, processes 24 kHz audio
  - Strided convolution blocks for temporal downsampling
  - Total downsampling factor: 320x (similar to EnCodec)
  - Produces 75 embeddings per second
- **Decoder**: Fully convolutional, mirrors encoder with transposed convolutions
- **Quantizer**: Residual Vector Quantization (RVQ)
  - Experiments with up to 80 quantizer layers
  - Typical: ~10-32 quantizers depending on bitrate
  - Codebook size: typically 1024 entries per quantizer
  - Efficiency example: at 3 kbps with 100 vectors/sec, only 5 RVQ layers with 320 entries each suffice vs. a single VQ requiring 1 billion entries
- **Discriminator**: Wave-based and STFT-based discriminators
- **Training**: Joint adversarial + reconstruction losses end-to-end

### Variable Bitrate via Quantizer Dropout
- **Key innovation**: Structured dropout of quantizer layers during training
- Randomly uses only first N_q quantizers (N_q sampled uniformly)
- Single model operates across **3 kbps to 18 kbps** without retraining
- Quality degrades gracefully as quantizers are removed

### Information Captured
- Full audio reconstruction: speaker identity, prosody, acoustics, semantic content
- Similar to EnCodec — no explicit semantic/acoustic separation
- Lower RVQ levels: coarse spectral envelope, pitch
- Higher RVQ levels: fine acoustic texture, harmonics

### Bitrate and Quality
- At 3 kbps: **outperforms Opus at 12 kbps** (3.2-4x more efficient)
- Approaches EVS quality at 9.6 kbps
- Real-time on smartphone CPUs
- Supports joint compression + enhancement (e.g., noise suppression with no added latency)

### Downstream Usage
- **AudioLM** (Google): SoundStream tokens serve as the acoustic token layer, generated conditioned on semantic tokens from w2v-BERT
- **AudioPaLM** (Google): Uses SoundStream for audio decoding stage (via SoundStorm for fast parallel decoding)
- **SoundStorm** (Google): Non-autoregressive parallel generation of SoundStream tokens, ~100x faster than AudioLM's autoregressive approach
- **Lyra V2**: Google's on-device speech codec using SoundStream at 3.2 kbps

---

## 5. Mimi (Kyutai / Moshi)

**Paper**: "Moshi: a speech-text foundation model for real-time dialogue" (Defossez et al., 2024)

### Architecture
- **Type**: Streaming neural audio codec with semantic-acoustic disentanglement
- **Encoder**: Convolutional encoder + **Transformer layers** (unlike pure-conv EnCodec/SoundStream)
  - Processes 24 kHz audio
  - Convolutional downsampling followed by Transformer self-attention layers
  - Total downsampling to **12.5 Hz** frame rate (significantly lower than EnCodec's 75 Hz)
- **Decoder**: Convolutional decoder + Transformer layers
- **Quantizer**: RVQ with up to 32 codebooks (limited to **8 codebooks** in Moshi deployment)
  - Codebook size: 2048 entries per codebook (larger than EnCodec's 1024)
- **Streaming**: Fully streaming with 80ms latency (matching frame size at 12.5 Hz)

### Semantic-Acoustic Disentanglement
- **First codebook**: Distilled from **WavLM** self-supervised representations
  - First RVQ level is trained to match WavLM features via knowledge distillation
  - Captures semantic/linguistic content similar to HuBERT tokens
- **Remaining codebooks** (2-8): Capture acoustic details — speaker identity, prosody, recording conditions
- This mirrors SpeechTokenizer's approach but at a much lower frame rate
- Training uses **only adversarial loss + feature matching** (no reconstruction L1/L2), similar to EBEN architecture

### Bitrate
- **1.1 kbps** (with 8 codebooks at 12.5 Hz)
- Extremely efficient — 12.5 Hz is close to text token rates (3-4 Hz)
- Outperforms SpeechTokenizer (50 Hz, 4 kbps) and SemantiCodec (50 Hz, 1.3 kbps)

### Reconstruction Quality
- Achieves "strong improvements in subjective quality despite its low bitrate"
- Outperforms existing non-streaming codecs at comparable bitrates
- The low frame rate is key to Moshi's low dialogue latency

### Downstream Usage
- **Moshi** (Kyutai): Full-duplex voice dialogue LLM
  - Mimi encodes both user and system audio streams
  - Temporal Transformer in Moshi generates Mimi tokens autoregressively
  - Low 12.5 Hz rate reduces autoregressive steps, enabling ~200ms total dialogue latency on L4 GPUs
  - System models 8 codebook streams for its own speech + 1 semantic stream for user's speech simultaneously

---

## 6. WavTokenizer

**Paper**: "WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Models" (Ji et al., 2024)

### Architecture
- **Type**: Single-quantizer neural audio codec, designed specifically for language model compatibility
- **Encoder**: 1D convolutional network
  - Initial convolution: C=32 channels, kernel_size=7
  - B=4 convolutional blocks with residual units and strided downsampling
  - Hidden dimension D=512, ELU activation
  - Stride configurations: (2, 4, 5, 8) for 320x compression or (4, 5, 5, 6) for 600x compression
  - Operates on 24 kHz audio
- **Decoder**: **Not a mirror of encoder** — uses inverse Fourier transform structure (inspired by Vocos)
  - ConvNeXt blocks process features
  - Outputs STFT magnitude and phase coefficients
  - Inverse FFT reconstructs waveform directly
  - Reduces aliasing artifacts from transposed convolutions
- **Quantizer**: **Single VQ codebook** (not RVQ)
  - Codebook size: **4096 entries** (2^12)
  - K-means initialization with 200 cluster centers
  - Forced activation strategy: randomly replaces unused codes to maintain high utilization
  - Single codebook eliminates the multi-codebook generation challenge

### Token Rate
- **40 tokens/second** with 600x compression (small model)
- **75 tokens/second** with 320x compression
- Dramatically lower than DAC (900 tokens/sec) or EnCodec (600 tokens/sec at 8 codebooks)

### Information Captured
- Designed to preserve **rich semantic information** within a single codebook
- Extended contextual windows (1-3 seconds) during training enhance semantic capture
- Attention module in decoder improves semantic modeling
- Speaker identity and acoustic details also preserved (single codebook must encode everything)

### Bitrate
- At 40 tokens/sec: ~0.48 kbps (40 x 12 bits)
- At 75 tokens/sec: ~0.9 kbps

### Reconstruction Quality
- **UTMOS**: 4.0486 at 0.9 kbps on LibriTTS test-clean (state-of-the-art)
- Surpasses DAC's 3.9097 despite using 75 vs. 900 tokens/sec
- PESQ: 2.3730, STOI: 0.9139
- Strong speaker embedding similarity preservation

### Training
- Multi-scale discriminator ensemble: MPD + MRD + MSD + complex STFT discriminator
- Hinge loss formulation
- Trained on data ranging from LibriTTS to 80,000-hour corpora

### Downstream Usage
- Built specifically for "audio language models such as GPT-4o"
- Single-codebook design means standard autoregressive LM can directly generate tokens
- No need for multi-codebook modeling strategies (delay patterns, NAR stages, etc.)

---

## 7. SpeechTokenizer

**Paper**: "SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models" (Zhang et al., 2023)

### Architecture
- **Type**: Encoder-decoder codec with semantically-guided RVQ
- **Encoder**: Based on EnCodec architecture
  - 1D convolution (C channels, kernel_size=7) followed by B residual convolutional blocks
  - **BiLSTM layers** (replacing EnCodec's LSTM) for enhanced temporal/semantic modeling
  - Strided convolutions for downsampling
- **Decoder**: Mirror of encoder with transposed convolutions
- **Quantizer**: RVQ with **8 quantizers**, **1024 entries** per codebook
- **Frame rate**: **50 Hz** (one frame per 20ms)
- **Sample rate**: 16 kHz monophonic

### Semantic-Acoustic Disentanglement via RVQ
- **RVQ-1 (first quantizer)**: Distilled from **HuBERT representations**
  - Uses averaged representations across all HuBERT layers (or specifically layer 9)
  - Distillation via dimension-wise cosine similarity maximization
  - Captures primarily semantic/linguistic content
  - Functions as a drop-in replacement for HuBERT K-means tokens
- **RVQ 2-8 (remaining 7 quantizers)**: Capture acoustic residuals
  - Speaker identity, timbre, prosody, recording conditions
  - "Supplements for the information lost by the first quantizer"

### Bitrate
- 8 codebooks x 10 bits x 50 Hz = **4 kbps**
- Semantic-only (first codebook): 500 bps

### Reconstruction Quality
- Performs comparably to EnCodec for speech reconstruction
- Strong performance on SLMTokBench benchmark (evaluating token suitability for language modeling)

### Downstream Usage
- **USLM** (Unified Speech Language Model):
  - AR model generates RVQ-1 tokens from phoneme sequences (semantic generation)
  - NAR model generates RVQ 2-8 tokens conditioned on RVQ-1 + acoustic prompt
  - Outperforms VALL-E on zero-shot TTS (lower WER, competitive speaker similarity)
- The hierarchical separation allows downstream models to:
  - Use only semantic tokens for understanding tasks
  - Use all tokens for high-fidelity synthesis
  - Weight semantic vs acoustic components per task

---

## 8. DAC (Descript Audio Codec)

**Paper**: "High-Fidelity Audio Compression with Improved RVQGAN" (Kumar et al., 2023)

### Architecture
- **Type**: Improved RVQGAN neural audio codec
- **Encoder**: Convolutional encoder with strided downsampling
  - **Snake activations** (periodic activations, better for audio than ReLU/ELU)
  - Convolutional blocks similar to EnCodec but with improved activations
- **Decoder**: Transposed convolutional decoder, also with Snake activations
- **Quantizer**: RVQ with **factorized codes** (improved codebook learning)
  - Codebook dimension: 8 (low-dimensional, then projected back)
  - This factorization improves codebook utilization significantly
  - Number of codebooks varies by sample rate:
    - 44.1 kHz: 9 codebooks
    - 24 kHz: 12 codebooks (reported in some configs)
    - 16 kHz: 12 codebooks
  - Codebook size: **1024 entries** per codebook
- **Training losses**: L1 reconstruction + mel-spectrogram loss + multi-scale adversarial loss

### Key Innovations over EnCodec
1. **Snake activations**: Periodic activation function better suited for audio signals
2. **Factorized codes**: Low-dimensional codebook vectors (dim=8) improve utilization and learning
3. **Improved adversarial training**: Better discriminator architecture
4. **Universal model**: Single model handles speech, music, and environmental sounds

### Bitrate and Compression
- **~90x compression** at 8 kbps for 44.1 kHz audio
- Compare: EnCodec achieves ~32x at 24 kbps for 48 kHz
- DAC achieves higher compression at higher fidelity sample rates

### Model Variants
| Sample Rate | Bitrate | Parameters | Domain |
|-------------|---------|------------|--------|
| 44.1 kHz    | 8 kbps  | —          | Universal |
| 24 kHz      | —       | —          | Universal |
| 16 kHz      | —       | —          | Universal |

### Information Captured
- Full audio signal: speaker identity, prosody, acoustics, semantic content
- No explicit semantic-acoustic separation (like EnCodec)
- Higher fidelity than EnCodec at comparable bitrates

### Reconstruction Quality
- State-of-the-art at time of release for universal audio compression
- Designed as a "drop-in replacement for EnCodec for all audio language modeling applications"
- Supports AudioLMs, MusicLMs, MusicGen, etc.

### Downstream Usage
- Direct replacement for EnCodec in any codec language modeling pipeline
- Used in various audio generation research
- 44.1 kHz support enables high-fidelity music generation

---

## 9. Whisper as Encoder (OpenAI)

**Paper**: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2023)

### Original Architecture
- **Type**: Encoder-decoder Transformer for ASR/translation
- **Encoder**: 32-layer Transformer (for large-v2/v3)
  - Input: 80-channel mel-spectrogram (25ms window, 10ms hop)
  - Two convolutional downsampling layers as stem
  - ~640M parameters (encoder only, large-v2)
  - 16 kHz input resampled audio
  - Produces continuous frame-level representations
- **Decoder**: Transformer decoder for text token generation
- **Training**: Supervised on 680,000 hours of weakly-labeled web audio

### Repurposing as Speech Encoder for LLMs
Whisper's encoder (discarding the decoder) has become a popular speech encoder for multimodal LLMs because:
1. It is trained on massive diverse data (robust to noise, accents, domains)
2. Its representations encode rich information beyond just text — including background noise, speaker characteristics, and even enough acoustic detail to recover original speech
3. Supervised training provides more structured representations than self-supervised models

### Integration Patterns

**Pattern 1: Direct Feature Projection (Qwen-Audio)**
- Uses Whisper-large-v2 encoder as the audio backbone
- Encoder output: continuous features at ~25 Hz (after internal downsampling) with additional pooling stride of 2, giving ~40ms per frame
- Features projected to LLM dimension via a single linear layer (or lightweight adapter)
- Fed as prefix to LLM (Qwen-7B) — audio features prepended to text token embeddings
- No discretization — fully continuous pipeline
- During multi-task pretraining: encoder weights are trained, LLM frozen
- During instruction fine-tuning: encoder frozen, LLM trained

**Pattern 2: Speech Embedding Prepending (LLaSM, BLSP)**
- Whisper encoder features directly prepended to text token embeddings
- The LLM learns to attend over audio features as if they were token embeddings
- Simple linear projection maps Whisper hidden dim to LLM hidden dim

**Pattern 3: Q-Former Bridge (SALMONN)**
- SALMONN uses dual encoders: Whisper (for speech) + BEATs (for non-speech audio)
- Q-Former (from BLIP-2) bridges audio features to LLM space
- Cross-attention between learned queries and audio features
- More parameter-efficient than full linear projection

### What Information Whisper Encoder Captures
- **Rich semantic information**: Trained for ASR, so linguistic content is primary
- **Paralinguistic information**: Emotion, speaker characteristics (sufficient to reconstruct speech)
- **Environmental information**: Background noise, recording conditions
- **Not explicitly discrete**: Continuous representations, ~1024 or 1280 dimensional per frame

### Advantages and Limitations as Speech Encoder
**Advantages:**
- No information loss from discretization
- Robust to noise and domain shifts (trained on massive data)
- Captures both semantic and acoustic information in a single continuous representation
- Well-supported, widely available

**Limitations:**
- Cannot be used for speech generation (no discrete tokens to generate)
- Requires separate speech synthesis pipeline for output
- Higher sequence lengths compared to discrete token approaches
- Encoder-only usage discards the ASR decoder's linguistic structure

### Downstream Usage
- **Qwen-Audio / Qwen2-Audio**: Whisper-large-v2 encoder + linear projection + Qwen LLM
- **SALMONN**: Whisper + BEATs encoders + Q-Former + Vicuna LLM
- **LLaSM**: Whisper encoder + projection + LLaMA
- **BLSP**: Whisper encoder + continuation training with LLM
- **Gazelle** (Tincans AI): Whisper encoder features as input to speech LLM

---

## 10. CosyVoice Codec Approach

**Papers**: "CosyVoice: A Scalable Multilingual Zero-shot Text-to-Speech Synthesizer" (Du et al., 2024), "CosyVoice 2" (Du et al., 2024)

### Architecture Overview
CosyVoice takes a fundamentally different approach — rather than using an unsupervised codec or self-supervised features, it derives **supervised semantic tokens** from a speech recognition model.

### CosyVoice 1: Supervised Semantic Tokens via ASR + VQ

**Speech Tokenizer:**
- Takes a **multilingual ASR encoder** (trained for speech recognition)
- Inserts **Vector Quantization (VQ)** into the encoder architecture
- The VQ layer produces discrete tokens that are explicitly aligned with text content
- This is the first attempt to use supervised speech tokens in TTS models
- Because the ASR model is multilingual, tokens transfer across languages

**Key insight:** Unlike HuBERT (self-supervised) or EnCodec (codec), these tokens are trained with explicit text supervision, providing:
- Stronger alignment between tokens and linguistic content
- Better content consistency in reconstruction
- Superior speaker similarity in zero-shot voice cloning

**System Pipeline:**
1. **Text -> Tokens**: LLM generates speech token sequences from text input (+ optional speaker prompt)
2. **Tokens -> Speech**: Conditional flow matching model synthesizes waveform from tokens
   - Flow matching replaces traditional vocoder approaches
   - Conditions on speaker embeddings for voice cloning

### CosyVoice 2: Finite Scalar Quantization (FSQ)

**Key improvements:**
- Replaces VQ with **Finite Scalar Quantization (FSQ)**
  - FSQ maps each scalar dimension to a finite set of values (e.g., {-1, 0, 1})
  - No codebook collapse issues (unlike VQ which can have dead codes)
  - Better codebook utilization
- Uses a **pre-trained LLM backbone** directly (rather than training from scratch)
- **Chunk-aware causal flow matching** enables both streaming and non-streaming synthesis in a single model
- Achieves "human-parity naturalness" with minimal response latency

### Information Captured
- **Primarily semantic**: Tokens derived from ASR model capture linguistic content
- **Text-aligned**: Explicit alignment between speech tokens and text
- **Speaker information**: Encoded separately via speaker embeddings, not in the tokens
- **Prosody**: Partially captured; flow matching model handles much of the prosodic realization

### Downstream Usage
- CosyVoice is used as a complete TTS system within Alibaba's **FunAudioLLM** framework
- Demonstrates that supervised tokens outperform unsupervised ones for TTS
- The LLM-based text-to-token generation enables in-context learning and zero-shot voice cloning
- Supports cross-lingual voice cloning (speak in a new language with same voice)

---

## 11. SNAC (Multi-Scale Neural Audio Codec)

**Paper**: "Multi-Scale Neural Audio Codec" (Siuzdak et al., 2024)

### Architecture
- **Type**: Neural audio codec with multi-scale (variable temporal resolution) RVQ
- **Core Innovation**: Unlike standard RVQ where all quantizers operate at the same frame rate, SNAC operates quantizers at **different temporal resolutions**
- **Encoder**: Convolutional encoder (similar to EnCodec/SoundStream/DAC family)
- **Decoder**: Convolutional decoder
- **Quantizer**: Hierarchical RVQ with variable frame rates per level

### Multi-Scale Temporal Structure
For a 1-second clip at 32 kHz:

| Level | Resolution | Tokens/sec | Information |
|-------|-----------|------------|-------------|
| 1 (Coarsest) | ~12 Hz | 12 | Broad structure, pitch contour |
| 2 | ~24 Hz | 24 | Spectral envelope |
| 3 | ~48 Hz | 48 | Finer detail |
| 4 (Finest) | ~96 Hz | 96 | High-frequency texture |

- Coarse tokens cover broader time spans (lower frequency)
- Fine tokens capture rapid acoustic changes (higher frequency)
- Total: 12 + 24 + 48 + 96 = **180 tokens per second** (at 32 kHz)

### Model Variants

| Model | Bitrate | Sample Rate | Parameters | Domain |
|-------|---------|-------------|-----------|--------|
| snac_24khz | 0.98 kbps | 24 kHz | 19.8M | Speech |
| snac_32khz | 1.9 kbps | 32 kHz | 54.5M | Music/SFX |
| snac_44khz | 2.6 kbps | 44 kHz | 54.5M | Music/SFX |

### Information Captured
- Full audio signal (like other codecs)
- Multi-scale decomposition naturally separates slow-varying features (coarse) from fast-varying features (fine)
- Coarse level is somewhat more "semantic" (captures broad temporal structure)
- Not explicitly disentangled like SpeechTokenizer/Mimi

### Key Advantage for LLMs
- Coarse tokens at ~12 Hz enable extended context modeling
- "With coarse tokens of ~10 Hz and a context window of 2048 you can effectively model consistent structure for ~3 minutes"
- The hierarchical structure allows different generation strategies for different levels:
  - AR generation for coarse tokens (few tokens, long context)
  - Parallel/NAR generation for fine tokens (many tokens, local context)

### Downstream Usage
- **Ichigo** (Homebrew Research): Open-source speech-to-speech LLM using SNAC for audio tokenization
- Used in research on hierarchical audio generation
- Open-source model weights and code available
- The multi-scale approach is particularly suited to iterative refinement generation strategies

---

## 12. Vocos

**Paper**: "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis" (Siuzdak, 2023)

### Architecture
- **Type**: Neural vocoder (not a codec) — converts acoustic features to waveforms
- **Core Innovation**: Generates **Fourier spectral coefficients** (magnitude + phase) directly instead of time-domain samples
- **Backbone**: **ConvNeXt** blocks as the primary feature processor
  - Much simpler than HiFi-GAN's multi-stream architecture
  - No upsampling needed (works in spectral domain)
- **Output Head**: Inverse Short-Time Fourier Transform (iSTFT)
  - Predicts STFT magnitude and instantaneous frequency (for phase)
  - Applies iSTFT to reconstruct waveform
  - Leverages efficient FFT algorithms

### Model Variants
| Variant | Training Data | Parameters | Input |
|---------|--------------|------------|-------|
| Mel-spectrogram decoder | LibriTTS (1M iter) | 13.5M | Mel spectrograms |
| EnCodec token decoder | DNS Challenge (2M iter) | 7.9M | EnCodec discrete codes |

### Performance
- **Speed**: Order of magnitude faster than time-domain vocoders (HiFi-GAN, BigVGAN)
- **Quality**: Matches state-of-the-art audio quality
- Supports variable EnCodec bandwidth: [1.5, 3.0, 6.0, 12.0] kbps

### Role in Speech LLM Pipelines
Vocos serves as a **replacement decoder** for codec-based systems:
1. An LLM generates discrete codec tokens (e.g., EnCodec codes)
2. Instead of using EnCodec's own decoder, Vocos converts tokens to waveform
3. **Higher quality reconstruction** than EnCodec's decoder, especially at low bitrates
4. Much faster inference due to spectral-domain generation

### Integration Examples
- **WavTokenizer**: Uses a Vocos-inspired decoder with inverse Fourier transform structure
- **Bark** (Suno): Can use Vocos as the final vocoder stage
- Can serve as the waveform reconstruction module in any pipeline that produces mel-spectrograms or codec tokens
- Commonly used to improve EnCodec reconstruction quality without retraining the codec

---

## 13. The Key Debate: Semantic vs Codec vs Continuous

### The Three Paradigms

#### A. Semantic Tokens (HuBERT K-means, w2v-BERT K-means)

**What they encode**: Linguistic/phonetic content, roughly corresponding to speech sounds
**What they lose**: Speaker identity, exact prosody, acoustic environment, recording quality

| Pros | Cons |
|------|------|
| Clean semantic separation — easy for LM to model meaning | Cannot reconstruct high-fidelity audio alone |
| Very low bitrate (~500 bps) | Requires separate acoustic model for synthesis |
| Token space is small (500-2000 tokens) | Loses speaker identity (need separate conditioning) |
| Proven for understanding tasks (ASR, etc.) | No single-stage generation possible |
| Close to text token rates | Prosody is largely lost |

**Used by**: AudioLM (first stage), SpeechGPT, GSLM, TWIST, pGSLM

#### B. Codec Tokens (EnCodec, SoundStream, DAC, SNAC RVQ)

**What they encode**: Everything — full audio reconstruction possible
**Challenge**: Multiple codebook streams, high total token count, semantic information entangled with acoustics

| Pros | Cons |
|------|------|
| Full reconstruction — single model for encode/decode | Multiple codebooks = complex generation (need AR+NAR, delay patterns, etc.) |
| Preserves speaker identity naturally | Higher token rates (75 Hz x 8 codebooks = 600 tokens/sec) |
| End-to-end trainable | Semantic content mixed with acoustics — harder for LM to isolate meaning |
| Multiple bitrate options | Each RVQ level is dependent on previous |
| Widely adopted | Codebook 1 is not cleanly "semantic" |

**Used by**: VALL-E, MusicGen, AudioGen, VoiceCraft, SpeechX

#### C. Continuous Features (Whisper encoder, HuBERT encoder output)

**What they encode**: Rich continuous representations, maximum information
**Challenge**: Cannot be directly generated by standard autoregressive LMs

| Pros | Cons |
|------|------|
| No information loss from quantization | Cannot generate with standard LM (not discrete) |
| Rich representation for understanding | Requires specialized output heads for generation |
| Robust (Whisper) or transferable (HuBERT) | Higher sequence lengths per second |
| Simpler architecture (just encoder + projector) | Only works for understanding, not generation (unless paired with codec) |

**Used by**: Qwen-Audio, Qwen2-Audio, SALMONN, LLaSM, Gazelle

#### D. Hybrid/Disentangled (SpeechTokenizer, Mimi, WavTokenizer, CosyVoice)

**What they encode**: Semantic content in first codebook/quantizer, acoustics in remaining ones
**Key innovation**: Best of both worlds — discrete tokens that are also semantically structured

| Pros | Cons |
|------|------|
| First codebook usable as semantic tokens | Still requires multi-stage generation for full quality |
| Remaining codebooks add acoustic fidelity | Distillation adds training complexity |
| Single codec for both understanding and generation | Quality of semantic separation depends on teacher model |
| Can trade off quality vs token count | Newer, less battle-tested |

**Used by**: USLM (SpeechTokenizer), Moshi (Mimi), CosyVoice, various emerging systems

### Key Tradeoffs

**1. Token Count vs Information Preservation**
- HuBERT: 50 tokens/sec, but semantic only
- WavTokenizer: 40-75 tokens/sec, full audio (extreme efficiency)
- EnCodec at 6kbps: 8 x 75 = 600 tokens/sec, full audio
- SNAC at 24kHz: ~180 tokens/sec, full audio
- Mimi: 8 x 12.5 = 100 tokens/sec, full audio

Lower token counts mean longer audio can fit in LLM context windows.

**2. Generation Complexity**
- Single-codebook (HuBERT, WavTokenizer): Direct AR generation, simple
- Multi-codebook (EnCodec, SoundStream): Need AR+NAR stages, or delay patterns, or separate models per codebook
- Continuous (Whisper): Cannot generate directly, need separate synthesis

**3. Understanding vs Generation**
- Semantic tokens excel at understanding (ASR, translation)
- Codec tokens excel at generation (TTS, voice cloning)
- Continuous features are understanding-only (unless paired with generation codec)
- Hybrid approaches (Mimi, SpeechTokenizer) aim to support both

**4. The AudioLM Insight**
AudioLM (Google) established the influential two-stage paradigm:
1. Generate semantic tokens (from w2v-BERT) to capture content and structure
2. Generate acoustic tokens (from SoundStream) conditioned on semantic tokens to add acoustic fidelity

This semantic-then-acoustic pipeline inspired SpeechTokenizer and Mimi to internalize both stages within a single codec.

**5. The AudioPaLM Insight**
AudioPaLM showed that:
- Audio tokens can be added to a text LLM's vocabulary (simple embedding extension)
- Pre-trained text model weights dramatically improve speech task performance
- Choice of tokenizer matters enormously: USM-v2 tokens >> w2v-BERT >> USM-v1 on downstream tasks
- About 6-8 audio tokens correspond to 1 text token at 25 Hz, creating redundancy

---

## 14. Summary Comparison Table

| System | Type | Frame Rate | Codebooks | Codebook Size | Bitrate | Semantic? | Full Recon? |
|--------|------|-----------|-----------|---------------|---------|-----------|-------------|
| HuBERT | Self-supervised | 50 Hz | 1 (K-means) | 500-2000 | ~500 bps | Yes | No |
| EnCodec | Codec (RVQ) | 75 Hz | 2-32 | 1024 | 1.5-24 kbps | No (mixed) | Yes |
| SoundStream | Codec (RVQ) | 75 Hz | ~10-32 | ~1024 | 3-18 kbps | No (mixed) | Yes |
| Mimi | Codec (RVQ+distill) | 12.5 Hz | 8 (in Moshi) | 2048 | 1.1 kbps | RVQ-1 yes | Yes |
| WavTokenizer | Codec (single VQ) | 40-75 Hz | 1 | 4096 | 0.48-0.9 kbps | Partially | Yes |
| SpeechTokenizer | Codec (RVQ+distill) | 50 Hz | 8 | 1024 | 4 kbps | RVQ-1 yes | Yes |
| DAC | Codec (RVQ) | varies | 9-12 | 1024 | 8 kbps | No (mixed) | Yes |
| Whisper (enc) | Supervised encoder | ~25-50 Hz | N/A (continuous) | N/A | N/A | Rich | No (understanding only) |
| CosyVoice | Supervised VQ/FSQ | ~25 Hz | 1 | varies | ~250 bps | Yes (supervised) | Via flow matching |
| SNAC | Codec (multi-scale RVQ) | 12-96 Hz | 4 | ~1024 | 0.98-2.6 kbps | Partially | Yes |
| Vocos | Vocoder | N/A | N/A | N/A | N/A | N/A | Yes (from features) |

---

## 15. Which Voice LLMs Use Which Approach

### Semantic Token Approach
| Model | Tokenizer | Notes |
|-------|-----------|-------|
| AudioLM (Google) | w2v-BERT K-means + SoundStream | Pioneered semantic-then-acoustic pipeline |
| SpeechGPT (Fudan) | HuBERT K-means (500 clusters) | Expands LLaMA vocab with speech tokens |
| GSLM (Meta) | HuBERT K-means (100-200 clusters) | Text-free spoken language model |
| TWIST (Meta) | HuBERT K-means | Warm-starts from text LLM |
| AudioPaLM (Google) | w2v-BERT or USM K-means (1024) | Extends PaLM-2 vocabulary |
| Spirit-LM (Meta) | HuBERT tokens + pitch/style tokens | Interleaves speech and text tokens |

### Codec Token Approach
| Model | Tokenizer | Notes |
|-------|-----------|-------|
| VALL-E (Microsoft) | EnCodec (8 codebooks) | AR for codebook 1, NAR for 2-8 |
| VALL-E 2 (Microsoft) | EnCodec | Improved with repetition-aware sampling |
| MusicGen (Meta) | EnCodec | Delay pattern for parallel codebook generation |
| VoiceCraft (MIT) | EnCodec | Speech editing with masked generation |
| Moshi (Kyutai) | Mimi (8 codebooks) | Full-duplex dialogue, semantic first codebook |
| Ichigo (Homebrew) | SNAC | Multi-scale generation |

### Continuous Feature Approach
| Model | Encoder | Notes |
|-------|---------|-------|
| Qwen-Audio (Alibaba) | Whisper-large-v2 | Linear projection to Qwen-7B |
| Qwen2-Audio (Alibaba) | Whisper-large-v3 | Updated audio encoder |
| SALMONN (ByteDance/CUHK) | Whisper + BEATs | Q-Former bridge to Vicuna |
| LLaSM | Whisper | Encoder + projection + LLaMA |
| Gazelle (Tincans) | Whisper | Understanding-focused |
| Seamless (Meta) | Custom encoder | Streaming translation |

### Hybrid/Supervised Token Approach
| Model | Tokenizer | Notes |
|-------|-----------|-------|
| CosyVoice (Alibaba) | Supervised VQ from ASR | LLM text-to-token + flow matching |
| CosyVoice 2 (Alibaba) | FSQ from ASR | Improved codebook utilization |
| GLM-4-Voice (Zhipu) | CosyVoice-style tokens | Uses supervised speech tokens |
| Mini-Omni | SpeechTokenizer-like | Streaming speech output |

### Emerging Trend: Single-Codebook Efficiency
The field is moving toward fewer tokens per second:
- **Mimi**: 12.5 Hz (100 total tokens/sec with 8 codebooks)
- **WavTokenizer**: 40 tokens/sec with single codebook
- **CosyVoice**: ~25 tokens/sec supervised semantic tokens
- **SNAC coarse**: 12 Hz at coarsest level

This contrasts with earlier approaches:
- **EnCodec at 6 kbps**: 600 tokens/sec
- **DAC at 8 kbps**: ~900 tokens/sec

The reduction in token count is critical for fitting longer audio into LLM context windows and reducing autoregressive generation latency.

---

## References

1. Hsu et al. "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units." IEEE/ACM TASLP 2021. arXiv:2106.07447
2. Defossez et al. "High Fidelity Neural Audio Compression." ICLR 2023. arXiv:2210.13438
3. Zeghidour et al. "SoundStream: An End-to-End Neural Audio Codec." IEEE/ACM TASLP 2022. arXiv:2107.03312
4. Defossez et al. "Moshi: a speech-text foundation model for real-time dialogue." Kyutai 2024. arXiv:2410.00927
5. Ji et al. "WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Models." 2024. arXiv:2408.16532
6. Zhang et al. "SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models." ICLR 2024. arXiv:2308.16692
7. Kumar et al. "High-Fidelity Audio Compression with Improved RVQGAN." NeurIPS 2023. arXiv:2306.06546
8. Radford et al. "Robust Speech Recognition via Large-Scale Weak Supervision." ICML 2023. arXiv:2212.04356
9. Du et al. "CosyVoice: A Scalable Multilingual Zero-shot Text-to-Speech Synthesizer." 2024. arXiv:2407.05407
10. Siuzdak et al. "Multi-Scale Neural Audio Codec." 2024. arXiv:2410.14411
11. Siuzdak. "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders." ICLR 2024. arXiv:2306.00814
12. Rubenstein et al. "AudioPaLM: A Large Language Model That Can Speak and Listen." Google 2023. arXiv:2306.12925
13. Borsos et al. "AudioLM: a Language Modeling Approach to Audio Generation." Google 2023. arXiv:2209.03143
14. Wang et al. "VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers." Microsoft 2023. arXiv:2301.02111
15. Chu et al. "Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models." Alibaba 2023. arXiv:2311.07919
