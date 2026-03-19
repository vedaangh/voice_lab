# Comprehensive Survey: TTS Model Releases & Papers (2024-2026)
## Focus: Breakthroughs Relevant to Voice LLM Pipelines

*Compiled March 2026*

---

## Table of Contents
1. [F5-TTS](#1-f5-tts)
2. [MaskGCT](#2-maskgct)
3. [CosyVoice 2 (& v1, v3)](#3-cosyvoice)
4. [E2 TTS](#4-e2-tts)
5. [Seed-TTS](#5-seed-tts)
6. [Zonos TTS](#6-zonos-tts)
7. [Mars5-TTS](#7-mars5-tts)
8. [Orpheus TTS](#8-orpheus-tts)
9. [OuteTTS](#9-outetss)
10. [Parler-TTS](#10-parler-tts)
11. [StyleTTS 2](#11-styletts-2)
12. [XTTS v2 (Coqui)](#12-xtts-v2)
13. [Amphion Toolkit](#13-amphion)
14. [FireRedTTS](#14-fireredtts)
15. [VALL-E 2](#15-vall-e-2)
16. [VALL-E R](#16-vall-e-r)
17. [NaturalSpeech 3](#17-naturalspeech-3)
18. [Matcha-TTS](#18-matcha-tts)
19. [Voicebox (Meta)](#19-voicebox)
20. [Spark-TTS](#20-spark-tts)
21. [Llasa](#21-llasa)
22. [Orpheus TTS (LLM-native)](#8-orpheus-tts)
23. [MegaTTS 3](#22-megatts-3)
24. [LLMVoX](#23-llmvox)
25. [Moshi](#24-moshi)
26. [SALAD](#25-salad)
27. [E1 TTS](#26-e1-tts)

---

## 1. F5-TTS
**"A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching"**
- **Paper:** arXiv:2410.06885 (Oct 2024)
- **Authors:** Yushen Chen et al., Shanghai Jiao Tong University / Cambridge
- **Code:** https://github.com/SWivid/F5-TTS

### Architecture
- **Type:** Fully non-autoregressive, flow matching
- **Backbone:** Diffusion Transformer (DiT) with adaLN-zero, 22 layers, 16 heads, 1024 dim
- **Text Processing:** ConvNeXt V2 blocks (4 layers, 512 dim) refine character-level text input before concatenation with speech. No phoneme alignment, no duration model, no text encoder needed.
- **Training Task:** Text-guided speech infilling (same as Voicebox) using Conditional Flow Matching with Optimal Transport (OT-CFM)
- **Audio Representation:** 100-dim log mel-filterbank at 24kHz, hop length 256. Uses Vocos vocoder for mel-to-waveform.
- **Parameters:** 335.8M (base); 158M (small)
- **Audio Codec:** None (operates directly on mel spectrograms, not discrete tokens)
- **Key Innovation - Sway Sampling:** Non-uniform inference-time flow step sampling (sway coefficient s=-1) that biases toward early flow steps. Universally applicable to any flow-matching model without retraining. Consistently improves WER, SIM, and UTMOS.
- **Position Embeddings:** RoPE for self-attention, convolutional position embedding for sequences, sinusoidal for flow step

### Zero-Shot Voice Cloning
- Excellent. Given an audio prompt + its transcript + target text, generates speech in the prompt speaker's voice.
- Simply estimates duration from character length ratios (no separate duration predictor).

### Streaming / Real-Time
- RTF = 0.15 at 16 NFE on RTX 3090 (greatly improved over prior diffusion TTS)
- RTF = 0.0394 with TensorRT-LLM optimization on L20 GPU
- Chunk-based inference supported via Gradio interface
- Latency: 253ms in client-server mode (2 concurrent)

### Quality Metrics (LibriSpeech-PC test-clean, 1127 samples)

| Model | Params | Data | WER(%)↓ | SIM-o↑ | RTF↓ |
|-------|--------|------|---------|--------|------|
| Ground Truth | - | - | 2.23 | 0.69 | - |
| **F5-TTS (32 NFE)** | 336M | 100K Multi | **2.42** | 0.66 | 0.31 |
| **F5-TTS (16 NFE)** | 336M | 100K Multi | 2.53 | 0.66 | **0.15** |
| E2 TTS (32 NFE) | 333M | 100K Multi | 2.95 | 0.69 | 0.68 |
| CosyVoice | ~300M | 170K Multi | 3.59 | 0.66 | 0.92 |
| FireRedTTS | ~580M | 248K Multi | 2.69 | 0.47 | 0.84 |
| Voicebox | 330M | 60K EN | 1.9 | 0.662 | 0.64 |
| NaturalSpeech 3 | 500M | 60K EN | 1.94 | 0.67 | 0.296 |
| MaskGCT | 1048M | 100K Multi | 2.634 | 0.687 | - |
| VALL-E 2 | - | 50K EN | 2.44 | 0.643 | 0.732 |

- Seed-TTS test-en: CMOS +0.31, SMOS 3.89
- Seed-TTS test-zh: CMOS +0.21, SMOS 3.83
- Training: 100K hours Emilia dataset (En+Zh), 1.2M updates on 8xA100 80G in ~1 week

### Voice LLM Pipeline Integration
- Ideal as the TTS backend in a voice LLM pipeline due to very low RTF (0.04-0.15)
- Mel-spectrogram approach means no dependency on neural codec quality
- Vocos vocoder is lightweight and fast
- Supports voice chat demo with Qwen2.5-3B-Instruct
- Code-switching capable (English/Chinese seamless)

### What Makes It Different
- Eliminates ALL complex components: no phoneme alignment, no duration model, no text encoder, no semantic codec
- ConvNeXt V2 provides text refinement that E2 TTS lacks, solving alignment robustness issues
- Sway Sampling is a universal contribution applicable to any flow-matching model
- Much faster training convergence than E2 TTS

---

## 2. MaskGCT
**"Masked Generative Codec Transformer"**
- **Paper:** arXiv:2409.00750 (Sept 2024, ICLR 2025)
- **Authors:** Wang et al., via Amphion project
- **Code:** https://github.com/open-mmlab/Amphion (MaskGCT module)
- **Model:** https://huggingface.co/amphion/MaskGCT

### Architecture
- **Type:** Fully non-autoregressive, mask-and-predict paradigm
- **Two-Stage Pipeline:**
  1. **T2S (Text-to-Semantic):** 695M params. Predicts semantic tokens from text + prompt semantic tokens. Uses W2V-BERT-2.0 for semantic token extraction.
  2. **S2A (Semantic-to-Acoustic):** 353M params. Converts semantic tokens to acoustic tokens using masked generation.
- **Total Parameters:** ~1048M
- **Audio Codec:** Custom acoustic codec (encoder + decoder), outputs at 24kHz
- **Semantic Model:** W2V-BERT-2.0 (multilingual SSL model)
- **Key Innovation:** During training, predicts masked semantic/acoustic tokens. During inference, generates all tokens in parallel at specified length -- no autoregressive sequential bottleneck.

### Zero-Shot Voice Cloning
- Excellent zero-shot capability using prompt audio + prompt semantic tokens as conditioning
- No explicit alignment between text and speech needed
- Eliminates phone-level duration prediction

### Streaming / Real-Time
- Non-autoregressive parallel generation is inherently fast
- Specific RTF not widely reported but expected to be competitive given NAR design

### Quality Metrics
- Outperforms SOTA zero-shot TTS on quality, similarity, and intelligibility benchmarks
- On LibriSpeech test-clean (40-sample subset): WER 2.634, SIM-o 0.687
- Trained on 100K hours Emilia dataset (50K En + 50K Zh)
- Supports 6 languages

### Voice LLM Pipeline Integration
- T2S + S2A two-stage design maps well to a modular pipeline
- Semantic tokens from W2V-BERT could potentially be shared with an LLM's speech understanding module
- High quality but larger model footprint (1B total)

### What Makes It Different
- First fully NAR TTS without ANY explicit alignment (no forced alignment, no duration prediction)
- Mask-and-predict paradigm borrowed from masked language models (BERT-style) applied to speech
- Two-stage semantic-then-acoustic decomposition provides modularity

---

## 3. CosyVoice (v1, v2, v3)
**Alibaba/FunAudioLLM**
- **CosyVoice v1 Paper:** 2024
- **CosyVoice 2 Paper:** arXiv:2412.10117 (Dec 2024)
- **CosyVoice 3 (Fun-CosyVoice 3.0):** 2025
- **Code:** https://github.com/FunAudioLLM/CosyVoice

### Architecture (v1)
- **Type:** Two-stage: AR text-to-token LLM + flow matching token-to-speech
- **Backbone:** LLM-based text-to-token model (~300M params)
- **Semantic Tokens:** Supervised discrete speech tokens
- **Flow Matching:** Token-to-waveform generation via conditional flow matching
- **Training Data:** 170K hours multilingual speech

### CosyVoice 2 Changes (Dec 2024)
- **FSQ (Finite Scalar Quantization):** Replaces standard VQ to improve codebook utilization
- **Chunk-Aware Causal Flow Matching:** Enables BOTH streaming and non-streaming synthesis within a single unified model
- **Simplified LLM backbone:** Allows direct use of a pre-trained LLM as the backbone (no custom architecture needed)
- **Streaming Latency:** Minimal response latency with "virtually lossless synthesis quality" in streaming mode
- **Human-parity naturalness** claimed

### CosyVoice 3.0 (Fun-CosyVoice, 2025)
- 9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
- 18+ Chinese dialects
- Chinese CER: 0.81%, English WER: 1.68%
- Speaker similarity: 77.4% (Zh), 69.5% (En)
- Streaming latency as low as 150ms
- Instruction-based control (language, dialect, emotion, speed, volume)
- vLLM support, TensorRT-LLM (4x speedup), RL-based refinement
- Pronunciation inpainting via Pinyin/CMU phonemes

### Zero-Shot Voice Cloning
- Excellent cross-lingual zero-shot cloning

### Voice LLM Pipeline Integration
- Designed explicitly for multi-modal LLM applications
- Streaming-first design with 150ms latency makes it ideal for real-time voice agents
- Direct LLM backbone compatibility in v2 means LLM and TTS can share weights
- GRPC/FastAPI server deployment ready

### What Makes It Different
- FSQ for better codebook utilization (v2)
- Unified streaming/non-streaming in single model (v2)
- Most production-ready open-source TTS for voice LLM pipelines

---

## 4. E2 TTS
**"Embarrassingly Easy Text-to-Speech"**
- **Paper:** arXiv:2406.18009 (June 2024, SLT 2024)
- **Authors:** Microsoft

### Architecture
- **Type:** Fully non-autoregressive, flow matching
- **Approach:** Converts text to character sequence with filler tokens, uses flow matching to generate mel spectrograms via audio infilling task
- **Key Simplification:** No duration model, no grapheme-to-phoneme, no monotonic alignment search
- **Backbone:** Flat U-Net Transformer

### Quality
- Achieves human-level naturalness, SOTA speaker similarity and intelligibility
- Comparable to or surpasses Voicebox and NaturalSpeech 3
- However: slow convergence and robustness issues (addressed by F5-TTS)

### Voice LLM Pipeline Integration
- Conceptually clean but F5-TTS is the practical successor with better robustness and speed
- E2 TTS pioneered the filler-token approach that F5-TTS refined

---

## 5. Seed-TTS
**ByteDance**
- **Paper:** arXiv:2406.02430 (June 2024)

### Architecture
- **Type:** Family of large-scale AR models + a DiT-based NAR variant (Seed-TTS_DiT)
- **Scale:** Trained on "several million hours" of data (orders of magnitude larger than others)
- **AR variant:** Foundation model for speech generation with in-context learning
- **DiT variant:** Non-autoregressive, end-to-end, no pre-estimated phoneme durations
- **Advanced Techniques:** Self-distillation for speech factorization, reinforcement learning for robustness/similarity/controllability

### Zero-Shot Voice Cloning
- "Virtually indistinguishable from human speech" in speaker similarity and naturalness
- Matches ground truth human speech quality

### Quality
- Best-in-class on Seed-TTS benchmarks (test-en, test-zh)
- Controllable emotion, diverse speech for in-the-wild speakers
- Not open-source (proprietary ByteDance system)

### Voice LLM Pipeline Integration
- Sets the quality ceiling for zero-shot TTS
- Proprietary, but the Seed-TTS eval benchmarks are open and widely used
- DiT variant shows NAR approaches can match AR quality at scale

---

## 6. Zonos TTS
**Zyphra (Jan 2025)**
- **Model:** https://huggingface.co/Zyphra/Zonos-v0.1-transformer
- **Code:** https://github.com/Zyphra/Zonos

### Architecture
- **Type:** Autoregressive token prediction
- **Backbone:** Transformer or Hybrid backbone, two variants:
  - `Zonos-v0.1-transformer`
  - `Zonos-v0.1-hybrid`
- **Audio Codec:** DAC (Descript Audio Codec) for tokenization and reconstruction
- **Text Processing:** eSpeak for normalization and phonemization
- **Output:** Native 44kHz (unusually high for TTS)
- **Training Data:** 200K+ hours multilingual speech

### Conditioning System (Key Differentiator)
- **Speaker Embedding Conditioning:** Creates speaker embedding from 10-30s reference audio
- **Emotion Control:** Fine-grained control over happiness, anger, sadness, fear
- **Quality/Pitch/Rate Control:** Adjustable speaking rate, pitch variation, audio quality, max frequency
- **Audio Prefix Mode:** Concatenate audio prefix for behaviors hard to replicate from embeddings alone (e.g., whispering)

### Zero-Shot Voice Cloning
- Yes, from 10-30 second reference audio via speaker embeddings

### Streaming / Real-Time
- ~2x real-time on RTX 4090
- 6GB+ VRAM requirement

### Languages
- English, Japanese, Chinese, French, German

### Voice LLM Pipeline Integration
- Emotion conditioning makes it valuable for expressive voice agents
- 44kHz output is high quality but may need downsampling for some pipelines
- Apache 2.0 license (fully open)
- Speaker embedding approach is efficient for multi-speaker scenarios

### What Makes It Different
- Explicit emotion and quality conditioning knobs (not just style transfer)
- DAC codec at 44kHz is higher fidelity than typical 16-24kHz systems
- Hybrid backbone option

---

## 7. Mars5-TTS
**Camb.ai**
- **Code:** https://github.com/camb-ai/mars5-tts

### Architecture
- **Type:** Two-stage AR + NAR pipeline
- **AR Component:** ~750M params (fp16), autoregressive transformer generating coarse Encodec speech features
- **NAR Component:** ~450M params (fp16), multinomial DDPM refining remaining codebook values
- **Audio Codec:** Encodec (L0-L3 codebook values)
- **Text:** Byte-pair encoded
- **Output:** 24kHz

### Zero-Shot Voice Cloning
- **Shallow Clone:** Fast, requires 1-12s reference audio (optimal ~6s)
- **Deep Clone:** Higher quality, requires reference audio + transcript, computationally slower
- **Prosodic Control:** Add commas for pauses, capitals for emphasis

### Streaming
- No streaming documented; generates complete sequences

### Quality
- English-only model
- License: AGPL 3.0

### Voice LLM Pipeline Integration
- Two-stage design adds latency; better for offline/batch use
- Deep clone mode is unique for high-quality personalization when transcripts are available

---

## 8. Orpheus TTS
**Canopy Labs (2025)**
- **Model:** https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod
- **Code:** https://github.com/canopyai/Orpheus-TTS

### Architecture
- **Type:** LLM-native speech synthesis (Speech-LLM)
- **Backbone:** Llama-3.2-3B-Instruct (finetuned)
- **Audio Codec:** SNAC (inferred from token structure)
- **Parameters:** 3B
- **Training Data:** 100K+ hours English speech
- **Sequence Length:** 8192 tokens

### Emotion & Expression Tags (Key Differentiator)
Inline emotive markup: `<laugh>`, `<chuckle>`, `<sigh>`, `<cough>`, `<sniffle>`, `<groan>`, `<yawn>`, `<gasp>`

### Zero-Shot Voice Cloning
- Yes, without prior fine-tuning

### Streaming / Real-Time
- **~200ms streaming latency** (reducible to ~100ms with input streaming)
- Suitable for real-time voice agents

### Voice Options
- 8 English voices: tara, leah, jess, leo, dan, mia, zac, zoe
- Multilingual models with language-specific voices

### Fine-Tuning
- High quality results after ~50 examples
- Optimal results ~300 examples per speaker
- Recommendation: use natural (not synthetic) speech data

### Voice LLM Pipeline Integration
- MOST DIRECT integration path: same Llama backbone as many LLMs
- Can potentially share model weights between language understanding and speech generation
- vLLM framework support for fast inference
- Sub-200ms latency enables real-time conversation
- Emotion tags enable expressive voice agents
- Apache 2.0 license

### What Makes It Different
- LLM-native: leverages Llama's semantic reasoning for prosody
- Emotion markup tags are unique and intuitive
- Claims to surpass closed-source SOTA in naturalness
- Extremely low fine-tuning data requirements

---

## 9. OuteTTS
- **Code:** https://github.com/edwko/OuteTTS

### Architecture
- **Type:** LLM-based, interface approach (no external adapters)
- **Backbone:** Llama-based, two sizes:
  - 1B (Llama-OuteTTS-1.0-1B)
  - 0.6B (OuteTTS-1.0-0.6B)
- **Audio Codec:** DAC (Descript Audio Codec) for reconstruction + WavTokenizer for tokenization
- **Sampling:** Requires 64-token recent window for repetition penalties; temp=0.4, rep_penalty=1.1, top_k=40, top_p=0.9

### Zero-Shot Voice Cloning
- Reference audio encoded into speaker profile
- Inherits referenced speaker's emotion, style, and accent

### Streaming / Real-Time
- Sub-1.0 RTF on NVIDIA L40S for batched inference
- Max generation: ~42s (~8192 tokens per window), quality peaks ~7000 tokens
- Backend agnostic: llama.cpp, ExLlamaV2, VLLM, 8+ inference engines

### Voice LLM Pipeline Integration
- Single API across 8+ backends is attractive for production
- Multilingual with accent preservation
- Lightweight models (0.6B-1B) suitable for edge deployment

### What Makes It Different
- No external adapters -- pure interface-based design
- Backend agnosticity is unique (single API for llama.cpp, VLLM, ExLlama, etc.)
- Explicit documentation of DAC reconstruction limitations (transparent about lossy codec)

---

## 10. Parler-TTS
**HuggingFace (2024)**
- **Model:** https://huggingface.co/parler-tts/parler-tts-large-v1
- **Code:** https://github.com/huggingface/parler-tts
- **Paper basis:** arXiv:2402.01912 ("Natural language guidance of high-fidelity TTS with synthetic annotations")

### Architecture
- **Type:** Conditional generation (text-described TTS)
- **Parameters:** 2.2B
- **Training Data:** 45,000 hours (MLS, LibriTTS-R)
- **License:** Apache 2.0

### Text-Described Voice Control (Key Differentiator)
Instead of reference audio, control voice via natural language:
- Gender, speaking rate, pitch, reverberation, audio quality, expressiveness
- 34 named speakers for reproducible generation (Jon, Lea, Gary, Jenna, Mike, Laura, etc.)
- Example: "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality."

### Zero-Shot Voice Cloning
- Not traditional cloning. Instead, creates voices from TEXT DESCRIPTIONS
- Named speakers provide consistency

### Streaming
- Supported via inference optimizations (SDPA, torch.compile, batching)

### Voice LLM Pipeline Integration
- Unique for scenarios where voice DESCRIPTIONS are more practical than reference audio
- An LLM could generate voice descriptions programmatically
- Data-Speech toolkit enables annotating any speech dataset with descriptions
- 2.2B params is relatively large

### What Makes It Different
- Only TTS where voice characteristics are controlled by natural language text
- No reference audio needed at all
- Pairs naturally with LLM-generated voice descriptions

---

## 11. StyleTTS 2
- **Paper:** arXiv:2306.07691 (NeurIPS 2023, still widely used in 2024-2025)
- **Code:** https://github.com/yl4579/StyleTTS2

### Architecture
- **Type:** Style diffusion + adversarial training
- **Key Components:**
  - Models speech style as latent random variable through diffusion models
  - WavLM-based SLM discriminator (pre-trained speech LM as adversary)
  - Differentiable duration modeling for end-to-end training
  - 24kHz output
- **No reference speech needed** for style generation (diffusion generates appropriate style from text alone)

### Zero-Shot Voice Cloning
- When trained on LibriTTS, outperforms prior public models for zero-shot speaker adaptation

### Quality
- **Surpasses human recordings** on LJSpeech (single-speaker)
- **Matches human recordings** on VCTK (multi-speaker)
- First publicly available model to claim human-level quality

### StyleTTS 3
- **No StyleTTS 3 has been announced** as of March 2026

### Voice LLM Pipeline Integration
- Mature, well-tested model with strong community
- Style diffusion approach is complementary to LLM-based systems
- Good baseline for quality comparison

---

## 12. XTTS v2
**Coqui AI**
- **Model:** https://huggingface.co/coqui/XTTS-v2
- **Code:** https://github.com/coqui-ai/TTS

### Architecture
- **Type:** End-to-end, GPT-based backbone
- **Output:** 24kHz
- **Key Feature:** Improved speaker conditioning over v1

### Languages (17)
English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi

### Zero-Shot Voice Cloning
- Requires only 6-second audio clip
- Cross-language voice cloning
- Emotion and style transfer via cloning
- Multiple speaker references and interpolation

### Streaming
- <200ms latency streaming supported
- Production-ready deployment

### Quality
- 7M+ monthly downloads on HuggingFace (massive adoption)
- Fine-tuning supported with example recipes
- License: CPML (Coqui Public Model License, somewhat restrictive)

### Voice LLM Pipeline Integration
- Most widely deployed open TTS model
- 17-language coverage is broadest among open models
- Low-latency streaming suits voice agents
- Note: Coqui AI company shut down; community-maintained

---

## 13. Amphion
**Open-Source Audio Toolkit**
- **Code:** https://github.com/open-mmlab/Amphion

### Scope
Not a single model but a comprehensive toolkit supporting:
- **TTS Models:** FastSpeech2, VITS, VALL-E, NaturalSpeech2, Jets, MaskGCT, Vevo-TTS
- **Voice Conversion:** Vevo, FACodec, Noro
- **Vocoders:** HiFi-GAN, MelGAN, WaveGlow, DiffWave
- **Evaluation:** F0, energy, intelligibility, spectrogram distortion, speaker similarity

### Recent 2025 Updates
- **DualCodec:** Low-frame-rate neural audio codec with semantic enhancement
- **Vevo 1.5:** Unified speech and singing generation
- **Metis:** Foundation model for zero-shot TTS, VC, and speech enhancement
- **Emilia-Large:** Dataset expanded to 200K+ hours

### Voice LLM Pipeline Integration
- Best toolkit for benchmarking and evaluating TTS models
- Standardized evaluation metrics
- MaskGCT is available through Amphion
- MIT-licensed evaluation tools

---

## 14. FireRedTTS
**FireRedTeam (2024)**
- **Code:** https://github.com/FireRedTeam/FireRedTTS

### Architecture
- **Type:** Autoregressive LLM-empowered, with optional flow matching decoder
- **Two decoder options:**
  - Acoustic LLM decoder (AR style)
  - Flow matching decoder (added May 2025)
- **Audio Codec:** BigCodec / Encodec (causal convolution)
- **Parameters:** ~580M total (~400M AR text-to-semantic + ~180M token-to-waveform)
- **Training Data:** 248K hours labeled speech
- **Output:** 24kHz WAV

### Zero-Shot Voice Cloning
- Yes, with 3-10s reference audio + accurate transcription

### Quality (from F5-TTS comparison)
- LibriSpeech-PC WER: 2.69% (good intelligibility)
- SIM-o: 0.47 (weak speaker similarity -- significantly below competitors)
- RTF: 0.84

### Languages
- Currently Chinese (zh) only with text normalization

### Voice LLM Pipeline Integration
- Primarily for Chinese speech applications
- Flow matching decoder option (May 2025) modernizes the architecture
- Academic-use restriction limits commercial deployment

### What Makes It Different
- Draws from Tortoise-TTS, XTTS-v2, Fish-Speech lineage
- Speaker verification via ECAPA-TDNN (SpeechBrain)
- HuBERT-based acoustic modeling

---

## 15. VALL-E 2
**Microsoft (June 2024)**
- **Paper:** arXiv:2406.05370

### Architecture
- **Type:** Neural codec language model (AR)
- **Key Innovation 1 - Repetition Aware Sampling:** Refines nucleus sampling by considering token repetition in decoding history. Stabilizes decoding and eliminates infinite loop issues.
- **Key Innovation 2 - Grouped Code Modeling:** Organizes codec codes into groups to reduce sequence length, increasing inference speed and handling long sequences.

### Quality
- **First system to achieve human parity** on LibriSpeech and VCTK for zero-shot TTS
- Enhanced robustness, naturalness, speaker similarity
- Handles complex/repetitive sentences well
- LibriSpeech test-clean: WER 2.44%, SIM-o 0.643, RTF 0.732

### Zero-Shot Voice Cloning
- Yes, via in-context learning with audio prompt

### Voice LLM Pipeline Integration
- Sets theoretical ceiling for codec LM approach
- Proprietary (Microsoft), not open-source
- Grouped code modeling idea transferable to other codec LM systems

---

## 16. VALL-E R
**Microsoft (June 2024)**
- **Paper:** arXiv:2406.07855

### Architecture
- **Type:** Neural codec language model with monotonic alignment
- **Key Innovation 1 - Phoneme Monotonic Alignment:** Constrains acoustic tokens to match associated phonemes for precise alignment
- **Key Innovation 2 - Codec-Merging:** Downsamples discrete codes in shallow quantization layers to accelerate decoding
- **Related:** VALL-T (arXiv:2401.14321, ICASSP 2025) adds shifting relative position embeddings for monotonic alignment, achieving 28.3% WER reduction vs. prior decoder-only models

### Quality
- WER approaching ground truth
- Inference time reduced by 60%+
- Mitigates typos, omissions, and repetitions

### Voice LLM Pipeline Integration
- Codec-merging technique is widely applicable for speeding up any codec LM
- Monotonic alignment improves robustness critical for production systems

---

## 17. NaturalSpeech 3
**Microsoft (March 2024)**
- **Paper:** arXiv:2403.03100

### Architecture
- **Type:** Factorized diffusion model
- **Key Innovation - Factorized Codec:** Neural codec with factorized vector quantization (FVQ) disentangles speech into 4 subspaces:
  1. Content (linguistic)
  2. Prosody (intonation, rhythm)
  3. Timbre (voice identity)
  4. Acoustic details (fine-grained)
- **Factorized Diffusion:** Separate diffusion model generates attributes in each subspace following respective prompts
- **Parameters:** 1B
- **Training Data:** 200K hours

### Quality
- Outperforms SOTA on quality, similarity, prosody, intelligibility
- On-par quality with human recordings
- LibriSpeech (40-sample): WER 1.94%, SIM-o 0.67, RTF 0.296

### Zero-Shot Voice Cloning
- Yes, excellent via disentangled timbre subspace

### Voice LLM Pipeline Integration
- Disentangled representations could allow independent control of content/prosody/timbre/acoustics
- 1B params and 200K hours training sets high resource bar
- Not open-source (Microsoft)

---

## 18. Matcha-TTS
**ICASSP 2024**
- **Code:** https://github.com/shivammehta25/Matcha-TTS

### Architecture
- **Type:** Non-autoregressive, optimal transport conditional flow matching (OT-CFM)
- **Structure:** Encoder-decoder; encoder processes text, decoder generates mel spectrograms via ODE-based sampling
- **ODE Solver:** Euler solver, configurable steps (default 5 NFE -- very fast)
- **Key Innovation:** Replaces traditional diffusion denoising with OT-CFM for dramatically fewer sampling steps

### Quality
- "Highly natural" speech
- Compact memory footprint
- Configurable speaking rate and sampling temperature
- Published at ICASSP 2024

### Streaming
- ONNX export supported for cross-platform deployment
- Very fast inference due to low NFE count

### Voice LLM Pipeline Integration
- Extremely lightweight and fast (5 NFE vs. 16-32 for F5-TTS)
- Good for latency-critical applications
- Less zero-shot capability than newer models
- Precursor to the flow-matching approach used by F5-TTS, CosyVoice, etc.

---

## 19. Voicebox
**Meta (2023, still influential)**

### Architecture
- **Type:** Non-autoregressive, flow matching, infilling task
- **Parameters:** 330M
- **Training Data:** 60K hours English
- **Key Innovation:** Text-guided speech infilling with flow matching. The infilling formulation (predict masked speech given surrounding audio + full text) became the basis for E2 TTS and F5-TTS.
- **Frame-wise phoneme alignment** used (unlike E2 TTS/F5-TTS which removed this)

### Quality
- LibriSpeech: WER 1.9-2.03%, SIM-o 0.64-0.662, RTF 0.64
- Set the standard for flow-matching TTS

### Voice LLM Pipeline Integration
- Established the speech infilling paradigm now used by F5-TTS
- Not open-source (Meta)
- Conceptual ancestor of modern flow-matching TTS

---

## 20. Spark-TTS
**SparkAudio (March 2025, ACL 2025)**
- **Paper:** arXiv:2503.01710
- **Code:** https://github.com/SparkAudio/Spark-TTS

### Architecture
- **Type:** LLM-based with novel BiCodec
- **Backbone:** Qwen2.5 LLM (no separate flow matching or diffusion)
- **Key Innovation - BiCodec:** Single-stream speech codec decomposing speech into:
  1. Low-bitrate semantic tokens (linguistic content)
  2. Fixed-length global tokens (speaker attributes)
- **Model Size:** 0.5B (Spark-TTS-0.5B)
- **Chain-of-thought generation:** Methodical token generation

### Controllable Generation (Key Differentiator)
- **Coarse-grained:** Gender, speaking style
- **Fine-grained:** Precise pitch values, speaking rate
- Surpasses limitations of reference-based synthesis

### Quality
- SOTA zero-shot voice cloning
- RTF on L20 GPU: 0.1362 (1 concurrent), 0.0704 (4 concurrent)
- Latency: 876ms (1 concurrent), 921ms (2 concurrent)

### Dataset
- **VoxBox:** 100K hours with comprehensive attribute annotations (released with paper)

### Languages
- Chinese, English (multilingual, code-switching)

### Voice LLM Pipeline Integration
- Qwen2.5 backbone means potential weight sharing with Qwen-based LLMs
- BiCodec's semantic/speaker decomposition is architecturally elegant
- Small model size (0.5B) enables efficient deployment
- Directly generates audio from LLM-predicted codes (no separate vocoder or flow matching)

### What Makes It Different
- BiCodec uniquely separates content (variable-length semantic) from identity (fixed-length global)
- Only model using chain-of-thought for TTS generation
- Qwen2.5 alignment enables LLM reasoning for speech

---

## 21. Llasa
**Feb 2025**
- **Paper:** arXiv:2502.04128

### Architecture
- **Type:** Single Transformer, fully LLM-aligned (Llama-style)
- **Key Innovation:** Single-layer vector quantizer (VQ) codec (vs. multi-layer RVQ in most systems)
- **Model Sizes:** 1B, 3B, 8B parameters
- **No multi-stage pipeline:** No separate diffusion or flow matching model needed

### Scaling Properties
- **Train-time scaling:** Larger models consistently produce more natural speech with better prosody
- **Inference-time scaling:** Uses speech understanding models as verifiers during search to improve emotional expressiveness, timbre consistency, content accuracy

### Quality
- Improvements in naturalness, prosody accuracy, emotional expressiveness, timbre consistency
- Inference-time compute scaling is novel for TTS

### Voice LLM Pipeline Integration
- Directly uses Llama architecture -- maximum compatibility with LLM ecosystems
- Single VQ codec simplifies the audio representation
- 1B-8B size range allows quality/speed tradeoff
- Inference-time scaling (using verifiers) is analogous to LLM best-of-N or reward-model guided generation
- Open checkpoints and training code

### What Makes It Different
- Simplest architecture: single Transformer + single VQ layer
- First to demonstrate inference-time compute scaling for TTS
- Llama-aligned means standard LLM tooling (vLLM, etc.) works directly

---

## 22. MegaTTS 3
**Feb 2025**
- **Paper:** arXiv:2502.18924

### Architecture
- **Type:** Latent Diffusion Transformer (DiT) with sparse alignment
- **Key Innovation 1 - Sparse Alignment:** Provides sparse alignment boundaries that reduce alignment difficulty without limiting search space (middle ground between no alignment and forced alignment)
- **Key Innovation 2 - Piecewise Rectified Flow:** Accelerates generation -- produces one-minute speech with only 8 sampling steps
- **Key Innovation 3 - Multi-Condition CFG:** Enables flexible accent intensity control

### Quality
- SOTA zero-shot TTS results
- One-minute speech generation in 8 steps is remarkably efficient
- Handles hard sentences robustly

### Voice LLM Pipeline Integration
- 8-step generation for minute-long speech is breakthrough efficiency
- Sparse alignment balances robustness and naturalness
- Accent control useful for multilingual voice agents

---

## 23. LLMVoX
**March 2025**
- **Paper:** arXiv:2503.04724

### Architecture
- **Type:** LLM-agnostic autoregressive streaming TTS
- **Parameters:** Only 30M (extremely lightweight)
- **Key Innovation:** Multi-queue token streaming for seamless infinite-length dialogues
- **Design:** Plug-and-play with ANY LLM backbone (does not modify the base LLM)

### Quality
- Significantly lower WER than competing speech-enabled LLMs
- Comparable latency and UTMOS to alternatives
- Low CER on Arabic (language generalization demo)

### Streaming
- Yes, designed for streaming from the ground up

### Voice LLM Pipeline Integration
- **IDEAL for voice LLM pipelines:** 30M parameter add-on to any existing LLM
- Does not degrade base LLM capabilities (a common problem with speech-enabled LLMs)
- Works with Vision-Language Models too (omni-modal)
- Infinite-length dialogue support

### What Makes It Different
- Smallest TTS model in this survey (30M)
- Truly LLM-agnostic (works with any LLM)
- Preserves base LLM quality while adding voice
- Multi-queue streaming for infinite dialogues

---

## 24. Moshi
**Kyutai Labs (Sept 2024)**
- **Paper:** arXiv:2410.00037

### Architecture
- **Type:** Speech-text foundation model, full-duplex
- **Key Innovation - Inner Monologue:** Predicts time-aligned text tokens as prefix to audio tokens, enabling simultaneous: improved linguistic quality, streaming ASR, and TTS
- **Audio:** Neural audio codec with residual quantizer
- **Parallel Streams:** Maintains separate speech streams for user and system utterances
- **No explicit turn-taking:** Eliminates VAD -> ASR -> LLM -> TTS cascade

### Streaming / Real-Time
- Theoretical latency: 160ms
- Practical latency: 200ms
- **First real-time full-duplex spoken LLM**

### Voice LLM Pipeline Integration
- Not a TTS model per se, but a UNIFIED speech-text LLM
- Represents the end state of voice LLM pipeline evolution: everything in one model
- Eliminates the entire cascade pipeline
- Open source (Kyutai Labs)

### What Makes It Different
- Full-duplex: listens and speaks simultaneously
- Inner monologue provides text reasoning while generating speech
- Most integrated speech-LLM architecture

---

## 25. SALAD
**Oct 2024, ASRU 2025**
- **Paper:** arXiv:2410.16048

### Architecture
- **Type:** Autoregressive with per-token latent diffusion
- **Key Innovation:** Each autoregressive step uses a diffusion process to refine and predict continuous speech representations (not discrete tokens)
- **Operates on continuous representations** rather than discrete codec tokens

### Quality
- Superior intelligibility vs. baselines
- Speech quality matching ground truth
- Speaker similarity comparable to reference

### What Makes It Different
- Hybrid AR + per-token diffusion on continuous features
- Avoids information loss from discretization

---

## 26. E1 TTS
**Sept 2024**
- **Paper:** arXiv:2409.09351

### Architecture
- **Type:** Non-autoregressive, single-pass
- **Key Innovation:** Denoising diffusion pretraining + distribution matching distillation
- **Result:** Only ONE neural network evaluation per utterance (single-step generation)

### Quality
- Naturalness and speaker similarity comparable to strong baselines
- Extreme efficiency: single forward pass

### Voice LLM Pipeline Integration
- Single-step generation is fastest possible inference
- Good for ultra-low-latency requirements

---

## Summary: Architecture Taxonomy

### By Generation Paradigm
| Paradigm | Models |
|----------|--------|
| **Autoregressive (AR)** | Seed-TTS (AR), VALL-E 2, VALL-E R, CosyVoice (T2T stage), FireRedTTS, Mars5 (stage 1), XTTS v2, Orpheus, OuteTTS, Spark-TTS, Llasa, Zonos, SALAD |
| **Non-Autoregressive (NAR)** | F5-TTS, MaskGCT, E2 TTS, NaturalSpeech 3, Matcha-TTS, Voicebox, E1 TTS, MegaTTS 3, Seed-TTS DiT |
| **Hybrid AR+NAR** | Mars5 (AR+DDPM), CosyVoice (AR LLM + flow matching) |
| **Full-Duplex Speech-LLM** | Moshi |

### By Core Technique
| Technique | Models |
|-----------|--------|
| **Flow Matching** | F5-TTS, E2 TTS, Voicebox, CosyVoice, Matcha-TTS, MegaTTS 3 |
| **Diffusion** | NaturalSpeech 3, StyleTTS 2, Mars5 (NAR stage), SALAD |
| **Codec Language Model** | VALL-E 2, VALL-E R, Orpheus, OuteTTS, Spark-TTS, Llasa |
| **Masked Generation** | MaskGCT |
| **LLM-Native** | Orpheus (Llama), Spark-TTS (Qwen), Llasa (Llama), LLMVoX |

### By Audio Representation
| Representation | Models |
|----------------|--------|
| **Mel Spectrogram** | F5-TTS, E2 TTS, Voicebox, Matcha-TTS, StyleTTS 2, MegaTTS 3 |
| **Encodec tokens** | VALL-E 2, VALL-E R, Mars5, MaskGCT (acoustic) |
| **DAC tokens** | Zonos, OuteTTS |
| **SNAC tokens** | Orpheus |
| **Custom VQ** | Llasa (single-layer VQ), Spark-TTS (BiCodec) |
| **Factorized VQ** | NaturalSpeech 3 |

---

## Recommendations for Voice LLM Pipeline Integration

### Lowest Latency (Real-Time Conversation)
1. **LLMVoX** (30M add-on, streaming, LLM-agnostic)
2. **CosyVoice 2/3** (150ms streaming, production-ready)
3. **Orpheus TTS** (100-200ms streaming, Llama backbone)
4. **XTTS v2** (<200ms streaming, 17 languages)

### Highest Quality (Zero-Shot Cloning)
1. **Seed-TTS** (human parity, proprietary)
2. **VALL-E 2** (human parity, proprietary)
3. **F5-TTS** (best open-source quality/speed tradeoff)
4. **MaskGCT** (highest SIM-o among open models)

### Most Controllable
1. **Zonos** (emotion, pitch, rate, quality knobs)
2. **Spark-TTS** (fine-grained pitch/rate via BiCodec)
3. **Orpheus** (emotion tags: laugh, sigh, gasp)
4. **Parler-TTS** (natural language voice descriptions)
5. **CosyVoice 3** (instruction-based control)

### Best for LLM Weight Sharing
1. **Orpheus** (Llama-3.2-3B backbone)
2. **Spark-TTS** (Qwen2.5 backbone)
3. **Llasa** (Llama-aligned, 1B-8B)
4. **LLMVoX** (works with any LLM, 30M adapter)

### Most Production-Ready (Open Source)
1. **CosyVoice 3** (9 languages, streaming, Docker, GRPC)
2. **F5-TTS** (TensorRT-LLM, Triton, RTF 0.04)
3. **XTTS v2** (17 languages, 7M monthly downloads)

### Paradigm-Shifting
1. **Moshi** (eliminates entire pipeline -- full-duplex speech-LLM)
2. **Llasa** (inference-time compute scaling for TTS)
3. **MegaTTS 3** (1-minute speech in 8 steps)
4. **E1 TTS** (single-step generation)

---

## Key Trends (2024-2026)

1. **Flow Matching dominance:** OT-CFM has become the default for NAR TTS, replacing score-based diffusion. F5-TTS, CosyVoice, MegaTTS 3 all use variants.

2. **LLM-native TTS:** The biggest shift -- Orpheus, Spark-TTS, Llasa, LLMVoX all use standard LLM architectures (Llama, Qwen) directly for speech generation, enabling weight sharing and standard LLM tooling.

3. **Death of complex pipelines:** F5-TTS showed you need NOTHING beyond a DiT + ConvNeXt + filler tokens. No phoneme alignment, no duration model, no text encoder, no semantic codec.

4. **Streaming-first design:** CosyVoice 2's chunk-aware causal flow matching enables streaming + non-streaming in one model. This is becoming table stakes.

5. **Codec innovation:** BiCodec (Spark-TTS), single-layer VQ (Llasa), FSQ (CosyVoice 2), DAC (Zonos) -- moving beyond Encodec/SoundStream.

6. **Full-duplex convergence:** Moshi shows the endgame: no separate ASR/LLM/TTS, just one model that listens and speaks simultaneously.

7. **Inference-time scaling:** Llasa's use of speech verifiers during search mirrors LLM best-of-N/reward-model patterns, suggesting TTS will follow the same scaling laws.

8. **Emotion and control:** Orpheus tags, Zonos conditioning, Spark-TTS fine-grained control -- expressive TTS is no longer optional.
