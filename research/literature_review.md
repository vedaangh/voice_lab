# Literature Review: Voice & Speech LLMs

*Updated March 2026 — Voice-LLM Workbench (Vedaangh Rungta, Cambridge Part II)*

---

## How to Read This Document

**Section 1** explains the paradigm shift from early (2023) to current (2025-26) approaches — the "old to new" narrative. **Section 2** covers the ~25 most important papers in detail, ordered by influence. **Section 3** synthesizes cross-cutting themes. **Section 4** covers multiturn stability specifically. **Section 5** maps implications to this project. A full catalogue of all ~70+ papers surveyed (with arXiv IDs) is in Section 6.

---

## 1. The Paradigm Shift: 2023 -> 2026

### The Old World (2023 - early 2024)

The first generation of speech LLMs treated speech as a **translation problem**: convert speech to discrete tokens, bolt them onto a text LLM, and hope the LLM generalises.

- **SpeechGPT** (May 2023): Expanded LLaMA's vocabulary with HuBERT K-means tokens. Three-stage training. Had to generate text *then* speech sequentially — latency >1s. Proved the concept but was impractical.
- **AudioPaLM** (June 2023): Extended PaLM-2 with w2v-BERT semantic tokens. Key finding: fine-tuning from a text LLM gives 18.4 BLEU vs 6.9 from scratch — **text pretraining massively benefits speech**.
- **SALMONN** (Oct 2023): Dual encoder (Whisper + BEATs) + Q-Former + frozen Vicuna. Understanding only. Discovered "task overfitting" — the model defaults to ASR and ignores instructions. Required "activation tuning" (12 story samples) to fix.

**Characteristics of this era**: discrete semantic tokens (HuBERT K-means), vocabulary expansion, sequential text-then-speech generation, high latency, single-turn, understanding-only or poor generation quality.

### The Transition (mid-2024)

Two breakthroughs changed the field:

1. **LLaMA-Omni** (Sept 2024, ICLR 2025): Showed you can get 226ms latency with a simple architecture: Whisper encoder + MLP adapter + frozen Llama-3.1-8B + non-autoregressive CTC decoder. Simultaneous text and speech generation. Trained in 3 days on 4 GPUs. This democratised speech LLM research.

2. **Moshi** (Sept 2024): Showed full-duplex is possible — listen and speak simultaneously with no turn-taking logic, 200ms latency, by modelling user and system as 17 parallel codec streams. Required 1,016 H100s and 7M hours of audio. Introduced Mimi codec (12.5 Hz, 1.1 kbps).

**Also important**: Mini-Omni (first open-source real-time), Freeze-Omni (frozen LLM throughout), OmniFlatten (full-duplex via token flattening, no arch changes).

### The New World (2025 - present)

Five shifts define the current era:

**Shift 1: Thinker-Talker replaces encoder-decoder.** Qwen2.5-Omni (March 2025) introduced a clean decomposition: a Thinker (full LLM with encoders) reasons and generates text; a Talker (smaller AR decoder) receives the Thinker's *hidden states* (not just text) and generates speech concurrently. The Talker gets information that text cannot express — emphasis, uncertainty, emotion. This is the practical open-source answer to GPT-4o.

**Shift 2: LLM-native TTS.** Instead of bolting speech onto LLMs, the newest TTS models *are* LLMs. Orpheus (Llama-3.2-3B + SNAC), Spark-TTS (Qwen2.5 + BiCodec), Llasa (Llama + single VQ, 1B-8B) — standard LLM architectures generating speech tokens directly. This means shared weights, shared tooling (vLLM, TensorRT), and a path to unified models.

**Shift 3: Flow matching kills complex TTS pipelines.** F5-TTS (Oct 2024) showed you need nothing but a DiT + ConvNeXt + filler tokens — no phoneme alignment, no duration model, no text encoder, no semantic codec. 2.42% WER, RTF 0.04. CosyVoice 2/3 added streaming flow matching (150ms). MaskGCT (ICLR 2025) went fully non-autoregressive via mask-and-predict.

**Shift 4: Single-codebook, low-token-rate representations.** The field moved from EnCodec (75 Hz x 8 codebooks = 600 tok/s) to WavTokenizer (40 tok/s, single codebook), Mimi (12.5 Hz x 8 = 100 tok/s), and CosyVoice FSQ (25 tok/s). Fewer tokens = longer audio in context = better multiturn.

**Shift 5: Massive-scale end-to-end training.** Kimi-Audio (April 2025) trained on 13M hours. Step Audio (Feb 2025) is 130B parameters. GLM-4-Voice trained on 1T tokens. The scale gap between speech and text LLMs is closing.

### The Latest Wave (H2 2025 - Q1 2026)

Three further shifts have emerged since mid-2025:

**Shift 6: Audio reasoning unlocked.** VERA (Sept 2025) quantified the devastating gap: competition math drops from 74.8% (text) to 6.1% (voice input). Step-Audio-R1 (Nov 2025) became the first model to address this via Modality-Grounded Reasoning Distillation, surpassing Gemini 2.5 Pro on audio understanding. STITCH (Microsoft, Jul 2025) introduced "think-while-talking" — alternating unspoken reasoning chunks with spoken output, improving math reasoning ~15% with zero latency increase. Mind-Paced Speaking (Oct 2025) took a brain-inspired dual-processing approach (one module reasons, one speaks). The consensus: speech LLMs *must* preserve explicit reasoning pathways or face catastrophic performance loss.

**Shift 7: Full-duplex matures into a subfield.** From a single model (Moshi, 2024) to 15+ systems by Q1 2026. FLAIR (Mar 2026) introduces latent "think-while-listening" reasoning. DuplexCascade (Mar 2026) achieves SOTA turn-taking via VAD-free micro-turns. PersonaPlex (NVIDIA, Jan 2026) adds voice cloning and role control to duplex. F-Actor (Jan 2026) makes duplex behaviour instruction-controllable. A dedicated survey (2509.14515), three benchmarks (FLEXI, FD-Bench, HumDial Challenge), and an ICASSP 2026 challenge now exist. The field has moved from "can we do it?" to "how do we control and evaluate it?"

**Shift 8: Codec compression race reaches extreme lows.** Frame rates pushed from 12.5 Hz (Mimi) down to 6.25 Hz (TaDiCodec, 0.0875 kbps) and 5 Hz (U-Codec, Tencent). MOSS-Audio-Tokenizer (Feb 2026) scales the tokenizer itself to 1.6B parameters trained on 3M hours. Disentangled codecs mature — MSR-Codec (4 streams: semantic/timbre/prosody/residual), Kanade (single-layer speaker disentanglement), DisCodec (content/prosody/timbre separation). The tokenizer is becoming its own foundation model, not just a preprocessing step.

---

## 2. The Most Important Papers (Detailed)

### Tier 1: The Defining Papers

---

#### 2.1 Qwen2.5-Omni — The Thinker-Talker Blueprint
**arXiv: 2503.20215 | Alibaba, March 2025 | 7B**

The most important open-source voice LLM architecture as of early 2026.

**Architecture:**
- **Thinker**: Full Transformer decoder LLM with attached encoders for audio (Whisper-style), images, and video. Processes all modalities, generates text tokens and hidden states. This is the reasoning brain.
- **Talker**: Dual-track autoregressive decoder. Track 1 receives text tokens from the Thinker. Track 2 receives the Thinker's *hidden representations* — not just text, but the full internal state including information about emphasis, uncertainty, emotion that the Thinker inferred but didn't express in text. Generates speech audio tokens in streaming fashion.
- **TMRoPE**: Novel positional embedding synchronising video frame timestamps with audio timestamps in interleaved input.
- **Sliding-window DiT**: Restricts receptive field for streaming audio decoding, reducing first-packet delay.

**Key results:**
- Speech instruction following on MMLU/GSM8K **matches text input** — the first model where speaking to it is as good as typing.
- SOTA on OmniBench (multimodal).
- Talker speech "outperforms most existing streaming and non-streaming alternatives in robustness and naturalness."
- Open weights (7B).

**Why it matters:** The Thinker-Talker decomposition is elegant because: (a) you can upgrade the Thinker (better reasoning) or Talker (better speech) independently; (b) the Talker gets hidden states, not just text, so it can express what text cannot; (c) joint training optimises both; (d) it streams natively. This is the architecture pattern most likely to be adopted widely.

---

#### 2.2 LLaMA-Omni — The Accessible Baseline
**arXiv: 2409.06666 | ICT/CAS, Sept 2024 | ICLR 2025**

**Architecture (your project re-implements this):**
- Whisper-large-v3 encoder (frozen) -> 5x downsampling + 2-layer MLP adapter -> Llama-3.1-8B-Instruct -> non-autoregressive CTC speech decoder (2 Transformer layers, 4096 hidden dim, 425M params) -> HuBERT discrete units (K=1000) -> HiFi-GAN vocoder.
- Two-stage training: (1) adapter+LLM on speech->text (frozen encoder); (2) decoder with CTC loss (everything else frozen).

**Key results:**
- 226ms streaming latency (lower than GPT-4o's 320ms).
- 3.99 ChatGPT Score (S2T), 3.47 (S2S), UTMOS 3.93, 10.82% ASR-WER.
- Trained in 65 hours on 4x L40 GPUs.
- InstructS2S-200K dataset: 50K Alpaca + 150K UltraChat, rewritten by Llama-3-70B, TTS-synthesised.

**Why it matters:** Proved that a competent speech LLM doesn't require thousands of GPUs. The modular encoder-adapter-LLM-decoder pattern is reproducible and extensible. The CTC decoder's non-autoregressive nature is key to streaming — speech is generated in parallel with text at negligible overhead.

**LLaMA-Omni 2** (arXiv: 2505.02625, May 2025): Replaces CTC with autoregressive streaming TTS decoder (init from Qwen2.5-0.5B), Read-Write strategy (3 text -> 10 speech tokens per chunk), FSQ tokenizer at 25 Hz from SenseVoice. UTMOS 4.19 (vs 3.93), native multiturn (1-5 turns), Qwen2.5 backbone (0.5B-14B). The 14B model gets 4.56 ChatGPT score.

---

#### 2.3 Moshi — Full-Duplex Foundation
**arXiv: 2410.00037 | Kyutai Labs, Sept 2024 | 7B**

**Architecture:**
- 7B Helium backbone (pretrained on 2.1T text tokens) with RQ-Transformer hierarchy.
- Models 17 parallel streams: 1 text (Inner Monologue) + 8 Moshi Mimi codec streams + 8 user Mimi codec streams.
- **Mimi codec**: 12.5 Hz, 8 codebooks of 2048 entries, 1.1 kbps. First codebook distilled from WavLM (semantic); remaining 7 are acoustic. Split RVQ architecture.
- **Inner Monologue**: Predicts time-aligned text tokens as prefixes to audio tokens. This improves linguistic quality by grounding speech in text reasoning.
- No explicit turn-taking — full duplex is emergent from parallel stream modelling.

**Key results:**
- 200ms practical latency (160ms theoretical).
- Trained on 7M hours audio + 2.1T text tokens on 1,016 H100s.
- ~5 minute context window.
- Handles overlapping speech, interruptions, backchannels natively.

**Why it matters:** First system where the model genuinely listens while speaking. The Inner Monologue (text as prefix to audio) became influential — it shows that even in an "end-to-end" model, text reasoning helps. The Mimi codec's 12.5 Hz rate was a breakthrough in token efficiency.

---

#### 2.4 Kimi-Audio — Scaling Speech Pretraining
**arXiv: 2504.18425 | Moonshot AI, April 2025 | 7B**

**Architecture:**
- **Audio Tokenizer**: 12.5 Hz discrete semantic tokens via VQ + continuous acoustic features from Whisper encoder (dual representation).
- **Audio LLM**: Transformer initialised from Qwen2.5-7B with parallel heads for text and audio token generation.
- **Audio Detokenizer**: Chunk-wise streaming flow-matching + BigVGAN vocoder for 24 kHz output.

**Key results:**
- Pretrained on **13 million hours** of audio data (speech + sound + music) — the largest reported audio pretraining dataset.
- SOTA on speech recognition, audio understanding, audio QA, and speech conversation.
- Open source (weights, code, eval toolkit).

**Why it matters:** Demonstrates that massive-scale audio pretraining works, analogous to how massive text pretraining worked for GPT. The dual tokenizer (semantic tokens for LLM + continuous features for quality) is a pragmatic design. Being Qwen2.5-based and open-source makes it directly relevant to this project.

---

#### 2.5 F5-TTS — The Flow Matching Standard
**arXiv: 2410.06885 | SJTU/Cambridge, Oct 2024**

**Architecture:**
- DiT (Diffusion Transformer, 22 layers, 1024 dim, 336M params) with ConvNeXt V2 text refinement.
- Text-guided speech infilling via Optimal Transport Conditional Flow Matching (OT-CFM).
- Operates on mel spectrograms (no codec). Vocos vocoder for final waveform.
- **No phoneme alignment, no duration model, no text encoder, no semantic codec.**
- Sway Sampling: non-uniform flow step sampling that universally improves quality without retraining.

**Key results:**
- 2.42% WER, 0.66 SIM-o on LibriSpeech (near ground truth 2.23% / 0.69).
- RTF 0.15 at 16 NFE on RTX 3090; RTF 0.04 with TensorRT-LLM.
- Trained on 100K hours Emilia dataset in ~1 week on 8xA100.

**Why it matters:** Proved that TTS can be radically simplified. The "embarrassingly easy" approach (from its predecessor E2 TTS) actually works when you add ConvNeXt text refinement. Sway Sampling is universally applicable. This is the best open-source TTS quality/speed tradeoff and the default choice for cascade voice LLM pipelines.

---

#### 2.6 Qwen3-Omni — Thinker-Talker at MoE Scale
**arXiv: 2509.17765 | Alibaba/Qwen, September 2025**

The paper Qwen2.5-Omni promised. Extends Thinker-Talker to a Mixture-of-Experts architecture supporting 119 languages for text, 19 for speech comprehension, and 10 for speech generation.

**Architecture:**
- Thinker-Talker preserved from Qwen2.5-Omni, now with MoE backbone.
- Lightweight ConvNet replaces diffusion-based vocoding, achieving theoretical 234ms first-packet latency.
- Open-source under Apache 2.0.

**Key results:**
- Outperforms Gemini 2.5 Pro on **32 of 36** audio and audiovisual benchmarks.
- Open-source SOTA across the board as of late 2025.

**Why it matters:** Validates that Thinker-Talker scales. The MoE architecture means capacity grows without proportional compute. The 119-language support and open weights make this the definitive open-source omni-model.

---

#### 2.7 Step-Audio-R1 — First Audio Reasoning Model
**arXiv: 2511.15848 | StepFun, November 2025**

The first model to successfully unlock reasoning capabilities for audio language models.

**Architecture:**
- **Modality-Grounded Reasoning Distillation**: Distills reasoning from text-mode chain-of-thought into the audio pathway. Rather than hoping the model reasons from speech, it explicitly transfers text reasoning ability.
- Builds on Step Audio (130B) foundation.

**Key results:**
- Surpasses Gemini 2.5 Pro on audio understanding tasks.
- Approaches Gemini 3 Pro on several audio reasoning benchmarks.

**Why it matters:** Directly addresses the VERA finding (74.8% text vs 6.1% voice on math). Proves that speech LLMs *can* reason if you explicitly distil the capability, rather than hoping it transfers from text pretraining. This is the first credible approach to closing the modality reasoning gap.

---

#### 2.8 VERA — Quantifying the Speech Reasoning Gap
**arXiv: 2509.26542 | September 2025**

2,931 voice-based test episodes across Math, Web, Science, Long-Context, and Factual domains.

**Key results:**
- Competition math accuracy: **74.8% text → 6.1% voice**. Overall: 54.0% text → 11.3% voice.
- Increasing compute or decoupling reasoning from speech provides only marginal improvement.
- The gap is not just about ASR errors — speech mode fundamentally disrupts the reasoning process.

**Why it matters:** The quantitative proof that the "LLM Dormancy" problem is devastating for complex tasks. Every speech LLM project must reckon with this gap. Makes Step-Audio-R1's reasoning distillation approach essential reading.

---

### Tier 2: Architecturally Important

---

#### 2.9 Freeze-Omni — Frozen LLM Speech
**arXiv: 2411.00774 | Nov 2024**


Core insight: **the LLM never changes**. Qwen2-7B is completely frozen. Only the encoder (350M), adapter, and speech decoder (NAR+AR, ~240M) are trained. Six-stage training. Uses TiCodec (single 1024-codebook at 40 Hz).

Results: 3.24% WER LibriSpeech clean, supports multiturn and duplex via chunk-level state prediction. Demonstrates speech capabilities added without catastrophic forgetting — critical for large models like your Qwen3-30B where you don't want to degrade text reasoning.

---

#### 2.10 OmniFlatten — Full-Duplex Without Architecture Changes
**arXiv: 2410.17799 | Oct 2024 | Qwen2-0.5B**

Converts a text LLM to full-duplex speech dialogue with **zero architectural modifications** — only a "flattening" operation interleaving speech (CosyVoice tokens) and text tokens. Progressive training: modality alignment -> half-duplex -> full-duplex. Beats Moshi on turn-taking latency (193ms vs 553ms) and accuracy (71.7% vs 55.1%). Shows that the power is in the training recipe, not the architecture.

---

#### 2.11 Step Audio — Chinese Multimodal at Scale
**arXiv: 2502.11946 | StepFun, Feb 2025 | 130B**

130B parameter unified speech-text model. 3B TTS via knowledge distillation. Instruction-driven control for dialects, emotions, singing, rap. Tool-calling and role-playing. 9.3% average improvement on open benchmarks. Open source. The largest open speech LLM by parameter count.

---

#### 2.12 GLM-4-Voice — Interleaved Text-Speech Generation
**arXiv: 2412.02612 | Zhipu AI, Dec 2024 | 9B**

Ultra-low bitrate speech tokenizer (175 bps, 12.5 Hz) built by inserting VQ into Whisper encoder. GLM-4-9B generates interleaved text and speech tokens. CosyVoice flow-matching decoder. Trained on 1T tokens. Can control emotion, intonation, rate, dialect via instructions. 4.16 ChatGPT score but 1563ms latency.

---

#### 2.13 Orpheus TTS — LLM-Native Speech Generation
**Canopy Labs, 2025 | Llama-3.2-3B**

A Llama-3.2-3B model fine-tuned to generate SNAC codec tokens directly. Emotion tags (`<laugh>`, `<sigh>`, `<gasp>`, etc.) in the text prompt. 100-200ms streaming latency. The significance: this is a standard LLM generating speech — you can use vLLM, quantisation, LoRA, etc. The weight-sharing path to unified models.

---

#### 2.14 MaskGCT — Fully Non-Autoregressive TTS
**arXiv: 2409.00750 | ICLR 2025 | 1B**

Two-stage mask-and-predict: (1) Text -> W2V-BERT semantic tokens; (2) Semantic -> acoustic tokens. All tokens generated in parallel (no sequential bottleneck). 0.687 SIM-o (highest among open models). Shows that autoregressive generation is not necessary for high-quality TTS.

---

#### 2.15 IntrinsicVoice — GroupFormer for Length Matching
**arXiv: 2410.08035 | Oct 2024 | Qwen2-7B**

Solves the fundamental speech-text length mismatch: when the LLM predicts `<speech>`, a small NAR GroupFormer predicts G speech tokens simultaneously, reducing speech to ~5 tok/s (matching text rate). <100ms latency. Multi-turn trained. Architectural solution to a problem most models paper over.

---

#### 2.16 CSM (Sesame) — Context-Conditioned Speech Quality
**Sesame AI, 2025 | Llama-3.2-1B**

Speech generation only, but the best at it. Dual AR transformer (backbone for RVQ-1, decoder for remaining levels) using Mimi codec. Conditions on full conversation history (text + audio of prior turns). The only model that demonstrably **improves** with more context — quality increases as conversation develops. Addresses the "one-to-many" problem (same text, many valid prosodies, context decides which).

---

#### 2.17 Ichigo — Early Fusion Done Right
**arXiv: 2410.15316 | Oct 2024 | Llama-3.1-8B**

WhisperVQ tokenizer (512 codes) added to Llama vocabulary. No adapter — single unified transformer. 111ms first-token latency (4x faster than cascade). <10% text capability degradation. Text output only. Proves early fusion matches adapter-based approaches while being simpler.

---

#### 2.18 Spark-TTS — BiCodec + Chain-of-Thought TTS
**arXiv: 2503.01710 | ACL 2025 | Qwen2.5-0.5B**

BiCodec separates speech into: (1) variable-length semantic tokens (content), (2) fixed-length global tokens (speaker identity). Qwen2.5 backbone generates tokens via chain-of-thought. Fine-grained pitch/rate control. 0.5B parameters. Comes with VoxBox dataset (100K hours with attribute annotations).

---

### Tier 2b: Key Papers from H2 2025 - Q1 2026

---

#### 2.19 FLAIR — Think-While-Listening Full-Duplex
**arXiv: 2603.17837 | March 2026 | incl. Yoshua Bengio**

The Silent Thought: performs latent reasoning *simultaneously* with speech perception via recursive latent embeddings. No explicit reasoning annotations needed. ELBO-based training objective. Full-duplex model that genuinely thinks while it listens, rather than waiting for a turn boundary to reason.

---

#### 2.20 DuplexCascade — VAD-Free Full-Duplex
**arXiv: 2603.09180 | March 2026**

VAD-free cascaded ASR-LLM-TTS pipeline with "micro-turn" optimisation — converts utterance-level long turns into chunk-wise micro-turn interactions. SOTA full-duplex turn-taking among open-source S2S systems. Important because it shows that a carefully optimised cascade can match E2E duplex quality.

---

#### 2.21 Fun-Audio-Chat — Solving Catastrophic Forgetting
**arXiv: 2512.20156 | Alibaba/Tongyi, December 2025**

Explicitly identifies and addresses "catastrophic forgetting of text LLM knowledge" during speech training. Proposes **Core-Cocktail Training**: two-stage fine-tuning with intermediate model merging. Uses **Dual-Resolution Speech Representations** at 5Hz/25Hz, reducing GPU by 50%. Directly validates the dormancy problem and offers the first targeted training recipe to mitigate it.

---

#### 2.22 Qwen3-TTS — Production-Grade LLM-Native TTS
**arXiv: 2601.15621 | Alibaba/Qwen, January 2026**

Dual-track LM architecture trained on 5M+ hours across 10 languages. Two tokenizers: 25Hz semantic for streaming, 12.5Hz multi-codebook for quality. Supports 3-second voice cloning and description-based control. Sub-100ms first-packet latency. Open-source (Apache 2.0). The production TTS from the Qwen ecosystem.

---

#### 2.23 STITCH — Think-While-Talking
**arXiv: 2507.15375 | Microsoft, July 2025**

Alternates between unspoken reasoning chunks and spoken response chunks. Uses audio playback duration as thinking time (speaks one chunk while reasoning about the next). Improves math reasoning ~15% with no latency increase. Elegant solution: the model literally uses the time it takes to speak as "thinking time."

---

#### 2.24 EchoX — Mitigating Acoustic-Semantic Gap
**arXiv: 2509.09174 | September 2025**

Addresses how speech LLMs derived from text LLMs lose knowledge and reasoning capabilities — the acoustic-semantic gap. Dynamically generates speech training targets using semantic representations. Relevant to LLM dormancy: shows the gap is in the representation space, not just attention patterns.

---

#### 2.25 CLEAR — Continuous Latent AR TTS
**arXiv: 2508.19098 | August 2025**

Zero-shot TTS operating on continuous audio representations (not discrete tokens) with enhanced VAE + MLP-based rectified flow. 1.88% WER (beats F5-TTS's 2.42%), RTF 0.29, 96ms first-frame streaming delay. Challenges the assumption that discrete tokens are necessary.

---

#### 2.26 TaDiCodec — Extreme Low-Rate Codec
**arXiv: 2508.16790 | August 2025**

Diffusion autoencoder with text guidance achieving **6.25 Hz** frame rate and **0.0875 kbps** with single-layer codebook for 24kHz speech. For context: Mimi is 12.5Hz/1.1kbps, WavTokenizer is 40tok/s. Single-stage end-to-end training. Compatible with both AR and masked generative modelling.

---

#### 2.27 X-Talk — The Case for Optimised Cascades
**arXiv: 2512.18706 | December 2025**

Argues that modular cascaded pipelines are "underestimated." Demonstrates that a systematically optimised cascade (ASR→LLM→TTS) achieves sub-second latency without sacrificing flexibility. Open-source framework. Important counterpoint to the E2E narrative: maybe the answer is a better cascade, not a bigger monolith.

---

#### 2.28 MiMo-Audio — Few-Shot Audio LLM
**arXiv: 2512.23808 | Xiaomi, December 2025**

Scaled pretraining to 100M+ hours enables few-shot learning for audio tasks. Open-source SOTA on MMSU, MMAU, MMAR, MMAU-Pro benchmarks. Demonstrates that sufficient pretraining scale unlocks in-context learning for audio, analogous to GPT-3's finding for text.

---

#### 2.29 Dragon-FM — AR + Flow Matching Unified
**arXiv: 2507.22746 | Microsoft, July 2025**

Chunk-level AR for global consistency + parallel flow-matching within chunks at 12.5 tok/s. Bridges continuous and discrete feature modelling using finite scalar quantisers. High-quality zero-shot podcast generation. The AR+FM hybrid is becoming a dominant pattern.

---

#### 2.30 MOSS-Audio-Tokenizer — Foundation Model Codecs
**arXiv: 2602.10934 | Fudan, February 2026 | 1.6B params**

1.6B-parameter Transformer-based tokenizer trained end-to-end on 3M hours. Consistently outperforms all prior codecs across all bitrates for speech, sound, and music. Represents the paradigm where the tokenizer itself becomes a foundation model requiring massive-scale training.

---

## 3. Running Themes

### 3.1 The Quality Gap Is Closing but Not Closed

VoiceBench (Oct 2024) showed cascade (ASR+LLM+TTS) scoring 81.88 vs best E2E at 60.45 — a 20+ point gap. As of Q1 2026:
- Qwen3-Omni beats Gemini 2.5 Pro on 32/36 audio benchmarks — the open-source gap to proprietary is closing fast
- LLaMA-Omni 2 (14B) achieves 4.56 ChatGPT score, competitive with cascades
- SD-Eval shows E2E outperforms cascade for paralinguistic tasks (emotion, accent)
- VoiceAgentBench (Oct 2025) confirms cascade still wins for agentic tasks (~60.6% accuracy)
- X-Talk (Dec 2025) argues optimised cascades are "underestimated" and achieve sub-second latency
- **The gap is now primarily on reasoning-heavy tasks** (VERA: 74.8% text → 6.1% voice on math)

### 3.2 The LLM Dormancy / Catastrophic Forgetting Problem

Now the single most well-documented challenge in the field:
- **VERA** (Sept 2025): The definitive quantification — 54.0% text vs 11.3% voice overall
- **Fun-Audio-Chat** (Dec 2025): Explicitly names "catastrophic forgetting of text LLM knowledge" and proposes Core-Cocktail Training
- **EchoX** (Sept 2025): Shows the gap is in the representation space — dynamically generates speech targets using semantic representations
- **Step-Audio-R1** (Nov 2025): Proves reasoning can be recovered via Modality-Grounded Reasoning Distillation
- **STITCH** (Jul 2025): Think-while-talking circumvents the problem by interleaving silent reasoning with speech
- Earlier findings: SALMONN attention collapse, FireRedASR prompt ignoring, Freeze-Omni's frozen LLM as prevention
- Freeze-Omni's approach (never modify LLM) prevents forgetting but limits adaptation; Core-Cocktail Training is the first recipe to *recover* lost capabilities

### 3.3 Codec Convergence — Now a Foundation Model Problem

The field has pushed well beyond the 2024 convergence point:
- **5-6.25 Hz** at the extreme low end: U-Codec (5Hz, Tencent), TaDiCodec (6.25Hz, 0.0875 kbps)
- **12.5-25 Hz** as the practical standard (Qwen3-TTS: 12.5Hz multi-codebook + 25Hz semantic)
- **Foundation-scale tokenizers**: MOSS-Audio-Tokenizer (1.6B params, 3M hours), UniAudio 2.0 (100B tokens)
- **Disentangled codecs dominate**: MSR-Codec (4 streams), Kanade (speaker disentanglement), DisCodec (3 factors), DashengTokenizer (reversed: semantic-first with acoustic injection)
- **RL for codec quality**: GRPO applied to pronunciation accuracy (IndexTTS 2.5), stability (Multi-Reward GRPO)
- Winners shifting: Mimi (for duplex models), SNAC (for LLM-native TTS), TaDiCodec/U-Codec (for extreme compression), MOSS (general-purpose)

### 3.4 Training Data Remains the Bottleneck (but Scale is Increasing)

| Dataset | Size | Used By |
|---------|------|---------|
| MiMo-Audio pretraining | 100M+ hours | MiMo-Audio |
| Kimi-Audio pretraining | 13M hours | Kimi-Audio |
| Moshi pretraining | 7M hours | Moshi |
| Qwen3-TTS pretraining | 5M+ hours | Qwen3-TTS |
| MOSS-Audio-Tokenizer | 3M hours | MOSS codec |
| GLM-4-Voice pretraining | 1T tokens | GLM-4-Voice |
| Emilia | 100-200K hours | F5-TTS, MaskGCT |
| GLM-TTS | 100K hours | GLM-TTS |
| InstructS2S-200K | 200K pairs | LLaMA-Omni |
| UltraVoice | 830+ hours (style-annotated) | SLAM-Omni, VocalNet |

Scale is increasing (100M+ hours for MiMo-Audio), but quality annotation remains scarce. True duplex conversational data is still extremely rare (Fisher: 2K hours). FLM-Audio (Sept 2025) shows natural monologues with pauses can substitute, and SpeechJudge (Nov 2025) provides a 99K-pair human feedback corpus for quality evaluation.

### 3.5 The LLM-Native TTS Revolution (Matures in H2 2025)

The 2025 trend of TTS-as-LLM has fully matured by early 2026:

| Model | LLM Backbone | Codec | Date | Significance |
|-------|-------------|-------|------|-------------|
| Qwen3-TTS | Qwen3 | Dual (25Hz/12.5Hz) | Jan 2026 | Production-grade, 10 languages, 5M hrs |
| GLM-TTS | Custom | F0-constrained | Dec 2025 | RL-optimised, LoRA voice custom |
| Dragon-FM | Custom | FSQ @ 12.5Hz | Jul 2025 | AR+flow matching hybrid |
| CLEAR | Custom | Continuous VAE | Aug 2025 | 1.88% WER, no discrete tokens |
| CTC-TTS | LLM | CTC-aligned | Feb 2026 | CTC for LLM-TTS (dual variants) |
| Orpheus | Llama-3.2-3B | SNAC | Early 2025 | Emotion tags, streaming |
| Spark-TTS | Qwen2.5-0.5B | BiCodec | Mar 2025 | Chain-of-thought, controllable |
| Llasa | Llama (1-8B) | Single VQ | Feb 2025 | Inference-time scaling |
| Kimi-Audio | Qwen2.5-7B | 12.5Hz VQ + Whisper | Apr 2025 | Unified understand+generate |

Key new trend: **RL-based post-training** for TTS quality. Multi-Reward GRPO (Nov 2025) applies group relative preference optimisation with multiple rewards (prosody, naturalness, stability). DMOSpeech 2 (Jul 2025) extends this to duration prediction. IndexTTS 2.5 uses GRPO for pronunciation accuracy. This parallels the RLHF revolution in text LLMs.

### 3.6 Full-Duplex Becomes a Subfield

From 1 model (Moshi, 2024) to 15+ systems by Q1 2026:

| Model | Date | Key Innovation |
|-------|------|---------------|
| FLAIR | Mar 2026 | Think-while-listening via latent reasoning |
| DuplexCascade | Mar 2026 | VAD-free micro-turns, SOTA turn-taking |
| SoulX-Duplug | Mar 2026 | Plug-and-play state prediction |
| PersonaPlex (NVIDIA) | Jan 2026 | Voice cloning + role control in duplex |
| F-Actor | Jan 2026 | Instruction-controllable duplex behaviour |
| Chroma 1.0 | Jan 2026 | Sub-second E2E with voice cloning |
| X-Talk | Dec 2025 | Optimised cascade for duplex |
| Covo-Audio (Tencent) | Feb 2026 | Full-duplex from Tencent |
| Phoenix-VAD | Sept 2025 | LLM-based semantic endpoint detection |
| FLM-Audio | Sept 2025 | Natural monologues for duplex training |
| RoboEgo | Jun 2025 | 80ms theoretical duplex latency |
| OmniFlatten | Oct 2024 | Duplex via flattening only |
| Moshi | Sept 2024 | The original full-duplex system |

Dedicated evaluation: FLEXI (Sept 2025), FD-Bench (Jul 2025), HumDial ICASSP 2026 Challenge. FD-Bench finding: all tested models (Moshi, Freeze-Omni, VITA-1.5) fail to reliably handle interruptions under noisy conditions.

### 3.7 Benchmark Explosion (H2 2025 - Q1 2026)

At least 20 new speech LLM benchmarks since June 2025. The field has far outgrown VoiceBench:

| Category | Benchmarks |
|----------|-----------|
| **Multi-turn dialogue** | Audio MultiChallenge, MTalk-Bench, SDiaReward, VoxRole |
| **Full-duplex** | FLEXI, FD-Bench, HumDial Challenge |
| **Voice reasoning** | VERA, MMAR, MMAU-Pro |
| **Agentic tasks** | VoiceAgentBench, VoiceAgentEval, Stream RAG |
| **Holistic/multimodal** | AHELM, VoiceAssistant-Eval, MultiVox, WildSpeech-Bench |
| **Safety/bias** | SACRED-Bench, MedVoiceBias |
| **TTS evaluation** | SpeechJudge, Vox-Evaluator, RVCBench |
| **Multilingual** | mSTEB, VCB Bench, MAEB |

Key findings across benchmarks: Even Gemini 3 Pro achieves only 54.65% on Audio MultiChallenge. SACRED-Bench shows 66% attack success rate on Gemini 2.5 Pro. Models can capture *what* was said but fail to identify *who* said it (M3-SLU). Proprietary models do not universally outperform open-source (VoiceAssistant-Eval).

---

## 4. Multiturn Conversation Stability

### State of Evaluation (Significantly Improved Since Mid-2025)
- **MTalk-Bench** (2508.18240): First dedicated multi-turn S2S benchmark. 9 scenarios across semantic, paralinguistic, and ambient dimensions. Key finding: models sacrifice efficiency (longer responses) to maintain coherence.
- **Audio MultiChallenge** (2512.14865): 452 conversations, 1712 rubrics. Tests Inference Memory, Instruction Retention, Self Coherence, Voice Editing. Even Gemini 3 Pro Preview achieves only 54.65% pass rate.
- **SDiaReward** (2603.14889): First end-to-end multi-turn reward model for spoken dialogue. Addresses "modality gap" (prosody/emotion) and "colloquialness gap" (natural vs scripted). Introduces ESDR-Bench.
- **VoxRole** (2509.03940): 13K multi-turn dialogues, 65.6 hours, 1228 characters across 261 movies. Tests persona consistency and paralinguistic portrayal.
- **M3-SLU** (2510.19358): 12K+ instances for multi-speaker multi-turn. Key finding: models capture *what* was said but fail to identify *who*.
- **Full-Duplex Survey** (2509.14515): Identifies synchronous data scarcity as critical challenge.

### Known Degradation Patterns
1. **Catastrophic forgetting / LLM Dormancy**: Now quantified by VERA (74.8% → 6.1% on math), documented by Fun-Audio-Chat, EchoX
2. **Context exhaustion**: At 50 Hz x 8 codebooks, 1 min = 24K tokens. Even at 5Hz (U-Codec), context length remains limiting
3. **Coherence-efficiency tradeoff**: Models generate longer responses to stay coherent (MTalk-Bench)
4. **Voice drift**: GPT-4o shows voice consistency drift in >30 min conversations
5. **Speaker attribution failure**: Models understand content but lose track of who said what in multi-speaker scenarios (M3-SLU)
6. **Instruction retention decay**: Audio MultiChallenge shows Self Coherence and Instruction Retention degrade across turns even in SOTA models
7. **Decoder degradation**: Your LLaMA-Omni implementation shows 7-8s coherence before degrading (likely decoder-size related: 1024 vs paper's 4096 hidden dim)

### Which Models Handle Multiturn Best
- **CSM (Sesame)**: Only model that *improves* with more conversation history (conditioning on prior turns)
- **Qwen3-Omni**: Thinker-Talker MoE, 32/36 benchmarks beat Gemini 2.5 Pro. MoE provides capacity without compute scaling per turn
- **Qwen2.5-Omni**: Thinker maintains full LLM context; Talker conditions per-turn from Thinker hidden states
- **Fun-Audio-Chat**: Core-Cocktail Training explicitly designed to preserve multiturn text capabilities during speech training
- **Freeze-Omni**: Frozen LLM preserves text-model multiturn capability intact
- **LLaMA-Omni 2**: Explicitly trained on 1-5 turn conversations with varied speaker timbres
- **MGM-Omni** (2509.25131): Chunk-based parallel decoding for long-horizon speech with stable timbre

### Mitigations for Multiturn Degradation (New in H2 2025)
1. **Core-Cocktail Training** (Fun-Audio-Chat): Two-stage fine-tuning with intermediate model merging preserves text LLM knowledge
2. **Reasoning Distillation** (Step-Audio-R1): Transfer text reasoning to speech pathway explicitly
3. **Think-While-Talking** (STITCH): Interleave silent reasoning chunks to maintain quality
4. **Echo Training** (EchoX): Dynamic speech targets from semantic representations
5. **Dual-Resolution Representations** (Fun-Audio-Chat): 5Hz for context + 25Hz for quality, cutting GPU 50%
6. **Latent Reasoning in Duplex** (FLAIR): Recursive latent embeddings for concurrent reasoning and listening

### What's Still Missing
- No benchmark testing degradation over 10, 20, 50+ turns systematically
- No standardised "conversation stability score" despite SDiaReward making progress
- FD-Bench shows all duplex models fail under noisy/frequent interruptions
- Prosody/emotion consistency across turns remains unquantified

---

## 5. Implications for This Project

### For your LLaMA-Omni re-implementation
- Your 1024 hidden dim decoder (vs paper's 4096) is the likely cause of 7-8s coherence limit — scaling this up should be priority
- LLaMA-Omni 2's AR decoder (UTMOS 4.19 vs CTC's 3.93) is a stronger alternative if you're considering decoder ablations
- Freeze-Omni's approach is directly relevant for your Qwen3-30B run — keep the LLM frozen
- **Fun-Audio-Chat's Core-Cocktail Training** is now the best recipe for training speech on large LLMs without forgetting
- **CTC-TTS** (Feb 2026) shows CTC alignment can be used in LLM-TTS, relevant to your CTC decoder

### For the workbench evaluation
- VoiceBench is now just one of 20+ benchmarks. At minimum add: MTalk-Bench (multiturn), VERA (reasoning), Audio MultiChallenge (multi-turn rubrics)
- The cascade baseline *will* outperform your E2E model on reasoning tasks — VERA quantifies why (74.8% → 6.1%)
- X-Talk's finding (optimised cascade matches E2E) should inform your cascade baseline design
- **SDiaReward** provides a reward model for multiturn dialogue evaluation — consider using it
- Consider evaluating Qwen3-Omni as the Thinker-Talker MoE reference point (Apache 2.0, open weights)

### Most promising extension architectures
1. **Thinker-Talker MoE** (Qwen3-Omni): Proven at scale, SOTA on 32/36 benchmarks, open weights
2. **LLM-native TTS** (Qwen3-TTS / CLEAR as decoder): Production-grade, sub-100ms latency
3. **Frozen-LLM + Core-Cocktail** (Freeze-Omni + Fun-Audio-Chat recipe): For preserving Qwen3-30B quality
4. **Think-While-Talking** (STITCH): For reasoning tasks, ~15% math improvement at zero latency cost
5. **Extreme codecs** (TaDiCodec / U-Codec): For maximising context window in multiturn
6. **CLEAR** (continuous latent AR): 1.88% WER, challenges assumption that discrete tokens are necessary

---

## 6. Full Reference Catalogue

### Tier 1: Must-Read (11 papers)
| Paper | arXiv | Date | Key Contribution |
|-------|-------|------|-----------------|
| Qwen3-Omni | 2509.17765 | Sep 2025 | Thinker-Talker MoE, SOTA on 32/36 benchmarks |
| Qwen2.5-Omni | 2503.20215 | Mar 2025 | Thinker-Talker architecture (original) |
| Step-Audio-R1 | 2511.15848 | Nov 2025 | First audio reasoning model |
| VERA | 2509.26542 | Sep 2025 | Quantifies speech reasoning gap (74.8% → 6.1%) |
| LLaMA-Omni | 2409.06666 | Sep 2024 | Accessible encoder-LLM-decoder baseline |
| LLaMA-Omni 2 | 2505.02625 | May 2025 | AR streaming decoder, multiturn |
| Moshi | 2410.00037 | Sep 2024 | Full-duplex, Inner Monologue, Mimi codec |
| Kimi-Audio | 2504.18425 | Apr 2025 | 13M hour pretraining, dual tokenizer |
| F5-TTS | 2410.06885 | Oct 2024 | Flow matching TTS, no alignment needed |
| MaskGCT | 2409.00750 | Sep 2024 | Fully NAR mask-and-predict TTS |
| Freeze-Omni | 2411.00774 | Nov 2024 | Frozen LLM speech adaptation |

### Tier 2: Architecturally Important (20 papers)
| Paper | arXiv | Date | Key Contribution |
|-------|-------|------|-----------------|
| FLAIR | 2603.17837 | Mar 2026 | Think-while-listening full-duplex |
| DuplexCascade | 2603.09180 | Mar 2026 | VAD-free micro-turns, SOTA duplex |
| Fun-Audio-Chat | 2512.20156 | Dec 2025 | Core-Cocktail Training for catastrophic forgetting |
| Qwen3-TTS | 2601.15621 | Jan 2026 | Production LLM-native TTS, 5M hours |
| STITCH | 2507.15375 | Jul 2025 | Think-while-talking reasoning |
| EchoX | 2509.09174 | Sep 2025 | Acoustic-semantic gap mitigation |
| CLEAR | 2508.19098 | Aug 2025 | Continuous latent AR TTS, 1.88% WER |
| X-Talk | 2512.18706 | Dec 2025 | Optimised cascade = competitive with E2E |
| MiMo-Audio | 2512.23808 | Dec 2025 | 100M+ hours, few-shot audio LLM |
| Dragon-FM | 2507.22746 | Jul 2025 | AR + flow matching TTS |
| MOSS-Audio-Tokenizer | 2602.10934 | Feb 2026 | 1.6B param foundation codec |
| OmniFlatten | 2410.17799 | Oct 2024 | Full-duplex via flattening only |
| IntrinsicVoice | 2410.08035 | Oct 2024 | GroupFormer for length matching |
| GLM-4-Voice | 2412.02612 | Dec 2024 | Interleaved text-speech, ultra-low bitrate |
| Step Audio | 2502.11946 | Feb 2025 | 130B scale, dialect/emotion control |
| CSM (Sesame) | — (blog) | 2025 | Context-conditioned speech quality |
| Orpheus TTS | — (GitHub) | 2025 | LLM-native TTS, Llama-3.2 |
| Spark-TTS | 2503.01710 | Mar 2025 | BiCodec, chain-of-thought TTS |
| CosyVoice 2/3 | 2412.10117 | Dec 2024 | FSQ, streaming flow matching |
| Llasa | 2502.04128 | Feb 2025 | Llama-aligned TTS, inference-time scaling |

### Tier 3: Foundational / Historical
| Paper | arXiv | Date | Key Contribution |
|-------|-------|------|-----------------|
| SpeechGPT | 2305.11000 | May 2023 | First speech I/O LLM |
| SpeechGPT-Gen | 2401.13527 | Jan 2024 | Chain-of-Information generation |
| AudioPaLM | 2306.12925 | Jun 2023 | Text pretraining benefits speech |
| SALMONN | 2310.13289 | Oct 2023 | Dual encoder, activation tuning |
| Qwen2-Audio | 2311.07919 | Nov 2023 | Understanding-only, multi-task |
| Mini-Omni | 2408.16725 | Aug 2024 | First open real-time E2E |
| Mini-Omni 2 | 2410.11190 | Oct 2024 | Vision + audio + interruption |
| VITA | 2408.05211 | Aug 2024 | Omni-modal, external TTS |
| SLAM-Omni | 2412.15649 | Dec 2024 | Timbre control, single-stage training |
| LSLM | 2408.02622 | Aug 2024 | Listen-while-speak, fusion strategies |
| SpiritLM | 2402.05755 | Feb 2024 | Interleaved text-speech tokens |
| dGSLM | 2203.16502 | Mar 2022 | First dual-tower dialogue model |
| SALMONN-Omni | 2411.18138 | Nov 2024 | Codec-free full-duplex |
| Ichigo | 2410.15316 | Oct 2024 | Early fusion, 111ms latency |
| LLMVoX | 2503.04724 | Mar 2025 | 30M LLM-agnostic TTS adapter |

### Full-Duplex Models (H2 2025 - Q1 2026)
| Paper | arXiv | Date | Key Innovation |
|-------|-------|------|---------------|
| FLAIR | 2603.17837 | Mar 2026 | Think-while-listening, latent reasoning |
| SoulX-Duplug | 2603.14877 | Mar 2026 | Plug-and-play streaming state prediction |
| DuplexCascade | 2603.09180 | Mar 2026 | VAD-free micro-turns |
| Discourse-Aware Dual-Track | 2602.23266 | Feb 2026 | Listen+think+speak simultaneously |
| PersonaPlex (NVIDIA) | 2602.06053 | Jan 2026 | Voice cloning + role control |
| F-Actor | 2601.11329 | Jan 2026 | Instruction-controllable duplex |
| Chroma 1.0 | 2601.11141 | Jan 2026 | Sub-second E2E + voice cloning |
| Covo-Audio (Tencent) | 2602.09823 | Feb 2026 | Tencent full-duplex system |
| Conversational Behavior Foundation | 2602.11065 | Feb 2026 | Graph-of-Thoughts, multi-level |
| X-Talk | 2512.18706 | Dec 2025 | Optimised cascade for duplex |
| Mind-Paced Speaking | 2510.09592 | Oct 2025 | Dual-brain reasoning + speaking |
| Phoenix-VAD | 2509.20410 | Sep 2025 | LLM-based semantic endpoint detection |
| FLM-Audio | 2509.02521 | Sep 2025 | Natural monologues for duplex |
| RoboEgo | 2506.01934 | Jun 2025 | 80ms theoretical duplex latency |
| NTPP | 2506.00975 | Jun 2025 | Next-token-pair prediction (ICML 2025) |

### TTS Models (H2 2025 - Q1 2026)
| Paper | arXiv | Date | Key Innovation |
|-------|-------|------|---------------|
| Qwen3-TTS | 2601.15621 | Jan 2026 | Dual-track LM, 5M+ hrs, 10 languages |
| GLM-TTS | 2512.14291 | Dec 2025 | RL-optimised, 100K hours, LoRA voice |
| IndexTTS 2.5 | 2601.03888 | Jan 2026 | Multilingual emotional, GRPO |
| CLEAR | 2508.19098 | Aug 2025 | Continuous latent AR, 1.88% WER |
| Dragon-FM | 2507.22746 | Jul 2025 | AR + flow matching hybrid |
| ARCHI-TTS | 2602.05207 | Feb 2026 | 1.98% WER, semantic aligner |
| DiSTAR | 2510.12210 | Oct 2025 | AR + masked diffusion |
| CaT-TTS | 2509.22062 | Sep 2025 | S3Codec, dual-Transformer |
| Vevo2 | 2508.16332 | Aug 2025 | Unified speech + singing |
| VoiceCraft-X | 2511.12347 | Nov 2025 | Multilingual editing + cloning |
| TADA (Hume AI) | 2602.23068 | Feb 2026 | Text-acoustic synchronisation |
| CTC-TTS | 2602.19574 | Feb 2026 | CTC alignment for LLM-TTS |
| DiFlow-TTS | 2509.09631 | Sep 2025 | Discrete flow matching, 34x faster |
| DMOSpeech 2 | 2507.14988 | Jul 2025 | RL for duration prediction |
| Multi-Reward GRPO | 2511.21270 | Nov 2025 | RL for TTS stability/prosody |
| UniVoice | 2510.04593 | Oct 2025 | Unified ASR + TTS in one LLM |
| JoyVoice | 2512.19090 | Dec 2025 | 8-speaker long-context synthesis |
| Streaming Boundary-Aware | 2603.06444 | Mar 2026 | 66.2% WER reduction in streaming |
| ChipChat (Apple) | 2509.00078 | Aug 2025 | On-device sub-second latency |
| MGM-Omni | 2509.25131 | Sep 2025 | Long-horizon streaming + voice cloning |

### Codecs (including H2 2025 - Q1 2026)
| System | arXiv | Date | Key Property |
|--------|-------|------|-------------|
| MOSS-Audio-Tokenizer | 2602.10934 | Feb 2026 | 1.6B params, 3M hours, foundation codec |
| UniAudio 2.0 | 2602.04683 | Feb 2026 | ReasoningCodec, 100B tokens |
| DashengTokenizer | 2602.23765 | Feb 2026 | Continuous, semantic-first, 22 tasks |
| DyCAST | 2601.23174 | Feb 2026 | Character-aligned variable-rate |
| Kanade | 2602.00594 | Jan 2026 | Single-layer disentangled |
| TaDiCodec | 2508.16790 | Aug 2025 | 6.25 Hz, 0.0875 kbps, text-guided |
| U-Codec (Tencent) | 2510.16718 | Oct 2025 | 5 Hz, 32-layer RVQ, 3x faster LLM-TTS |
| HH-Codec | 2507.18897 | Jul 2025 | 24 tok/s, 0.3 kbps, single quantizer |
| MSR-Codec | 2509.13068 | Sep 2025 | 4-stream disentangled |
| FuseCodec | 2509.11425 | Sep 2025 | Cross-modal semantic alignment |
| FlexiCodec | 2510.00981 | Oct 2025 | Dynamic 3-12.5 Hz frame rate |
| SACodec | 2512.20944 | Dec 2025 | Semantic anchor, 1.5 kbps |
| DisCodec | 2512.13251 | Dec 2025 | Content/prosody/timbre separation |
| LongCat-Audio-Codec | 2510.15227 | Oct 2025 | 16.67 Hz, industrial-grade |
| Q2D2 | 2512.01537 | Dec 2025 | Geometric 2D quantisation |
| Mimi | 2410.00927 | Sep 2024 | 12.5 Hz, semantic-acoustic split, 1.1 kbps |
| WavTokenizer | 2408.16532 | Aug 2024 | Single codebook, 40 tok/s |
| SpeechTokenizer | 2308.16692 | Aug 2023 | HuBERT-distilled first level |
| SNAC | 2410.14411 | Oct 2024 | Multi-scale temporal resolution |
| EnCodec | 2210.13438 | Oct 2022 | RVQ standard |
| DAC | 2306.06546 | Jun 2023 | Snake activations |
| Vocos | 2306.00814 | Jun 2023 | iSTFT vocoder, 10x faster |

### Benchmarks (including H2 2025 - Q1 2026)
| Benchmark | arXiv | Date | What It Tests |
|-----------|-------|------|--------------|
| Audio MultiChallenge | 2512.14865 | Dec 2025 | Multi-turn (Gemini 3 Pro: 54.65%) |
| VERA | 2509.26542 | Sep 2025 | Voice reasoning (74.8%→6.1% gap) |
| SDiaReward | 2603.14889 | Mar 2026 | Multi-turn spoken dialogue reward model |
| FLEXI | 2509.22243 | Sep 2025 | Full-duplex interaction |
| FD-Bench | 2507.19040 | Jul 2025 | Full-duplex pipeline (40+ hrs generated) |
| HumDial Challenge | 2601.05564 | Jan 2026 | ICASSP 2026 full-duplex challenge |
| MTalk-Bench | 2508.18240 | Aug 2025 | Multi-turn S2S dialogue |
| VoxRole | 2509.03940 | Sep 2025 | Persona consistency (13K dialogues) |
| VoiceAgentBench | 2510.07978 | Oct 2025 | Agentic tasks (cascade still wins) |
| VoiceAgentEval | 2510.21244 | Oct 2025 | Outbound calling, 30 sub-scenarios |
| WildSpeech-Bench | 2506.21875 | Jun 2025 | Real-world conversational |
| AHELM | 2508.21376 | Aug 2025 | 10-dimension holistic evaluation |
| MMAU-Pro | 2508.13992 | Aug 2025 | 49 skills, audio general intelligence |
| MMAR | 2505.13032 | May 2025 | Deep reasoning in audio |
| MultiVox | 2507.10859 | Jul 2025 | Multimodal voice assistant |
| VoiceAssistant-Eval | 2509.22651 | Sep 2025 | 10K+ examples, 13 categories |
| M3-SLU | 2510.19358 | Oct 2025 | Speaker attribution in multi-turn |
| SACRED-Bench | 2511.10222 | Nov 2025 | Audio adversarial attacks (66% on Gemini) |
| MedVoiceBias | 2511.06592 | Nov 2025 | Clinical decision bias from voice |
| SpeechJudge | 2511.07931 | Nov 2025 | 99K human feedback pairs |
| Vox-Evaluator | 2510.20210 | Oct 2025 | Multi-level zero-shot TTS stability |
| RVCBench | 2602.00443 | Feb 2026 | Voice cloning robustness |
| mSTEB | 2506.08400 | Jun 2025 | Multilingual speech+text |
| MAEB | 2602.16008 | Feb 2026 | 30 tasks, 100+ languages, 50+ models |
| VCB Bench | 2510.11098 | Oct 2025 | Chinese audio-grounded LLM |
| Stream RAG / AudioCRAG | 2510.02044 | Oct 2025 | Streaming tool use in dialogue |
| VoiceBench | 2410.17196 | Oct 2024 | Single-turn quality + robustness |
| SD-Eval | 2406.13340 | Jun 2024 | Paralinguistic understanding |
| AudioBench | 2406.16020 | Jun 2024 | 8 tasks, 26 datasets |
| AIR-Bench | 2402.07729 | Feb 2024 | Audio information retrieval |
| Dynamic-SUPERB | 2309.09510 | Sep 2023 | 33 tasks, seen vs unseen |

### Surveys
| Survey | arXiv | Date | Venue |
|--------|-------|------|-------|
| Full-Duplex Survey | 2509.14515 | Sep 2025 | — |
| Holistic LALM Evaluation | 2505.15957 | May 2025 | — |
| Recent Advances in Speech LMs | 2410.03751 | Oct 2024 | ACL 2025 |
| WavChat | 2411.13577 | Nov 2024 | — |
| Speech LLMs for Understanding | 2410.18908 | Oct 2024 | IEEE JSTSP |
| LLM-Speech Integration | 2502.19548 | Feb 2025 | ACL 2025 Findings |

### Speech Reasoning & Forgetting
| Paper | arXiv | Date | Key Contribution |
|-------|-------|------|-----------------|
| Step-Audio-R1 | 2511.15848 | Nov 2025 | Modality-Grounded Reasoning Distillation |
| VERA | 2509.26542 | Sep 2025 | Quantifies speech reasoning gap |
| STITCH | 2507.15375 | Jul 2025 | Think-while-talking |
| Mind-Paced Speaking | 2510.09592 | Oct 2025 | Dual-brain reasoning + speaking |
| EchoX | 2509.09174 | Sep 2025 | Acoustic-semantic gap mitigation |
| Fun-Audio-Chat | 2512.20156 | Dec 2025 | Core-Cocktail Training |
| EmotionThinker | 2601.15668 | Jan 2026 | Emotion reasoning, ICLR 2026 oral |

### Proprietary / Unreleased
| Model | Status | Notes |
|-------|--------|-------|
| GPT-4o/5o | Proprietary | No technical report on speech. Used as baseline everywhere. |
| Gemini 2.5 Pro/3 Pro | Proprietary | Leads on 5/10 AHELM dims. 54.65% Audio MultiChallenge. 66% SACRED attack rate. |
| Sesame/CSM | Blog only | No post-March 2025 academic publications found. |
| Anthropic voice | — | No speech/voice technical reports. |
