# Full-Duplex and Voice-Native Speech Foundation Models: Research Survey

**Date compiled:** 2026-03-19

---

## 1. Moshi (Kyutai Labs)

**Full Title:** Moshi: A Speech-Text Foundation Model for Real-Time Dialogue

**Authors:** Alexandre Defossez, Laurent Mazare, Manu Orsini, Amelie Royer, Patrick Perez, Herve Jegou, Edouard Grave, Neil Zeghidour

**Date:** September 17, 2024 (revised October 2, 2024)

**arXiv ID:** 2410.00037

### Architecture

Moshi uses a two-level RQ-Transformer hierarchy:

- **Temporal Transformer (large):** Processes across time steps (the S dimension), producing context vectors for each frame.
- **Depth Transformer (small):** Predicts K sub-sequences hierarchically at each time step, reducing per-timestep predictions from K*S to S+K steps.

The backbone is **Helium**, a 7B-parameter Transformer pretrained on 2.1 trillion tokens of curated English text. It uses RMS normalization, RoPE positional embeddings, 4096-token context, FlashAttention, and Gated Linear Units with SiLU activation.

### Audio Codec: Mimi

Mimi is a causal SeaNet-based autoencoder with residual vector quantization:

- **Frame rate:** 12.5 Hz
- **Quantizers:** Q=8, each with codebook size N_A=2048
- **Bitrate:** 1.1 kbps
- **Innovation — Split RVQ:** One vector quantizer handles semantic tokens while an RVQ with 7 levels handles acoustic details in parallel. This avoids quality degradation from competing semantic-acoustic constraints.
- **Training:** Adversarial-only training (feature + discriminator loss, no reconstruction loss). Quantization applied 50% of the time per-sequence during training.
- **Streaming:** Transformer bottlenecks (8 layers, pre and post-quantization) with finite 250-frame context enable streaming.

### Duplex Mechanism

Moshi models two parallel audio streams (user + system) simultaneously. Given two audio streams (A_{t,q}) and (A'_{t,q}), both receive acoustic delay and are concatenated into a single joint sequence V. This eliminates explicit turn boundaries entirely.

The final joint sequence contains 17 codebook streams:

- k=1: Text tokens (Inner Monologue)
- k=2: Moshi's semantic tokens
- k=3-9: Moshi's delayed acoustic tokens
- k=10: User's semantic tokens
- k=11-17: User's delayed acoustic tokens

An **acoustic delay** tau between semantic and acoustic tokens improves generation quality by reducing inter-token dependencies.

### Inner Monologue

Moshi predicts time-aligned text tokens (from SentencePiece tokenizer applied to Whisper transcriptions) as prefixes to audio tokens. Special tokens (PAD, EPAD) handle word boundaries and padding. A single delay hyperparameter switches between ASR mode (negative delay, ground-truth audio) and TTS mode (positive delay, text-conditioned).

### Latency

- **Theoretical:** 160 ms (80 ms codec frame size + processing)
- **Practical:** 200 ms
- **Reference:** Average human conversational response latency is ~230 ms across 10 languages.

### Training Data


| Type                       | Size                | Details                                                                           |
| -------------------------- | ------------------- | --------------------------------------------------------------------------------- |
| Unsupervised audio         | 7 million hours     | Transcribed with Whisper v3                                                       |
| Fisher dataset             | 2,000 hours         | Phone conversations, separate channels                                            |
| Supervised multi-speaker   | 170 hours           | Multi-stream dialogue recordings                                                  |
| Synthetic instruction data | 20,000+ hours       | Generated via fine-tuned Helium + multi-stream TTS, ~70 voice styles              |
| Text pretraining           | 2.1 trillion tokens | 12.5% curated (Wikipedia, StackExchange, scientific) + 87.5% filtered CommonCrawl |


**Compute:** 127 DGX nodes (1,016 H100 GPUs) provided by Scaleway.

Training proceeds in stages: unsupervised -> diarization-based -> Fisher -> instruction tuning.

### Turn-Taking

True full-duplex: the model speaks and listens simultaneously, generating natural silence when inactive. Overlapping speech, interruptions, and backchanneling are natively supported through parallel stream modeling — no explicit turn-taking logic is needed.

### Coherence and Stability

- Context window supports ~5 minutes of conversation.
- RoPE positional embeddings and progressive multi-stage training improve long-context naturalness.
- **Known limitations:** Relatively limited capabilities; suitable for casual conversation, facts, and roleplay only. Single voice output (to prevent impersonation). Middle-range toxicity. Bias toward over-represented training topics. No tool access. English only.

---

## 2. dGSLM (Meta AI)

**Full Title:** Generative Spoken Dialogue Language Modeling

**Authors:** Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoit Sagot, Abdelrahman Mohamed, Emmanuel Dupoux

**Date:** March 30, 2022 (revised November 22, 2022)

**arXiv ID:** 2203.16502

### Architecture

dGSLM is a **textless** spoken dialogue generation system with three components:

1. **Speech-to-Unit Encoder (Fisher HuBERT):** A HuBERT model trained on the Fisher dataset. Uses **layer 12 features** with **k-means clustering into 500 units** to convert raw two-channel audio into parallel streams of discrete tokens.
2. **Dialogue Language Model (SpeechDLM):** A **dual-tower transformer with cross-attention**.
  - Each tower processes one speaker's channel autoregressively.
  - Cross-attention layers (4 decoder-cross-layers) allow each tower to attend to the other speaker's representations.
  - Training config: dropout 0.1, Adam optimizer (betas 0.9/0.98), LR 0.0005 with inverse square root schedule, ~250K updates with 20K warmup, max-tokens 18432.
  - Includes three auxiliary objectives: **Edge Unit Prediction**, **Delayed Duration Prediction**, and **Duration Prediction** — these help model turn boundaries and speech rhythm.
3. **Unit-to-Speech Decoder:** HiFiGAN vocoder synthesizes waveforms from discrete units.

### Duplex Mechanism

The two towers process both channels simultaneously with cross-attention, allowing the model to capture when speakers overlap, interrupt, or backchannel. Edge Unit Prediction specifically helps the model learn where speaker turns begin and end. The model does not require explicit voice activity detection or turn segmentation.

### Audio Tokenization

- HuBERT layer 12 features
- 500 k-means clusters
- Separate unit sequences for each channel (unitA, unitB)
- No explicit duration tokens in the base units; duration is modeled via the Delayed Duration Prediction objective

### Training Data

- **Fisher dataset:** 2,000 hours of two-channel raw conversational telephone audio
- **No text or labels used** — entirely textless/unsupervised

### Turn-Taking

The dual-tower cross-attention plus edge unit prediction allows the model to generate naturalistic turn-taking patterns including overlapping speech, backchannels, and laughter. Evaluations showed more natural turn-taking compared to text-based cascaded baselines (ASR -> text LM -> TTS).

### Latency

Not explicitly reported as a real-time system. dGSLM is primarily a generative model for producing dialogue samples, not designed for interactive real-time use.

### Coherence and Stability

- Generates paralinguistic signals (laughter, backchannels) across both channels simultaneously.
- Progressive model variants (DLM0 through DLM5/dGSLM) showed improvement in naturalness.
- **Limitations:** Primarily a research/generative model, not an interactive dialogue system. The 2,000-hour training set is relatively small. The textless approach captures prosodic and turn-taking patterns well but has limited semantic coherence compared to text-grounded models.

---

## 3. SpiritLM (Meta AI)

**Full Title:** Spirit LM: Interleaved Spoken and Written Language Model

**Authors:** Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R. Costa-jussa, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, Itai Gat, Mary Williamson, Gabriel Synnaeve, Juan Pino, Benoit Sagot, Emmanuel Dupoux

**Date:** February 8, 2024 (revised October 18, 2024)

**arXiv ID:** 2402.05755

### Architecture

SpiritLM extends **Llama 2 (7B)** to handle interleaved text and speech tokens via continued training. Two variants exist:

- **SpiritLM-Base:** Uses HuBERT phonetic tokens (501-unit vocabulary) for speech representation, with BPE subword tokens for text.
- **SpiritLM-Expressive:** Adds pitch tokens (64 codebook, VQ-VAE based) and style tokens (100 units, k-means clustering on speechprop features) alongside HuBERT phonetic tokens to capture expressivity.

### Token Interleaving

Speech and text sequences are concatenated as a single stream of tokens with **word-level interleaving** during training. This is trained on a small automatically-curated speech-text parallel corpus, enabling the model to learn alignment between speech and text modalities.

### Training Data and Strategy


| Type                | Size                         |
| ------------------- | ---------------------------- |
| Text-only           | 300B tokens                  |
| Speech-only         | 30B tokens                   |
| Aligned speech+text | 7B speech + 1.4B text tokens |


Ablations confirmed that interleaving training helps the model learn speech-text alignment.

### Duplex Mechanism

**SpiritLM does NOT support full-duplex operation.** It processes sequential token streams without mechanisms for simultaneous bidirectional speech or conversation management. It is a foundational speech-text LM, not an interactive dialogue system.

### Latency

Not reported; the model is designed for offline generation rather than real-time interaction.

### Turn-Taking

No explicit turn-taking mechanism. The model generates sequentially and does not handle overlapping speech or barge-in.

### Key Results


| Task                             | Metric         |
| -------------------------------- | -------------- |
| ASR (LibriSpeech clean, 10-shot) | 21.9 WER       |
| TTS (10-shot)                    | 45.5 CER       |
| StoryCloze (cross-modal)         | 88.6% accuracy |


### Coherence and Stability

- Strong cross-modal transfer: the model can switch between speech and text mid-sequence.
- Few-shot capability for ASR, TTS, and speech classification.
- **Limitations:** The expressive version shows moderate degradation in lexical, grammatical, and semantic understanding compared to the base version. Not designed for interactive conversation or multi-turn dialogue.

### Relevance to Full-Duplex

SpiritLM is relevant as a foundational model showing how speech and text tokens can be unified in a single LM. It provides the tokenization and interleaving strategy that downstream full-duplex models can build upon, but itself does not address the duplex problem.

---

## 4. SALMONN-Omni

**Full Title:** SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation

**Authors:** Wenyi Yu, Siyin Wang, Xiaoyu Yang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, Chao Zhang

**Date:** November 27, 2024

**arXiv ID:** 2411.18138

### Architecture

Three interconnected components:

1. **Streaming Speech Encoder:** Processes input audio into continuous auditory embeddings in real time.
2. **Large Language Model (LLM):** Core reasoning backbone that processes auditory embeddings and generates word embeddings.
3. **Streaming Speech Synthesizer:** Converts word embeddings to speech output in real time.

All components are interconnected through **continuous embeddings** rather than discrete tokens — this is the codec-free innovation.

### Codec-Free Approach

Unlike Moshi or SyncLLM which quantize audio into discrete codec tokens, SALMONN-Omni operates on continuous embeddings throughout. This eliminates quantization losses and avoids the information bottleneck of discrete codebooks.

### "Thinking" Mechanism

Special tokens manage asynchronous text-speech generation:

- `<think>` tokens: Generated during non-speaking periods or when synthesis lags behind LLM output
- `<start_speak>` / `<end_speak>` tokens: Control transitions between speaking and non-speaking states
- A **negative coefficient lambda_think** is applied to the loss for `<think>` tokens, allowing flexible pacing without forcing repetitive placeholders

### Duplex Mechanism (Synchronization)

Conversations are divided into **fixed-duration time blocks** (delta_t seconds). Within each block:

1. The encoder processes delta_t seconds of input speech
2. The LLM generates n embeddings
3. If in speaking state, word embeddings are sent to the streaming speech synthesizer to generate a response matching the block duration

This periodic synchronization enables the model to listen to its own output and environmental audio concurrently.

### Barge-In Handling

The model transitions between speaking and non-speaking states via `<start_speak>` and `<end_speak>` tokens. It demonstrates **context-dependent barge-in**: it stops mid-response for relevant user interruptions but ignores irrelevant interjections.

### Echo Cancellation

The model can distinguish its own generated speech from user input, remaining unaffected by its own output — a critical capability for full-duplex without acoustic echo cancellation hardware.

### Training Data

- 60K hours of LibriHeavy for speech recognition
- 10K hours of GigaSpeech for additional speech tasks
- Synthetic data for turn-taking and barge-in training

### Latency

No specific latency figures reported in the available materials.

### Coherence and Stability

- Claimed to be the first codec-free full-duplex model.
- Demonstrations cover streaming ASR, speech enhancement, spoken QA, and barge-in scenarios.
- **Limitations:** The paper presents primarily case studies rather than formal benchmarks. Full technical report and model checkpoints were announced as forthcoming. Quantitative evaluation against Moshi or other baselines not provided in the initial release.

---

## 5. Hertz-Dev (Standard Intelligence)

**Full Title:** Hertz-Dev (base model for full-duplex conversational audio)

**Authors / Organization:** Standard Intelligence (SI)

**Date:** Late 2024

**arXiv ID:** None (open-source release, no formal paper)

### Architecture

Two-component system totaling **8.5 billion parameters**:

**hertz-codec (Audio VAE):**

- 5M encoder parameters, 95M decoder parameters
- Processes 6-second, 16kHz mono audio
- Encodes at **8 Hz latent frame rate** with KL-regularized 1 kbps bitrate
- Outputs Gaussian parameters sampled into single **32-dimensional latents** per 125 ms frame
- Uses causal convolutions for streaming compatibility

**hertz-ar (Transformer):**

- 40-layer decoder-only transformer, **8.4B parameters**, 2048-token context (~4.5 minutes)
- First 32 layers: hertz-lm, predicts 15-bit quantized latent projections
- Final 8 layers: predicts future latents
- Can be initialized from a text language model (trained on 2 trillion tokens) or randomly initialized; both approaches effectively learn linguistics

### Audio Codec

**hertz-codec quantizer:** Distills the most phonetically relevant 15 bits of each latent. Achieves lower tokens-per-second than any popular tokenizer. In subjective evaluations, outperforms SoundStream and EnCodec at 6 kbps and matches DAC at 8 kbps.

### Duplex Mechanism

Duplex audio is handled as a post-training task with **two projection heads concatenated together**, then separated into two quantized projection pipelines conditioned on their respective residuals. Processes two-speaker format natively and handles overlapping speech.

### Latency

- **Theoretical average latency:** 80 ms (single sampled latent per timestep)
- **Practical latency:** 120 ms on a single RTX 4090
- **Claimed:** 2x lower than previous state of the art
- **Runtime requirement:** 8 forward passes per second for continuous autoregressive generation
- **Latency breakdown:** 62.5 ms average time between utterance and end of one token + forward pass time + round-trip network delay

### Training Data

Not fully disclosed. The hertz-lm initialization option uses a language model trained on 2 trillion text tokens. Audio training data composition and scale not publicly documented.

### Turn-Taking

Native two-speaker modeling through the dual projection heads. The system processes conversational audio as inherently duplex rather than requiring explicit turn segmentation.

### Coherence and Stability

- Context window of ~4.5 minutes (2048 tokens at 8 Hz)
- **Limitations:** Described as an "open-source base model" — real-time microphone interaction is marked as "currently experimental." No formal evaluation benchmarks published. Training data composition undisclosed.

---

## 6. Ultravox (Fixie AI)

**Full Title:** Ultravox — Speech-native voice AI model

**Authors / Organization:** Fixie AI (now Ultravox AI)

**Date:** August 2024 (v0.4), iterating through v0.7 (December 2025)

**arXiv ID:** None (open-weight model, no formal paper; referenced in third-party study arXiv:2503.19586)

### Architecture

Ultravox is a **multimodal Speech LLM** that converts audio directly into the LLM's high-dimensional space without a separate ASR step:

1. **Speech Encoder:** Whisper-large-v3-turbo encoder (fine-tuned during training)
2. **Multi-modal Adapter/Projector:** Trained bridge layer that maps audio embeddings to text embedding space
3. **LLM Backbone:** Llama 3.3 70B-Instruct (frozen during training). Earlier versions used Llama 3, Mistral, and Gemma.

**Mechanism:** Input includes a text prompt with a special `<|audio|>` pseudo-token. The processor replaces this token with embeddings derived from input audio. Merged embeddings are fed into the LLM for text generation.

### Training Strategy

- **Adapter + encoder trained; LLM frozen** — only the adapter/projector and Whisper encoder are updated.
- **Loss:** Knowledge-distillation loss matching Llama backbone logits.
- **Data:** Mix of ASR datasets extended with continuations generated by Llama 3.1 8B, plus speech translation datasets.
- **Compute:** v0.4 training takes 2-3 hours on 8xH100 GPUs for 14K steps.
- **Precision:** BF16 mixed precision.

### Duplex Mechanism

**Ultravox does NOT natively support full-duplex speech.** It operates as speech-in, text-out (with plans for speech token output via unit vocoders). It is designed for real-time speech understanding rather than simultaneous bidirectional speech.

However, its architecture enables very low-latency speech understanding by eliminating the ASR bottleneck of cascaded systems, making it suitable as a component in duplex voice pipelines.

### Latency

No explicit latency benchmarks published. The direct audio-to-LLM projection avoids ASR transcription latency, which is the key architectural advantage over cascaded approaches.

### Audio Tokenization

Uses Whisper encoder features projected through a learned adapter — continuous embedding approach rather than discrete codec tokens.

### Evaluation Results (v0.5 70B)


| Benchmark       | Score      |
| --------------- | ---------- |
| Big Bench Audio | 82.70      |
| CoVoST2 es_en   | 43.29 BLEU |
| CoVoST2 ru_en   | 48.99 BLEU |


### Turn-Taking

Not inherently handled. Ultravox processes speech segments as input and produces text responses. Turn management would need to be handled by an external orchestration layer.

### Coherence and Stability

- Supports 42 languages (v0.5).
- Strong multilingual speech understanding.
- **Limitations:** Text-output only (speech output planned). No native full-duplex. A third-party study (arXiv:2503.19586) found limited sensitivity to speaker characteristics compared to Qwen2-Audio. No preference tuning applied.

### Relevance to Full-Duplex

Ultravox is relevant as a high-quality speech understanding front-end that eliminates ASR latency. In production voice AI systems, it can serve as the listening component in a duplex pipeline, but it does not itself generate speech or manage bidirectional conversation.

---

## 7. LSLM — Language Model Can Listen While Speaking

**Full Title:** Language Model Can Listen While Speaking

**Authors:** Ziyang Ma, Yakun Song, Chenpeng Du, Jian Cong, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen

**Date:** August 5, 2024

**arXiv ID:** 2408.02622

### Architecture

LSLM is a **decoder-only Transformer** with:

- 12 blocks, 12 attention heads, 768 embedding dimensions, 3072 FFN dimensions
- **106M parameters** total
- Two independent channels:
  - **Speaking Channel:** Token-based TTS using single-layer discrete audio tokens
  - **Listening Channel:** Streaming SSL encoder (vq-wav2vec, 34M parameters) for real-time audio input processing

### Fusion Strategies

Three approaches for merging listening and speaking channels:

1. **Early Fusion:** Integrates channels at input embeddings before autoregressive prediction.
2. **Middle Fusion:** Merges at each Transformer block by adding listening features to speaking hidden states. **This proved optimal**, balancing speech generation quality with interactive responsiveness.
3. **Late Fusion:** Combines channels at output logits before softmax.

### Duplex Mechanism

The model maintains separate generation and perception pathways that run simultaneously. The speaking channel generates speech tokens autoregressively while the listening channel continuously processes incoming audio. The fusion mechanism allows real-time audio input to influence the generation process.

### Interruption / Barge-In Handling

An **IRQ (Interruption) token** signals turn-taking detection. The model learns to terminate generation early when turn-taking occurs. Training sets mu=0.5 seconds as the detection interval. During inference, successful turn-taking occurs within [0, 2*mu] seconds after interruption begins.

### Audio Codec / Tokenization

- **Encoder:** vq-wav2vec SSL model converts speech to embeddings
- **Speech tokens:** Single-layer discretization (unlike multi-layer RVQ)
- **Vocoder:** GAN-based token-to-waveform decoder

### Training Data and Strategy


| Component                 | Details                                    |
| ------------------------- | ------------------------------------------ |
| TTS data                  | LibriTTS (585 hours)                       |
| Interruption data         | Speech Commands Dataset                    |
| Noise augmentation        | MUSAN background corpus, 50% probability   |
| Interruption augmentation | Added with 50% probability during training |
| Optimizer                 | AdamW, LR 5e-4, 20 epochs, batch size 4    |


### Evaluation Results

**Command-based Full-Duplex (middle fusion):**


| Condition | TTS WER | Interactive F1 |
| --------- | ------- | -------------- |
| Clean     | 4.05%   | 98.00%         |
| Noisy     | 4.51%   | 97.38%         |


**Voice-based Full-Duplex (unseen speakers):**


| Condition | TTS WER | Interactive F1 |
| --------- | ------- | -------------- |
| Clean     | 5.33%   | 95.50%         |
| Noisy     | 8.50%   | 85.15%         |


### Latency

Not explicitly benchmarked as an end-to-end latency figure. The system operates in a streaming fashion with the listening channel processing audio in real time during generation.

### Turn-Taking

The IRQ token mechanism provides explicit turn-taking detection. The model can stop mid-utterance when it detects user speech, with detection occurring within 0-1 second of interruption onset.

### Coherence and Stability

- **Limitations (stated by authors):** "A long way to go to achieve smooth human-computer speech interaction." Future work needed for speaker-following capabilities and audiovisual co-guidance for improved turn-taking.
- The 106M parameter model is relatively small, limiting semantic sophistication.
- Voice-based detection shows degradation in noisy conditions (F1 drops from 95.5% to 85.15%).

---

## Comparative Analysis

### Duplex Capability Spectrum


| Model            | True Full-Duplex  | Barge-In            | Parallel Streams        | Real-Time       |
| ---------------- | ----------------- | ------------------- | ----------------------- | --------------- |
| **Moshi**        | Yes (native)      | Yes (native)        | Yes (user + system)     | Yes (200ms)     |
| **dGSLM**        | Yes (generative)  | Yes (modeled)       | Yes (dual tower)        | No (offline)    |
| **SpiritLM**     | No                | No                  | No                      | No              |
| **SALMONN-Omni** | Yes (sync blocks) | Yes (context-aware) | Yes (encoder + decoder) | Claimed         |
| **Hertz-Dev**    | Yes (dual heads)  | Yes (native)        | Yes (two-speaker)       | Yes (120ms)     |
| **Ultravox**     | No (listen only)  | No (external)       | No                      | Partial         |
| **LSLM**         | Yes (fusion)      | Yes (IRQ token)     | Yes (listen + speak)    | Yes (streaming) |


### Tokenization Approaches


| Model            | Approach                       | Codec/Tokenizer                      | Rate       | Bitrate  |
| ---------------- | ------------------------------ | ------------------------------------ | ---------- | -------- |
| **Moshi**        | Discrete (RVQ)                 | Mimi (Split RVQ)                     | 12.5 Hz    | 1.1 kbps |
| **dGSLM**        | Discrete (k-means)             | HuBERT L12 + 500 clusters            | ~50 Hz     | N/A      |
| **SpiritLM**     | Discrete (k-means)             | HuBERT 501 units + pitch/style       | ~50 Hz     | N/A      |
| **SALMONN-Omni** | Continuous embeddings          | Codec-free (streaming encoder)       | Continuous | N/A      |
| **Hertz-Dev**    | Continuous latents (quantized) | hertz-codec (VAE + 15-bit quantizer) | 8 Hz       | 1 kbps   |
| **Ultravox**     | Continuous embeddings          | Whisper encoder + adapter            | Continuous | N/A      |
| **LSLM**         | Discrete (single-layer)        | vq-wav2vec                           | Variable   | N/A      |


### Latency Comparison


| Model            | Theoretical  | Practical       | Notes                                 |
| ---------------- | ------------ | --------------- | ------------------------------------- |
| **Moshi**        | 160 ms       | 200 ms          | End-to-end, single GPU                |
| **Hertz-Dev**    | 80 ms        | 120 ms          | Single RTX 4090, claims 2x SOTA       |
| **LSLM**         | Streaming    | Not benchmarked | Real-time listening during generation |
| **SALMONN-Omni** | Not reported | Not reported    | Claims real-time capability           |
| **dGSLM**        | N/A          | N/A             | Offline generative model              |
| **SpiritLM**     | N/A          | N/A             | Offline model                         |
| **Ultravox**     | Not reported | Not reported    | Eliminates ASR latency                |


### Model Scale


| Model            | Parameters                | Training Data                             |
| ---------------- | ------------------------- | ----------------------------------------- |
| **Moshi**        | 7B (Helium) + Mimi        | 7M hrs audio + 2.1T text tokens           |
| **dGSLM**        | Not disclosed             | 2K hrs (Fisher)                           |
| **SpiritLM**     | 7B (Llama 2)              | 300B text + 30B speech tokens             |
| **SALMONN-Omni** | Not disclosed             | 70K hrs audio                             |
| **Hertz-Dev**    | 8.5B total                | Undisclosed audio + 2T text tokens (init) |
| **Ultravox**     | 70B (Llama 3.3) + Whisper | ASR datasets + synthetic continuations    |
| **LSLM**         | 106M + 34M encoder        | 585 hrs (LibriTTS)                        |


---

## Key Architectural Patterns

### Pattern 1: Parallel Stream Modeling (Moshi, dGSLM, Hertz-Dev)

Model both speaker channels as parallel token/latent streams within a single architecture. The model jointly predicts what it says and what the user says. Advantages: native duplex, captures overlaps naturally. Disadvantage: requires multi-channel training data.

### Pattern 2: Dual-Channel Fusion (LSLM)

Maintain separate generation and perception channels with a learned fusion mechanism. The speaking channel generates while the listening channel monitors for interruptions. Advantage: clean separation of concerns, explicit barge-in detection. Disadvantage: fusion strategy is a design bottleneck; middle fusion worked best but the architecture is relatively constrained.

### Pattern 3: Synchronized Time Blocks (SALMONN-Omni)

Divide time into fixed blocks; within each block the encoder listens and the decoder speaks, with periodic synchronization. Advantage: codec-free continuous embeddings. Disadvantage: block boundaries may introduce artificial latency or alignment artifacts.

### Pattern 4: Audio-Conditioned LLM (Ultravox, SpiritLM)

Project audio features into an LLM's embedding space for understanding, without generating speech output. Advantage: leverages powerful pretrained LLMs. Disadvantage: no native speech output or duplex capability.

---

## Open Challenges Across All Models

1. **Long-conversation stability:** Moshi supports ~5 min context; Hertz-Dev ~4.5 min. No model has demonstrated stable, coherent conversations beyond 5-10 minutes.
2. **Semantic coherence under duplex:** Models that handle overlapping speech (Moshi, dGSLM, Hertz-Dev) can generate plausible turn patterns but may lose semantic thread during complex overlaps.
3. **Echo cancellation:** SALMONN-Omni explicitly addresses this; most other models assume clean separated channels or ignore the echo problem.
4. **Training data bottleneck:** True two-channel conversational data is scarce (Fisher: 2K hours is the standard). Moshi addresses this with 20K+ hours of synthetic data. Most models rely heavily on synthetic or single-channel data.
5. **Evaluation standards:** No unified benchmark exists for full-duplex speech dialogue. Models are evaluated on different metrics (WER, F1, subjective quality, turn-taking naturalness) making direct comparison difficult.
6. **Expressivity vs. coherence tradeoff:** SpiritLM's expressive tokens degrade semantic quality. Balancing paralinguistic richness with linguistic accuracy remains unsolved.

