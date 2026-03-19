# Voice/Speech LLM Benchmarks, Evaluation Methods, and Surveys

## Research Compilation (2024-2025)

---

## Part 1: Benchmarks

---

### 1. VoiceBench — Benchmarking LLM-Based Voice Assistants

**Paper:** "VoiceBench: Benchmarking LLM-Based Voice Assistants"
**Authors:** Yiming Chen, Xianghu Yue, Chen Zhang, Xiaoxue Gao, Robby T. Tan, Haizhou Li
**ArXiv:** 2410.17196

#### Evaluation Dimensions

**Capability assessment across three areas:**
- **General Knowledge** (AlpacaEval, CommonEval, SD-QA)
- **Instruction Following** (IFEval with format-specific requirements)
- **Safety** (AdvBench with harmful prompts)

**Robustness across three variation types:**
- **Speaker variations:** speaking speed, age, pitch, volume, accent
- **Environmental variations:** far-field speech, distortion, reverberation, packet loss, background noise
- **Content variations:** disfluencies (repairs, repetitions, pauses), mispronunciations, grammatical errors

#### Dataset Composition
- 1,817 total samples
- Real spoken data: CommonEval (200), SD-QA (553)
- Synthetic spoken data: AlpacaEval (199), IFEval (345), AdvBench (520)
- Synthetic generation pipeline: text instructions -> GPT-4o normalization -> Google TTS API

#### Models Evaluated and Rankings

| Rank | Model               | Text Score | Speech Score |
|------|---------------------|-----------|-------------|
| 1    | Naive (Whisper+LLaMA-3.1) | 80.27 | 74.50 |
| 2    | DiVA               | 81.14     | 64.02       |
| 3    | Qwen2-Audio         | 72.72     | 59.83       |
| 4    | VITA                | 75.14     | 39.33       |
| 5    | LLaMA-Omni          | 69.26     | 40.21       |
| 6    | Mini-Omni           | 57.11     | 41.56       |

#### Key Findings
1. **Pipeline models dramatically outperform end-to-end systems** on speech inputs by 10+ points
2. LLaMA-Omni shows 11-point degradation despite using the same base LLM as the Naive pipeline
3. Text-to-speech performance gaps vary widely: Naive drops 5.77 points; VITA drops 35+ points
4. **Safety concern:** Several models (e.g., Mini-Omni) fail to reject malicious instructions delivered as speech
5. **Accent challenges:** Low-resource accents (Indian, Philippine English) cause marked degradation vs. high-resource accents (Australian, British, US)
6. **Environmental robustness:** End-to-end models are more susceptible to noisy conditions than pipeline systems
7. **Content robustness:** Repairs cause highest degradation (12.55% average). VITA showed 33.60% degradation on mispronunciations
8. **No multiturn evaluation** -- focuses exclusively on single-turn spoken instructions

---

### 2. AudioBench — Universal Benchmark for Audio LLMs

**Paper:** "AudioBench: A Universal Benchmark for Audio Large Language Models"
**Authors:** Bin Wang, Xunlong Zou, Geyu Lin, Shuo Sun, Zhuohan Liu, Wenyu Zhang, Zhengyuan Liu, AiTi Aw, Nancy F. Chen
**ArXiv:** 2406.16020

#### Evaluation Dimensions: 8 Tasks, 26 Datasets

**Speech Understanding (15 datasets):**
- ASR (9): LibriSpeech-Clean/Other, CommonVoice-15, PeoplesSpeech, GigaSpeech, Tedlium3, Tedlium3-Longform, Earnings-21/22
- Speech QA (4): CN-College-Listen, SLUE-P2-SQA5, DREAM-TTS, Public-SG-SpeechQA
- Speech Instruction (2): OpenHermes-Audio, ALPACA-Audio

**Audio Scene Understanding (5 datasets):**
- Audio QA (3): Clotho-AQA, WavCaps-QA, AudioCaps-QA
- Audio Captioning (2): WavCaps, AudioCaps

**Voice Understanding (6 datasets):**
- Emotion: IEMOCAP-Emotion, MELD-Sentiment, MELD-Emotion
- Accent: VoxCeleb1-Accent
- Gender: VoxCeleb1-Gender, IEMOCAP-Gender

**7 newly proposed datasets:** CN-College-Listen, DREAM-TTS, Public-SG-SpeechQA, OpenHermes-Audio, ALPACA-Audio, WavCaps-QA, AudioCaps-QA

#### Models Evaluated
- SALMONN, Qwen-Audio-Chat, WavLLM, Qwen2-Audio-Instruct
- Cascade baseline: Whisper-Large-V3 + Llama-3-8B

#### Evaluation Metrics
- ASR tasks: Word Error Rate (WER)
- Audio captioning: METEOR score
- Open-ended generation: Model-as-Judge (Llama-3-70B-Instruct with >0.85 Spearman correlation with GPT-4)
- All scores rescaled to 100-point scale

#### Key Findings
1. **No single model excels consistently across all tasks**
2. Qwen2-Audio-Instruct strongest on long-form ASR
3. Whisper+Llama3 pipeline best for speech reasoning, but cannot access paralinguistic features
4. All models struggle with audio >10 minutes
5. **SALMONN "overfitting" problem:** demonstrates task-overfit to certain audio features and ignores proper instructions; phoneme recognition errors with different prompts
6. Cascade models excel at transcription-dependent tasks but fail entirely with non-verbal content
7. Non-speech audio understanding remains unsatisfactory overall

---

### 3. SD-Eval — Spoken Dialogue Understanding Benchmark

**Paper:** "SD-Eval: A Benchmark Dataset for Spoken Dialogue Understanding Beyond Words"
**Authors:** Junyi Ao, Yuancheng Wang, Xiaohai Tian, Dekun Chen, Jun Zhang, Lu Lu, Yuxuan Wang, Haizhou Li, Zhizheng Wu
**ArXiv:** 2406.13340 | **Venue:** NeurIPS 2024

#### Evaluation Dimensions (4 subsets)
- **Emotion (test-emo):** 1,289 utterances; sad, angry, fear, disgust, happy
- **Accent (test-acc):** 4,310 utterances; 9 regional varieties (English, Scottish, Welsh, American, Australian, etc.)
- **Age (test-age):** 1,014 utterances; child vs. adult speakers
- **Environment (test-env):** 690 utterances; 7 background sound scenarios (driving, children's voices, beaches, rain, bells, sports centers, transit)

**Total evaluation:** 7,303 utterances / 8.76 hours
**Training set:** 724.4k utterances / 1,052.72 hours from 11 source datasets

#### Training Set Construction
- Emotion: 100.5k utterances (MSP-Podcast, IEMOCAP, MELD, EmoV-DB, ESD, CREMA-D)
- Accent: 508.6k utterances (UK-Ireland, VCTK, Common Voice)
- Age: 73.2k utterances (MyST corpus)
- Environment: 47.1k utterances (LibriSpeech + AudioCaps + synthesized speech)
- Responses generated using ChatGPT variants

#### Models Evaluated
- Cascade LLM (Whisper large-v3 + InternLM2)
- VS-LLM (end-to-end encoder-adaptor-LLM)
- LLM Upper Bound (text + ground-truth labels)
- Qwen-Audio, Qwen2-Audio, SALMONN

#### Key Findings
1. **VS-LLM consistently outperformed Cascade LLM** -- direct speech input captures paralinguistic information implicitly
2. Models with ground-truth emotion labels achieved 12.35 BLEU-4 vs. 4.66 for content-only baselines
3. **LLM-based metrics outperform traditional metrics for evaluation:**
   - GPT-4o Spearman correlation: 0.731 (emotion), 0.659 (accent)
   - BLEU-4/ROUGE-L: 0.208-0.336 range
   - BERTScore: 0.291
4. **Explicitly limited to single-turn dialogues** -- multi-turn identified as future work

---

### 4. AIR-Bench — Audio Information Retrieval Benchmark

**Paper:** "AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension"
**Authors:** Qian Yang, Jin Xu, Wenrui Liu, et al. (Alibaba Group/Zhejiang University)
**ArXiv:** 2402.07729

#### Evaluation Dimensions

**Foundation Benchmark (19 tasks, ~19,000 single-choice questions):**

*Speech (10 tasks):* speech grounding, spoken language ID, speaker gender recognition, emotion recognition, speaker age prediction, speech entity recognition, intent classification, speaker number verification, synthesized voice detection, audio grounding

*Sound (4 tasks):* audio grounding, vocal sound classification, acoustic scene classification, sound QA

*Music (5 tasks):* instruments classification, genre classification, note analysis (pitch), note analysis (velocity), music QA, music emotion detection

**Chat Benchmark (2,000+ open-ended QA pairs):**
- Speech (800), Sound (400), Music (400), Mixed audio (400)
- Novel mixing strategy with loudness control and temporal dislocation

#### Model Rankings

**Foundation Benchmark (accuracy):**
1. Qwen-Audio Turbo: 57.8%
2. Qwen-Audio-Chat: 54.5%
3. PandaGPT: 40.2%
4. SALMONN: 36.0%

**Chat Benchmark (average score 1-10):**
1. Whisper+GPT-4: 7.54 (speech only)
2. Qwen-Audio Turbo: 6.34
3. SALMONN: 6.11
4. Qwen-Audio-Chat: 6.08

#### Key Findings
1. GPT-4 Turbo as evaluator showed **98.2% consistency with human judgments** (foundation); >70% pairwise preference consistency (chat)
2. GPT-4 exhibits position bias in scoring -- mitigated by averaging reversed evaluations
3. **No model surpasses Whisper+GPT-4 for speech transcription tasks** -- deficiencies in foundational competencies persist despite claimed advanced abilities
4. Models struggle with mixed audio scenarios that combine speech, sound, and music

---

### 5. Dynamic-SUPERB — Dynamic Speech Understanding Benchmark

**Paper:** "Dynamic-SUPERB: Towards A Dynamic, Collaborative, and Comprehensive Instruction-Tuning Benchmark for Speech"
**Authors:** Chien-yu Huang et al. (15 researchers including Shinji Watanabe, Hung-yi Lee)
**ArXiv:** 2309.09510 | **Venue:** ICASSP 2024

#### Evaluation Dimensions (6 categories, 33 tasks, 22 datasets, 55 evaluation instances)
- **Content (CNT):** Speech processing tasks on what is spoken
- **Speaker (SPK):** Speaker verification and identification
- **Semantics (SEM):** Intent classification and semantic understanding
- **Degradation (DEG):** Robustness to audio quality issues
- **Paralinguistics (PRL):** Emotion, accent, voice characteristics
- **Audio (AUD):** Non-speech audio processing

#### Models Evaluated
1. **BERT-GSLM:** Combined BERT + generative spoken language model; poor overall (best: 68.2% degradation on seen tasks)
2. **Whisper:** 95.3% on content seen tasks, struggled with unseen
3. **ImageBind-LLM:** Moderate; "much higher accuracy than random baseline in CNT"
4. **Whisper-LLM:** Best overall on seen tasks -- speaker (91.7%), degradation (91.0%)
5. **ASR-ChatGPT:** Dominated semantics unseen (69.3%) but failed speaker/paralinguistic tasks

#### Key Findings
1. **Significant performance gap between seen and unseen tasks** across all models
2. Models recognize "specific patterns in instructions" rather than understanding task semantics
3. Most models underperform on audio (AUD) tasks except ImageBind-LLM
4. ASR-ChatGPT discards essential speaker/paralinguistic information through transcription
5. Instruction-tuning with 10-30 instruction variants per task; labels as generative outputs constrained by explicit options
6. **Community-extensible design** with clear contribution pipelines

---

### 6. Additional Benchmarks (Recently Emerged)

#### MTalk-Bench (arXiv: 2508.18240)
"Evaluating Speech-to-Speech Models in Multi-Turn Dialogues"
- First benchmark specifically targeting **multi-turn S2S dialogue evaluation**
- Arena-style pairwise + Rubrics-based absolute scoring
- Tests semantic, paralinguistic, and ambient sound across 9 realistic scenarios each
- **Key finding:** Models "typically regain coherence by increasing response length, sacrificing efficiency in multi-turn dialogues"
- LLM judges show position bias, length bias; need text annotations for non-verbal assessment

#### VoxRole (arXiv: 2509.03940)
"A Comprehensive Benchmark for Evaluating Speech-Based Role-Playing Agents"
- 13,335 multi-turn dialogues, 65.6 hours, 1,228 characters across 261 movies
- Tests **long-term persona consistency** in speech-based agents
- Two-stage pipeline: movie audio-script alignment + LLM-based character profile construction

#### VoiceAgentBench (arXiv: 2510.07978)
"Are Voice Assistants Ready for Agentic Tasks?"
- 6,000+ synthetic queries across multiple languages
- Tests multi-turn dialogue for voice assistant agentic capabilities

#### Towards Holistic Evaluation of LALMs (arXiv: 2505.15957, EMNLP 2025)
Four-dimensional evaluation taxonomy:
1. General Auditory Awareness and Processing
2. Knowledge and Reasoning
3. Dialogue-oriented Ability
4. Fairness, Safety, and Trustworthiness

---

## Part 2: Survey Papers

---

### Survey 1: "Recent Advances in Speech Language Models: A Survey"
**Authors:** Wenqian Cui, Dianzhi Yu, Xiaoqi Jiao, et al.
**ArXiv:** 2410.03751 | **Venue:** ACL 2025

#### Taxonomy Proposed

**Contrasts three paradigms:**

1. **Cascaded Pipeline (ASR + LLM + TTS):**
   - Suffers from "information loss during modality conversion, significant latency, and error accumulation across three stages"

2. **End-to-End SpeechLMs:**
   - Generate speech without converting from text
   - Components: Speech Tokenizer + Language Model + Token-to-Speech Synthesizer/Vocoder

3. **Native speech processing** (implied through direct speech modality models)

**Speech Tokenization Strategies:**
- **Semantic:** HuBERT, Wav2vec 2.0, WavLM -- capture meaning but "lack expressive information such as prosody"
- **Acoustic:** EnCodec, SoundStream -- preserve high-fidelity but "struggle with inaccuracies in content generation"
- **Mixed:** SpeechTokenizer, Mimi -- combine both through layered quantization

#### Key Training Data Discussion
- Pre-training: LibriSpeech (1k hours), LibriLight (60k hours), Gigaspeech (40k hours)
- Instruction-tuning uses synthetic TTS data: **SpeechInstruct**, **InstructS2S-200K**, **VoiceAssistant-400K**
- "Training a SpeechLM is significantly more challenging than a TextLM" due to speech requiring independent learning of linguistic rules

#### Open Problems Identified
1. Lack of clarity on optimal combinations of tokenizers, LMs, and vocoders
2. Unified end-to-end training remains unexplored (most train components separately)
3. **Real-time generation:** enabling sub-100ms latency for natural conversational flow
4. Safety: voice cloning, content generation risks, misuse potential
5. Rare language performance: extending beyond well-resourced languages
6. Human post-alignment (DPO, RLHF) remains underexplored for SpeechLMs

#### On Multiturn Stability
- Mentions real-time interaction paradigms (simultaneous understanding/generation with interruption capability)
- Models like Parrot and Moshi implement full-duplex capabilities
- Limited explicit discussion of multiturn degradation patterns

---

### Survey 2: "WavChat: A Survey of Spoken Dialogue Models"
**Authors:** Shengpeng Ji, Yifu Chen, et al. (19 co-authors)
**ArXiv:** 2411.13577 | 60 pages

#### Taxonomy Proposed

**Two primary paradigms:**

1. **Cascaded Spoken Dialogue Models:** Text as central intermediary
   - Examples: AudioGPT, ParalinGPT, E-chat, Qwen-Audio
   - Advantage: leverage strong LLM capabilities

2. **End-to-End Spoken Dialogue Models:** Direct speech representation processing
   - Examples: dGSLM (first fully E2E), SpeechGPT, Spirit-LM, Moshi, Mini-Omni, LLaMA-Omni
   - Advantage: eliminate sequential-module latency; access acoustic information beyond text

**Nine evaluation dimensions:**
1. Text intelligence
2. Speech intelligence
3. Audio generation
4. Music generation
5. Audio understanding
6. Multilingual capability
7. Context learning
8. Interaction capability
9. Streaming latency

#### On Full-Duplex and Turn-Taking
- Full-duplex is a "distinguishing feature" vs. text's half-duplex structure
- Requirements: real-time interruption (both directions), conversational fillers, simultaneous listening/speaking
- dGSLM pioneered this using self-attention + cross-attention for duplex interactions
- Moshi's RQ-Transformer addresses multi-quantizer latency
- Mini-Omni uses "delayed steps to enable parallel generation"
- Freeze-Omni employs "chunk-wise streaming speech encoder"

#### On Latency
- **First-token latency:** "the wait time for the user"
- **Generation latency:** "average latency of the generation process"
- Streaming requirements: "chunk-based mechanism, dynamically processing and generating audio in real time"
- Models must operate in causal manner -- understanding audio based solely on past information

#### On Multiturn Conversation
- Identifies "ability to handle long-form and multi-turn conversations" as key capability
- Must "revise previous responses based on new user instructions"
- Must support "historical context" and "extended audio outputs"
- **Limited discussion of specific degradation patterns or stability metrics**

#### On Training Data
- dGSLM: "thousands of hours of dual-track data"
- Spirit-LM: "word-level interleaving method with small, automatically curated speech-text parallel corpus"
- Notes that even E2E systems align speech with text "to leverage pre-trained LLMs" given speech data scarcity

#### Open Problems
- Unified speech representations (integrating semantic + acoustic)
- Single-layer quantizers (WavTokenizer, BigCodec) vs. multi-layer
- Direct speech-to-speech generation (increased complexity but "promising direction")
- Text guidance trade-offs (intermediate text generation vs. response speed)
- Complex reasoning remains inferior in spoken vs. text dialogue

---

### Survey 3: "A Survey on Speech Large Language Models for Understanding"
**Authors:** Jing Peng, Yucheng Wang, Bohan Li, et al. (Kai Yu group)
**ArXiv:** 2410.18908 | **Venue:** IEEE JSTSP (Special Issue)

#### Three-Stage Framework
1. **Modality Feature Extraction:** Whisper, Conformer, WavLM (continuous); HuBERT, EnCodec (discrete)
2. **Modality Information Fusion:**
   - Pre-Decoder Alignment (MLP projection, Q-Former)
   - In-Decoder Alignment (cross-attention)
   - Token Mapping & Space Expansion (CTC prompts, vocabulary augmentation)
3. **LLM Inference:** Processing fused information through transformer layers

#### Three-Stage Training Pipeline
- **Stage 1 - Modality Alignment:** ASR/audio captioning; encoder/LLM frozen; only bridges trained
- **Stage 2 - Multitask Training:** Weighted loss across 30+ tasks (ASR, translation, emotion recognition, speaker ID)
- **Stage 3 - Instruction & Preference Alignment:** SFT on instruction datasets + DPO/PPO

#### Critical Challenge: "LLM Dormancy"
**The LLM stops responding to unseen prompts, losing original intelligent capabilities:**
- FireRedASR (ASR-only): attention "almost entirely focuses on speech tokens"
- Qwen2-Audio-Instruct: shows preference for speech tokens over text
- Bayesian explanation: P(Y|X,I) approximates P(Y|X) when instruction I lacks diversity
- FireRedLLM "almost does not respond to prompts"
- SALMONN referenced as suffering "task-overfit"

#### Performance Collapse Examples
- Japanese translation of Mandarin speech: outputs phonetic characters instead of translation
- Speaker distance estimation: outputs "25 miles" for phone conversation
- Untrained emotional inference: irrelevant or empty responses

#### Proposed Solutions
1. LoRA coefficient reduction to enhance response diversity
2. Activation fine-tuning: few-shot on creative tasks to restore emergent capabilities
3. Discrete representation approach: convert continuous to discrete tokens before text-space mapping
4. Expanded token space: modify LLM input space to natively incorporate audio tokens

#### On Multiturn
- Qwen2-Audio-Chat SFT stage trained on "multi-turn dialogues involving both voice and text instructions"
- Most work emphasizes task-specific performance over dialogue capability
- Identified as underexplored

---

### Survey 4: "When Large Language Models Meet Speech: A Survey on Integration Approaches"
**Authors:** Zhengdong Yang, Shuichiro Shimizu, Yahan Yu, Chenhui Chu
**ArXiv:** 2502.19548 | **Venue:** ACL 2025 Findings

#### Three Integration Approaches

**1. Text-Based Integration**
- Pipeline: ASR -> LLM -> TTS (e.g., AudioGPT, HuggingGPT)
- Also: LLM Rescoring (n-best selection) and Generative Error Correction
- Strengths: easy implementation, minimal LLM modification, interpretable
- Weaknesses: information loss (prosody, emotion), error propagation, latency

**2. Latent-Representation-Based Integration**
- Convolutional downsampling, CTC compression, Q-Former
- Examples: SALMONN (Q-Former), Speech-Llama (CTC), Seed-ASR, Qwen-Audio, WavLLM
- Q-Former shown superior to convolutional methods
- **Cannot generate speech** (generating from continuous vectors remains unsolved)

**3. Audio-Token-Based Integration**
- Semantic tokens (k-means on self-supervised representations) + Acoustic tokens (neural codecs)
- Examples: SpeechGPT, VoxtLM, AudioPaLM, Moshi, Spirit-LM
- Strengths: deeper integration, speech generation capability
- Weaknesses: massive training data (millions of hours), limited multilingualism

#### Key Comparative Rankings
- **Integration depth:** Latent > Audio-token > Text-based
- **Interpretability:** Text-based > Audio-token > Latent
- **Speech generation:** Only text-based and audio-token approaches can generate speech
- **Best ASR (LibriSpeech test-other):** Seed-ASR (latent) at 2.8% WER

#### Use Case Recommendations
- **Latent-representation:** real-time processing, max accuracy, speech-input-only
- **Audio-token:** speech generation needed, multilingual synthesis, deep integration justified
- **Text-based:** limited resources, interpretability needed, rapid prototyping

---

### Survey 5: "From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models"
**Authors:** Yuxuan Chen, Haoyuan Yu
**ArXiv:** 2509.14515 (September 2025)

#### Taxonomy
- **Engineered Synchronization** (modular architectures)
- **Learned Synchronization** (end-to-end architectures)

#### Four Evaluation Dimensions
1. Temporal Dynamics
2. Behavioral Arbitration
3. Semantic Coherence
4. Acoustic Performance

#### Critical Challenges
1. **Synchronous data scarcity**
2. **Architectural divergence**
3. **Evaluation gaps**

---

## Part 3: Specific Topics

---

### Latency Optimization for Voice LLMs

#### Key Systems and Latency Measurements

| System | Latency | Approach |
|--------|---------|----------|
| **Moshi** | 160ms theoretical, 200ms practical | Parallel stream processing, RQ-Transformer, Inner Monologue, Mimi codec, full-duplex |
| **LLaMA-Omni** | 226ms total (206ms LLM + 30ms vocoder) | Whisper encoder + adaptor + LLaMA-3.1-8B + CTC speech decoder; non-autoregressive |
| **Ichigo** | 111ms to first token | Tokenized early-fusion, interleaved speech/text sequences |
| **Freeze-Omni** | Low-latency (unspecified) | Frozen LLM + chunk-wise streaming encoder; 3-stage training with only 60k QA examples on 8 GPUs |
| **GPT-4o** | ~320ms average | Proprietary end-to-end architecture |
| **SyncLLM** | Handles Internet latencies up to 240ms | Time-aligned Llama3-8b for full-duplex; 212k hours synthetic + 2k hours real dialogue |
| **Mini-Omni** | Real-time streaming | Delayed steps for parallel generation; VoiceAssistant-400K dataset |
| **Telecom pipeline** | RTF < 1.0 | 4-bit quantized LLM + streaming ASR + real-time TTS |

#### Key Latency Optimization Strategies
1. **Non-autoregressive speech decoders** (CTC-based, as in LLaMA-Omni)
2. **Parallel stream processing** (Moshi's dual-stream for user/system speech)
3. **Model quantization** (4-bit for deployment)
4. **Chunk-based streaming** (process audio in small chunks causally)
5. **Frozen LLM approaches** (Freeze-Omni: avoid catastrophic forgetting, reduce training cost)
6. **Inner Monologue** (Moshi: text token prediction preceding audio tokens improves linguistic quality)
7. **Delayed parallel generation** (Mini-Omni)

---

### Training Data for Voice LLMs

#### Key Instruction Datasets

**InstructS2S-200K** (LLaMA-Omni, arXiv: 2409.06666)
- 200,000 speech instruction + speech response pairs
- Construction: (1) Instruction rewriting with Llama-3-70B (filler words, spoken forms, shortening), (2) Response generation (concise, synthesis-compatible), (3) Speech synthesis (CosyVoice-300M for instructions, VITS for responses)
- Sources: ~50K from Alpaca, ~150K from UltraChat first-turn
- 418 hours instruction audio, 1,058 hours response audio

**VoiceAssistant-400K** (Mini-Omni)
- 400,000 examples for fine-tuning speech output generation
- Designed to preserve original language model capabilities while adding speech

**SpeechInstruct** (SpeechGPT, arXiv: 2305.11000)
- Large-scale cross-modal speech instruction dataset
- Used in three-stage training: modality-adaptation pre-training -> cross-modal instruction fine-tuning -> chain-of-modality instruction fine-tuning

**SyncLLM Training Data**
- 212k hours of synthetic spoken dialogue data (generated from text dialogue data via TTS)
- 2k hours of real-world spoken dialogue data
- Trained for full-duplex interaction

**General Pre-Training Corpora:**
- LibriSpeech: 1k hours
- LibriLight: 60k hours
- GigaSpeech: 40k hours
- dGSLM: "thousands of hours of dual-track data"

**Key Challenge:** "Training a SpeechLM is significantly more challenging than training a TextLM" -- speech data is vastly scarcer than text, and models must independently learn linguistic rules that text models get from concentrated semantic information.

**Trend:** Heavy reliance on TTS-synthesized data from text instruction datasets, raising questions about naturalness, diversity, and coverage of real spoken phenomena (disfluencies, overlapping speech, environmental noise).

---

### Multiturn Conversation Stability in Voice LLMs

#### Current State: Largely Underexplored

**The gap is explicitly acknowledged across multiple surveys and benchmarks:**

1. **SD-Eval** states: "currently supports the evaluation of single-turn dialogues only, limiting its application to more complex, multi-turn interactions"

2. **VoiceBench** has no multiturn evaluation component

3. **WavChat survey** identifies "ability to handle long-form and multi-turn conversations" as key capability but provides limited discussion of specific degradation patterns

4. **Survey on Speech LLMs for Understanding** identifies multiturn dialogue as "underexplored" in current research

#### Emerging Multi-Turn Evaluation

**MTalk-Bench** (arXiv: 2508.18240) is the first dedicated benchmark:
- Key finding: Models "typically regain coherence by increasing response length, sacrificing efficiency in multi-turn dialogues"
- Task-specific, modality-aware architectures outperform scaling-only approaches
- LLM judges demonstrate position bias and length bias

**VoxRole** (arXiv: 2509.03940):
- 13,335 multi-turn dialogues testing long-term persona consistency
- Reveals "crucial insights into strengths and limitations in maintaining persona consistency"

**SDiaReward** (arXiv: 2603.14889):
- Multi-turn reward model for spoken dialogue evaluation
- Uses episode-level preference pairs

#### Known Degradation Patterns
1. **LLM Dormancy** (from Survey on Speech LLMs): model stops responding to unseen prompts; attention distribution collapses onto speech tokens, ignoring instructions
2. **Task-overfit** (SALMONN): model loses generalization in complex multi-turn dialogues
3. **Coherence-efficiency tradeoff** (MTalk-Bench): models increase response length to maintain coherence, reducing conversational efficiency
4. **Paralinguistic degradation:** Models handle semantic content well in multi-turn but struggle with paralinguistic cues and environmental sounds across turns
5. **Context window limitations:** Extended audio outputs strain context handling in speech modality

#### What is Missing
- Standardized metrics for measuring multi-turn stability in speech
- Benchmarks testing degradation over 5, 10, 20+ turns
- Evaluation of how speech-specific features (prosody, emotion, speaker identity) maintain consistency across turns
- Testing of error accumulation in cascade systems over multiple turns
- Evaluation of how environmental noise handling degrades across turns

---

## Part 4: Key Models Reference

| Model | Type | Speech In | Speech Out | Latency | Key Feature |
|-------|------|-----------|------------|---------|-------------|
| **Moshi** | E2E Native | Yes | Yes | 200ms | Full-duplex, Inner Monologue, parallel streams |
| **LLaMA-Omni** | E2E | Yes | Yes | 226ms | Non-autoregressive CTC decoder, InstructS2S-200K |
| **Ichigo** | E2E Early-Fusion | Yes | Yes | 111ms | Interleaved speech-text tokens, 111ms first-token |
| **Freeze-Omni** | E2E (Frozen LLM) | Yes | Yes | Low | 3-stage training, frozen LLM, duplex capable |
| **Mini-Omni** | E2E | Yes | Yes | Real-time | First open-source E2E real-time; VoiceAssistant-400K |
| **SpeechGPT** | E2E | Yes | Yes | N/A | First discrete speech token LLM; SpeechInstruct |
| **GPT-4o** | Native | Yes | Yes | ~320ms | Proprietary; reference standard |
| **Qwen2-Audio** | Latent-repr | Yes | No | N/A | Multi-task, voice chat + analysis modes |
| **SALMONN** | Latent-repr | Yes | No | N/A | Dual auditory encoders; few-shot activation tuning |
| **DiVA** | E2E | Yes | No | N/A | Self-supervised from text-only LLM; 72% win rate vs Qwen2-Audio |
| **SyncLLM** | E2E Full-Duplex | Yes | Yes | <240ms | Time-aligned Llama3-8b; 212k hours synthetic data |
| **Spirit-LM** | Token-based | Yes | Yes | N/A | Word-level interleaving of speech-text |

---

## Part 5: Cross-Cutting Themes and Open Problems

### 1. Pipeline vs. End-to-End: No Clear Winner Yet
- VoiceBench shows pipeline (Whisper+LLaMA) **outperforms** end-to-end models by 10+ points on speech tasks
- But pipeline models lose paralinguistic information (SD-Eval, AudioBench)
- End-to-end models have lower latency potential but currently sacrifice quality
- The field is converging on **hybrid approaches** that use text as an auxiliary signal within end-to-end architectures

### 2. The Evaluation Gap
- Most benchmarks are single-turn only
- No standard metric for measuring conversation quality over extended multi-turn interactions
- Paralinguistic evaluation (emotion, prosody, speaker characteristics) remains immature
- Full-duplex/turn-taking evaluation is nascent
- Safety evaluation for speech-specific attacks (adversarial audio, voice-based jailbreaks) is minimal

### 3. Data Scarcity as the Root Bottleneck
- Speech instruction data is orders of magnitude scarcer than text
- Most training data is TTS-synthesized, missing natural spoken phenomena
- Dual-track (simultaneous speaker) conversation data is extremely rare
- No large-scale multi-turn spoken dialogue instruction dataset exists

### 4. The Instruction Sensitivity / LLM Dormancy Problem
- Training on speech tasks causes LLMs to lose general instruction-following ability
- Attention collapses onto speech tokens, ignoring text instructions
- This is a fundamental architecture challenge, not just a data problem
- Proposed mitigations: LoRA coefficient tuning, activation fine-tuning, discrete representations, expanded token space

### 5. Latency vs. Quality Tradeoff
- Sub-200ms latency achieved by Moshi and Ichigo, but with quality compromises
- LLaMA-Omni achieves 226ms (beating GPT-4o's 320ms) with competitive quality
- Frozen-LLM approaches (Freeze-Omni) show promise for maintaining LLM quality at low latency
- Real-time full-duplex remains the holy grail -- requires solving simultaneous understanding and generation

---

## References (ArXiv IDs)

### Benchmarks
- VoiceBench: 2410.17196
- AudioBench: 2406.16020
- SD-Eval: 2406.13340
- AIR-Bench: 2402.07729
- Dynamic-SUPERB: 2309.09510
- MTalk-Bench: 2508.18240
- VoxRole: 2509.03940
- VoiceAgentBench: 2510.07978
- Holistic LALM Eval: 2505.15957

### Surveys
- Recent Advances in Speech Language Models: 2410.03751 (ACL 2025)
- WavChat - Spoken Dialogue Models: 2411.13577
- Speech LLMs for Understanding: 2410.18908 (IEEE JSTSP)
- LLM-Speech Integration Approaches: 2502.19548 (ACL 2025 Findings)
- Full-Duplex Spoken Language Models: 2509.14515

### Key Models
- Moshi: 2410.00037
- LLaMA-Omni: 2409.06666 (ICLR 2025)
- Ichigo: 2410.15316
- Freeze-Omni: 2411.00774
- Mini-Omni: 2408.16725
- SpeechGPT: 2305.11000
- SALMONN: 2310.13289
- DiVA: 2410.02678
- SyncLLM: 2409.15594 (EMNLP 2024)
- SpeechVerse: 2405.08295
