# Latice.AI – Finnish Automotive STT Benchmark

This repository documents the end-to-end benchmark we ran on a **Finnish automotive** speech corpus to compare Latice.AI’s specialized STT engine against leading general-purpose providers.

---

## 🎯 Objective
- Highlight Latice.AI’s advantage on real Finnish automotive data.
- Stress-test robustness across audio utterances.

---

## 🔬 Methodology
- **Dataset**: audio `.wav` files. Ground truth and model outputs align across `Audio/`, `dataset.json`, `result.csv`, and `wer_results.txt`.
- **Models Evaluated (9)**: Deepgram Nova 3, ElevenLabs, Fal, Gladia, Google latest_long, Groq Whisper Large V3, Groq Whisper Large V3 Turbo, **Latice.AI**, OpenAI GPT-4o Transcribe.
- **Metrics**: Word Error Rate (WER), distribution of perfect vs failed transcripts, average latency, qualitative inspection.
- **Scripts**: `launch_test.py` for bulk inference, `wer.py` for normalized WER computation.

---

## 📊 Results

### Average WER (lower is better)
| STT Service | Average WER | Delta vs Latice |
|-------------|-------------|-----------------|
| **Latice.AI** | **0.198** | **Baseline** |
| Groq Whisper Large V3 | 0.271 | -37.26% |
| Gladia | 0.278 | -40.68% |
| Fal | 0.290 | -46.57% |
| Groq Whisper Large V3 Turbo | 0.359 | -81.92% |
| Google latest_long | 0.513 | -159.76% |
| ElevenLabs | 0.553 | -180.16% |
| OpenAI GPT-4o Transcribe | 0.614 | -210.81% |
| Deepgram Nova 3 | 0.639 | -223.51% |

### Perfect vs Failed Transcriptions (WER = 0 / WER = 1)
| STT Service | Perfect | Failed |
|-------------|---------|--------|
| **Latice.AI** | **118** | **16** |
| Groq Whisper Large V3 | 115 | 37 |
| Gladia | 112 | 38 |
| Fal | 110 | 37 |
| Groq Whisper Large V3 Turbo | 89 | 50 |
| Google latest_long | 54 | 71 |
| ElevenLabs | 53 | 90 |
| OpenAI GPT-4o Transcribe | 49 | 87 |
| Deepgram Nova 3 | 37 | 111 |

### Average Latency
| STT Service | Avg latency |
|-------------|-------------|
| Groq Whisper Large V3 Turbo | 0.389 s |
| Groq Whisper Large V3 | 0.389 s (+0.18%) |
| **Latice.AI** | 0.511 s (+31.61%) |
| ElevenLabs | 0.823 s (+111.84%) |
| OpenAI GPT-4o Transcribe | 0.981 s (+152.47%) |
| Google latest_long | 1.249 s (+221.45%) |
| Fal | 1.276 s (+228.36%) |
| Deepgram Nova 3 | 2.129 s (+447.95%) |
| Gladia | 6.836 s (+1659.57%) |

**Global average latency**: 1.620 s

---

## 🏆 Key Takeaways
- **Accuracy**: Latice.AI delivers a 0.198 WER, leading every general-purpose alternative.
- **Robustness**: 118 perfect transcripts against only 16 complete failures, even with noisy/augmented inputs.
- **Speed**: Latice.AI sustains 0.511 s average latency, competitive with real-time stacks.
- **Industrialized pipeline**: ready-to-ship scripts, CSV exports, and reporting assets.

---

## 📋 Repository Structure
```
├── Audio/                       # audio Finnish automotive clips
├── dataset.json                 # Aligned ground-truth transcripts
├── result.csv                   # Raw model outputs
├── wer_results.txt              # Text export of per-utterance WER
├── launch_test.py               # Batch evaluation pipeline
├── wer.py                       # WER normalization workflow
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started
1. Run the audio-audio transcription batch: python launch_test.py
2. Compute and export normalized WER: python wer.py

---

## 📈 Business Impact
- **Less QA overhead**: higher accuracy reduces manual validation cycles.
- **Operational confidence**: consistent handling of plates, VINs, dealership jargon.
- **Automation**: reproducible benchmark pipeline for future RFPs and customer deliverables.

---

## 📊 Conclusion
Latice.AI tops the Finnish automotive benchmark with the strongest accuracy/latency balance. The engine is tuned for dealership and support workflows where every word matters.

> **Latice.AI delivers best-in-class accuracy, robustness, and workflow for Finnish automotive speech-to-text.**
