<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,10,24,30&height=220&section=header&text=Vaani%20Noise%20Event%20Detection&fontSize=42&fontColor=ffffff&animation=twinkling&fontAlignY=38&desc=Datathon@IndoML%202026%20|%20Track%201%20Competition%20System&descSize=16&descAlignY=58&descAlign=50" width="100%" alt="Vaani Noise Event Detection Banner" />
</p>

<p align="center">
  <a href="https://indoml.in/datathon"><img src="https://img.shields.io/badge/Competition-Datathon%40IndoML%202026-blueviolet?style=for-the-badge" alt="IndoML 2026" /></a>
  <a href="https://www.codabench.org/competitions/17825/"><img src="https://img.shields.io/badge/Codabench-Track%201%3A%20Noise%20Event%20Detection-blue?style=for-the-badge" alt="Codabench Track 1" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-ee4c2c?style=for-the-badge&logo=pytorch" alt="PyTorch" /></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface" alt="Transformers" /></a>
  <a href="#"><img src="https://img.shields.io/badge/Best%20Combined%20Score-0.8418-success?style=for-the-badge" alt="Combined Score 0.8418" /></a>
</p>

---

## 📌 Project Overview & Portfolio Note

This repository contains the end-to-end machine learning system developed for **Datathon@IndoML 2026 — Track 1: Noise Event Detection**. 

The goal is to detect real-world acoustic noise events (vehicle horns, traffic noise, dogs barking, doorbells, kitchen appliances, background chatter, static) embedded within Indic speech recordings, outputting precise onset and offset timestamps using the **[Vaani corpus](https://arxiv.org/abs/2603.28714)** (a large-scale, multi-lingual Indian speech dataset).

> [!NOTE]
> **Portfolio Attribution**: This repository is a personal portfolio copy maintained by **[Shubham Warkade](https://github.com/Lalla-dev)**, copied with full permission from Team Lead **Kanak Baghel** from the original team repository: [`Kanakbaghel/Vaani-Noise_Event_Detection`](https://github.com/Kanakbaghel/Vaani-Noise_Event_Detection).

---

## 👥 Team Deadlock & Contributions

| Contributor | Primary Contributions |
| :--- | :--- |
| **Kanak Baghel** *(Team Lead)* | Data pipeline architecture, annotation-tier unification (Gold/Silver/Bronze), exploratory data analysis (EDA), team coordination, submission tracking |
| **Shubham Warkade** | CRNN baseline model, Log-Mel feature extraction, PyTorch GPU training & inference pipelines, submission generation, corrected Wav2Vec2 + BiGRU architecture |
| **Ankit Kushwaha** | Advanced modeling research, pretrained audio encoder exploration, boundary prediction |
| **Rithish K** | Advanced modeling research, temporal collar analysis, onset/offset boundary prediction |
| **Soubhik Shit** | Local evaluation framework, metric calibration (Event F1 + Segment Dice), synthetic vs natural audio analysis |

---

## 🏆 Key Results & Benchmarks

The competition evaluates models on a **Combined Score** across a withheld 2,501-clip evaluation dataset:
$$\text{Combined Score} = \text{Event-Based } F_1 \ (\pm 20\% \text{ tolerance collar}) + \text{Segment-Level Dice} \quad (\text{Max: } 2.0)$$

### Model Performance Summary

| Architecture | Input Representation | Frame Grid | Training Subset | Event $F_1$ | Segment Dice | **Combined Score** | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **CRNN Baseline (Ours)** | **Log-Mel Spectrogram** | **10 ms** (100 fps) | 9,426 Gold | **0.2700** | **0.5717** | **`0.8418`** | **Official Submission (ID: 938825)** |
| 🥈 **Wav2Vec2 + BiGRU (V2 Corrected)** | Raw Audio (FP16) | 20 ms (50 fps) | 5,000 Gold | 0.1811 | 0.5597 | **`0.7408`** | Validated (Threshold: 0.55) |
| ⚠️ *Wav2Vec2 + BiGRU (V1 Unweighted)* | Raw Audio | 20 ms (50 fps) | 2,500 Gold | 0.0008 | 0.5934 | *0.5942* | Logit collapse baseline |

---

## 🔬 System Architectures

### 1. Champion Model: Log-Mel CRNN (`src/model.py`)
- **Front-End**: 64-band Log-Mel filterbanks extracted at 16 kHz ($N_{\text{fft}}=400$, $\text{hop}=160 \rightarrow 10\text{ms}$ resolution).
- **Data Augmentation**: Time/frequency SpecAugment masking applied during training.
- **Backbone**: 3-layer Convolutional block with BatchNorm, ReLU, and MaxPooling to compress spectral features.
- **Sequence Modeling**: 2-layer Bidirectional GRU (128 hidden units per direction) to capture temporal context.
- **Classification Head**: Frame-level Linear projection with Sigmoid activation.
- **Parameters**: **1.16M** parameters (fast, robust, lightweight).

### 2. Advanced Model: Fine-Tuned Wav2Vec2 + BiGRU (`src/advanced_model.py`)
- **Front-End**: Raw 16 kHz waveform passed directly into `facebook/wav2vec2-base` (7-layer CNN feature extractor downsamples by $320\times \rightarrow 20\text{ms}$ resolution).
- **Head**: Dropout ($p=0.10$), 1-layer Bidirectional GRU (128 hidden units), and frame-level Linear classifier (**689,921** trainable parameters).
- **Addressing Class Imbalance**: Incorporates `BCEWithLogitsLoss` with dynamically computed `pos_weight = 2.6351` (derived from 1.77M valid training frames).
- **Zero-Padding Masking**: Padded frames are masked out of the loss computation to prevent silent audio padding from corrupting gradients.

```
                    ┌─────────────────────────┐
                    │  Raw 16kHz Audio Wave   │
                    └────────────┬────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
┌──────────────────────┐                   ┌──────────────────────┐
│ Log-Mel Filterbanks  │                   │ Wav2Vec2 Base (CNN)  │
│ (64 bands, 10ms hop) │                   │ (320 stride, 20ms)   │
└──────────┬───────────┘                   └──────────┬───────────┘
           ▼                                           ▼
┌──────────────────────┐                   ┌──────────────────────┐
│  3x Conv2D + Pool    │                   │   Frozen Encoder     │
└──────────┬───────────┘                   └──────────┬───────────┘
           ▼                                           ▼
┌──────────────────────┐                   ┌──────────────────────┐
│   2-layer BiGRU      │                   │   1-layer BiGRU      │
└──────────┬───────────┘                   └──────────┬───────────┘
           ▼                                           ▼
┌──────────────────────┐                   ┌──────────────────────┐
│ Frame Classification │                   │ Frame Classification │
│  (100 frames/sec)    │                   │  (50 frames/sec)     │
└──────────┬───────────┘                   └──────────┬───────────┘
           ▼                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│     Post-Processing: Threshold + Median Filter + Gap Merge      │
└─────────────────────────────────┬───────────────────────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │   JSONL Event Predictions   │
                   └─────────────────────────────┘
```

---

## 💡 Engineering Insights: Why CRNN Outperformed Wav2Vec2

1. **Acoustic Energy vs. Linguistic Invariance**:
   - `wav2vec2-base` is pretrained via self-supervised contrastive learning to extract speech phonetics and linguistic tokens; its representations actively filter out background noise as nuisance variation.
   - For Noise Event Detection, the non-speech environmental noise (fans, engine hum, clicks, animal cries) is the primary target signal. The Log-Mel spectrogram directly retains all high- and low-frequency spectral energy.
2. **Temporal Collar Resolution ($10\text{ms}$ vs $20\text{ms}$)**:
   - The competition metric uses a collar of $\pm 20\%$ of event duration ($\ge 50\text{ms}$).
   - The CRNN's $10\text{ms}$ frame step allows significantly sharper onset/offset localization than Wav2Vec2's $20\text{ms}$ step, resulting in a **$49\%$ higher Event $F_1$ score** (`0.2700` vs `0.1811`).

---

## 📁 Repository Structure

```text
Vaani-Noise_Event_Detection/
├── configs/
│   └── baseline.yaml                 # CRNN hyperparameters and audio config
├── data/
│   ├── processed/                    # Preprocessed metadata splits
│   │   ├── train_split.jsonl         # Stratified training split
│   │   ├── val_split.jsonl           # Local validation split
│   │   └── unified.jsonl             # Unified Gold/Silver/Bronze annotations
│   ├── raw/                          # Raw HF Parquet shards (gitignored)
│   └── cache/audio_clips/            # Locally cached WAV files (gitignored)
├── docs/                             # Technical documentation & guides
│   ├── audio_loading_guide.md        # Lazy loading system guide
│   └── AUDIO_LOADING_QUICKSTART.md   # Quick audio loader reference
├── reports/
│   ├── eda_category_dist.png         # Noise category distributions
│   ├── eda_duration_hist.png         # Audio duration histograms
│   └── system_description.md         # Full technical system report
├── scripts/
│   ├── smoke_gpu_advanced.py         # RTX 4050 FP16 GPU smoke test
│   ├── cache_gold_subset.py          # Pre-cache local audio WAV files
│   ├── sweep_decoding.py             # Baseline post-processing grid sweep
│   └── sweep_fine_postprocess.py     # Fine-grained post-processing sweep
├── src/
│   ├── model.py                      # CRNN baseline architecture
│   ├── advanced_model.py             # Wav2Vec2 + BiGRU architecture
│   ├── features.py                   # Log-Mel spectrogram extraction & SpecAugment
│   ├── train.py                      # CRNN baseline training pipeline
│   ├── train_advanced.py             # Corrected Wav2Vec2 training with frame masking
│   ├── infer.py                      # CRNN inference & Codabench packaging
│   ├── infer_advanced.py             # Wav2Vec2 inference & validation sweep
│   ├── val_loader.py                 # Codabench 2,501-clip validation loader
│   ├── eval_local.py                 # Local Event F1 & Segment Dice metric engine
│   ├── dataset.py                    # PyTorch Dataset wrapper & collate functions
│   └── audio_loader.py               # Robust audio reader with disk cache
├── tests/                            # Automated test suite
│   ├── verify_setup.py               # Setup & environment verification
│   └── test_advanced_smoke.py        # Unit tests for Wav2Vec2 target framing
├── .gitignore                        # Git exclusion rules (weights, audio, secrets)
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and create a Python virtual environment:
```bash
git clone https://github.com/Lalla-dev/Vaani-Noise_Event_Detection.git
cd Vaani-Noise_Event_Detection

python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# or: .venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

### 2. GPU Smoke Test (RTX 4050 / T4)
Verify CUDA availability, FP16 mixed precision, and forward/backward memory limits:
```bash
python scripts/smoke_gpu_advanced.py
```

### 3. Training the CRNN Baseline
Train the 1.16M parameter CRNN baseline model on Gold data:
```bash
python src/train.py \
    --train-jsonl data/processed/train_split.jsonl \
    --val-jsonl data/processed/val_split.jsonl \
    --batch-size 32 \
    --epochs 10 \
    --lr 3e-4 \
    --save-path models/baseline_crnn.pt
```

### 4. Training the Advanced Wav2Vec2 Model
Train the Wav2Vec2 + BiGRU model with automated class-imbalance weighting and frame masking:
```bash
python src/train_advanced.py \
    --tier-filter gold \
    --max-samples 5000 \
    --epochs 5 \
    --batch-size 4 \
    --freeze-encoder \
    --lr-head 5e-4 \
    --pos-weight auto \
    --save-path models/advanced_wav2vec2_5000_gold_v2.pt
```

### 5. Running Validation & Parameter Sweeping
Evaluate predictions across all 2,501 validation clips and sweep post-processing thresholds:
```bash
python src/infer_advanced.py \
    --model models/advanced_wav2vec2_5000_gold_v2.pt \
    --validation-dir ../Validation/validation \
    --sweep
```

### 6. Generating Codabench Submission Package
Produce a valid `submission.zip` with root-level `predictions.jsonl`:
```bash
python src/infer.py \
    --model models/baseline_crnn_9426_gold.pt \
    --input-data input_data \
    --threshold 0.35 \
    --median-filter 7 \
    --zip-out submissions/submission_track1.zip
```

---

## 🔒 Security & Data Compliance
- **No Private Data Tracked**: Raw audio files, cached `.wav` clips, competition validation data (`Validation/`), and private tokens (`.env`) are strictly excluded via `.gitignore`.
- **Lightweight Portfolio Repository**: Binary checkpoints (`*.pt`) are kept locally and omitted from version control.

---

<p align="center">
  <sub>Developed for Datathlon@IndoML 2026. Portfolio copy maintained by <a href="https://github.com/Lalla-dev">Shubham Warkade</a>.</sub>
</p>
