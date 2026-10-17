# Track 2 Phase 2 Checkpoint & Resume Instructions

**Date & Time**: 2026-10-17 00:40 IST  
**Status**: Safely Paused for Workstation Shutdown  

---

## 1. Executive Summary

| Item | Status | Details |
| :--- | :---: | :--- |
| **Track 2 Input Dataset** | Verified | `input_data_track-2_phase-2\audio` (3,178 WAVs, 10.91 audio hours) |
| **Enhanced WAVs Completed** | **1,664 / 3,178** | **52.36% complete**, all 1,664 files verified as valid 16kHz mono `PCM_16` |
| **Pending Enhancement** | 1,514 clips | Exactly 47.64% remaining (~38–40 minutes on GPU) |
| **Enhancement Engine** | Verified | SpeechBrain SepFormer (`speechbrain/sepformer-dns4-16k-enhancement`) |
| **ASR Mandated Model** | Verified & Cached | `ARTPARK-IISc/SraVaani-1.0` (866.7 MB weights cached locally, tested on GPU) |
| **Track 1 Champion** | Preserved | Both CRNN (`0.8418`) and Ensemble (`0.9588`) models & submissions intact |
| **Running Jobs** | **0 (Stopped)** | All background tasks safely halted; **SAFE TO SHUT DOWN** |

---

## 2. Model & Checkpoint Storage on Disk

No files need to be re-downloaded when the workstation restarts:
- **SpeechBrain SepFormer Cache**: `~/.cache/huggingface/hub/models--speechbrain--sepformer-dns4-16k-enhancement` (215.3 MB)
- **SraVaani 1.0 ASR Cache**: `~/.cache/huggingface/hub/models--ARTPARK-IISc--SraVaani-1.0` (1,734.4 MB, includes `model-asr.fp16.ts` 866.7 MB)
- **Enhanced WAV Directory**: `data/enhanced_track2_phase2/` (1,664 valid files on disk)

---

## 3. Resume Workflow (Step-by-Step)

When you turn on your laptop and resume work:

### Step 1: Open Terminal & Activate Environment
```powershell
cd "C:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection"
& "C:\Users\Shubham\Desktop\vaani project\.venv\Scripts\Activate.ps1"
```

### Step 2: Resume Enhancement (Completes Remaining 1,514 Clips)
The script will automatically detect the 1,664 completed files and immediately begin processing from clip 1,665:
```powershell
python scripts/run_track2_phase2.py --input-dir "C:\Users\Shubham\Desktop\vaani project\input_data_track-2_phase-2\audio" --output-dir "data/enhanced_track2_phase2" --log-interval 50
```
*Estimated runtime: ~38–40 minutes on NVIDIA RTX 4050 Laptop GPU.*

### Step 3: Run Mandated SraVaani ASR Transcription
Once enhancement reaches 3,178 / 3,178 clips, run the transcription script to generate `data/transcripts.jsonl`:
```powershell
python scripts/run_track2_transcribe.py --audio-dir "data/enhanced_track2_phase2" --output-file "data/transcripts.jsonl" --batch-size 8 --log-interval 50
```
*Estimated runtime: ~35–45 minutes.*

### Step 4: Validate and Package Submission ZIP
Run the automated packaging and audit script to verify all 3,178 WAVs and transcripts, creating `submissions/submission_track2_phase2.zip`:
```powershell
python -c "from pathlib import Path; from track2.package_submission import validate_and_package_submission; validate_and_package_submission(input_audio_dir=Path(r'C:\Users\Shubham\Desktop\vaani project\input_data_track-2_phase-2\audio'), enhanced_audio_dir=Path('data/enhanced_track2_phase2'), transcripts_file=Path('data/transcripts.jsonl'), output_zip=Path('submissions/submission_track2_phase2.zip'))"
```

---

## 4. Track 1 Champion Status

Track 1 assets are protected and completely untouched:
- Standalone CRNN (Gold 9,426 clips): `models/baseline_crnn_9426_gold.pt` | Score: `0.8418`
- CRNN + Wav2Vec2 Ensemble: `models/advanced_wav2vec2_9426_gold.pt` | Score: `0.9588`
- Candidate ZIP: `submissions/submission_track1_ensemble_candidate.zip`
- Git branch: `track1-ensemble` (commit `36c64db` pushed to remote `personal`)
