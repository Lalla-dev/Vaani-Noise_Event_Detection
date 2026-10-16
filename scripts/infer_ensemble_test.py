"""
infer_ensemble_test.py
Generates official Track 1 test predictions using the winning CRNN + Wav2Vec2 ensemble configuration.

Winning Configuration:
- CRNN: models/baseline_crnn_9426_gold.pt (weight = 0.60)
- Wav2Vec2: models/advanced_wav2vec2_9426_gold.pt (weight = 0.40)
- Temporal Grid Alignment: 10 ms grid (resampling frame-centers, clipped to valid duration)
- Probability Threshold: 0.40
- Median Filter Window: 5
- Minimum Event Duration: 0.15s
- Merge Gap: 0.08s

Safeguards:
- Preserves all existing predictions and submission ZIPs.
- Writes to submissions/predictions_ensemble_candidate.jsonl.
- Packages submissions/submission_track1_ensemble_candidate.zip (containing predictions.jsonl at root).
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import median_filter
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows console UTF-8 safety
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from src.features import LogMelExtractor
from src.model import CRNNNoiseDetector
from src.advanced_model import Wav2Vec2NoiseDetector, WAV2VEC2_STRIDE, WAV2VEC2_SR
from src.infer import AudioFolderDataset, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Test set inference for Track 1 winning ensemble")
    parser.add_argument("--crnn-model", type=str, default="models/baseline_crnn_9426_gold.pt")
    parser.add_argument("--w2v-model", type=str, default="models/advanced_wav2vec2_9426_gold.pt")
    parser.add_argument("--input-data", type=str, default=None)
    parser.add_argument("--out-jsonl", type=str, default="submissions/predictions_ensemble_candidate.jsonl")
    parser.add_argument("--out-zip", type=str, default="submissions/submission_track1_ensemble_candidate.zip")
    parser.add_argument("--alpha-w2v", type=float, default=0.40)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--median-filter", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=0.15)
    parser.add_argument("--merge-gap", type=float, default=0.08)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", default=True)
    return parser.parse_args()


def probs_to_events_10ms(
    probs: np.ndarray,
    threshold: float,
    duration: float,
    med: int = 5,
    merge_gap: float = 0.08,
    min_duration: float = 0.15,
) -> List[Dict]:
    frame_step = 0.010
    mask = (probs >= threshold).astype(np.uint8)
    if med > 1:
        mask = median_filter(mask, size=med, mode="nearest")

    if mask.sum() == 0:
        return []

    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    events = []
    for s, e in zip(starts, ends):
        onset = round(s * frame_step, 3)
        offset = round(min(e * frame_step, duration), 3)
        if offset > onset:
            events.append({"onset": onset, "offset": offset})

    # Merge gap
    if merge_gap > 0.0 and len(events) > 1:
        merged = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            if ev["onset"] - prev["offset"] <= merge_gap:
                merged[-1] = {"onset": prev["onset"], "offset": round(max(prev["offset"], ev["offset"]), 3)}
            else:
                merged.append(ev)
        events = merged

    # Min duration
    if min_duration > 0.0:
        events = [ev for ev in events if round(ev["offset"] - ev["onset"], 3) >= min_duration]

    return events


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Locate input_data
    input_path = None
    if args.input_data:
        input_path = Path(args.input_data)
    else:
        candidates = [
            PROJECT_ROOT.parent / "input_data",
            PROJECT_ROOT / "input_data",
            PROJECT_ROOT.parent / "input_data.zip",
        ]
        for c in candidates:
            if c.exists():
                input_path = c
                break

    if not input_path:
        raise FileNotFoundError("Track 1 input_data directory not found.")

    print("=" * 80)
    print("VAANI TRACK 1: FINAL TEST SET INFERENCE (WINNING ENSEMBLE)")
    print("=" * 80)
    print(f"CRNN Model        : {args.crnn_model}")
    print(f"Wav2Vec2 Model    : {args.w2v_model}")
    print(f"Input Data Source : {input_path}")
    print(f"Device            : {device} (FP16: {args.fp16 and device.type == 'cuda'})")
    print("-" * 80)
    print("Ensemble Configuration:")
    print(f"  Grid Alignment        : 10 ms uniform grid (centered, clamped to clip duration)")
    print(f"  Wav2Vec2 Weight (alpha): {args.alpha_w2v:.2f}")
    print(f"  CRNN Weight           : {1.0 - args.alpha_w2v:.2f}")
    print(f"  Probability Threshold : {args.threshold:.2f}")
    print(f"  Median Filter Size    : {args.median_filter}")
    print(f"  Min Event Duration    : {args.min_duration:.2f}s")
    print(f"  Merge Gap             : {args.merge_gap:.2f}s")
    print("=" * 80)

    # 1. Load models
    print("\n[1/4] Loading models into memory...")
    crnn = CRNNNoiseDetector(n_mels=64)
    crnn = load_checkpoint(crnn, args.crnn_model, device)
    crnn.eval()
    print("  CRNN loaded successfully.")

    w2v = Wav2Vec2NoiseDetector()
    w2v = load_checkpoint(w2v, args.w2v_model, device)
    w2v.eval()
    print("  Wav2Vec2 loaded successfully.")

    extractor = LogMelExtractor(sr=16000, n_mels=64, n_fft=1024, hop_length=512)

    # 2. Load dataset
    print(f"\n[2/4] Initializing test dataset from {input_path}...")
    dataset = AudioFolderDataset(input_path, target_sr=16000)
    total_clips = len(dataset)
    print(f"  Found {total_clips} audio clips in test set.")

    # 3. Inference loop
    print(f"\n[3/4] Running dual-model inference on {total_clips} clips...")
    t_start = time.time()
    predictions = []
    clips_with_events = 0
    total_events = 0

    for idx in range(total_clips):
        item = dataset[idx]
        clip_id = item["clip_id"]
        audio_tensor = item["audio"]
        audio_np = audio_tensor.numpy()
        sr = item["sample_rate"]
        duration = float(item["duration"])

        # Forward CRNN
        try:
            feats = extractor.extract(audio_np)
            feats = feats.unsqueeze(0).to(device)
            with torch.no_grad():
                crnn_logits = crnn(feats).squeeze(0)
                p_crnn = torch.sigmoid(crnn_logits).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[Warning] CRNN forward failed for {clip_id}: {e}")
            p_crnn = np.zeros(1, dtype=np.float32)

        # Forward Wav2Vec2
        try:
            waveform = audio_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(args.fp16 and device.type == "cuda")):
                    w2v_logits = w2v(waveform).squeeze(0)
                p_w2v = torch.sigmoid(w2v_logits).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[Warning] Wav2Vec2 forward failed for {clip_id}: {e}")
            p_w2v = np.zeros(1, dtype=np.float32)

        # Grid alignment to 10ms
        T_crnn = len(p_crnn)
        T_w2v = len(p_w2v)
        t_crnn = (np.arange(T_crnn) + 0.5) * (512 / 16000)
        t_w2v = (np.arange(T_w2v) + 0.5) * (WAV2VEC2_STRIDE / WAV2VEC2_SR)

        M = max(1, int(round(duration / 0.010)))
        t_target = (np.arange(M) + 0.5) * 0.010

        interp_crnn = np.interp(t_target, t_crnn, p_crnn)
        interp_w2v = np.interp(t_target, t_w2v, p_w2v)

        p_blend = args.alpha_w2v * interp_w2v + (1.0 - args.alpha_w2v) * interp_crnn

        # Event decoding
        evs = probs_to_events_10ms(
            p_blend,
            threshold=args.threshold,
            duration=duration,
            med=args.median_filter,
            merge_gap=args.merge_gap,
            min_duration=args.min_duration,
        )

        if evs:
            clips_with_events += 1
            total_events += len(evs)

        predictions.append({
            "clip_id": clip_id,
            "events": evs,
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == total_clips:
            elapsed = time.time() - t_start
            rate = (idx + 1) / max(elapsed, 1e-3)
            print(f"  Processed [{idx + 1:5d}/{total_clips}] clips ({rate:.1f} clips/s) | Events: {total_events}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nInference completed in {elapsed_total:.1f}s ({total_clips / max(elapsed_total, 1e-3):.1f} clips/s)")
    print(f"Clips with detected events: {clips_with_events} / {total_clips} ({clips_with_events / total_clips * 100:.1f}%)")
    print(f"Total events detected     : {total_events}")

    # 4. Save predictions JSONL
    out_jsonl = Path(args.out_jsonl)
    if not out_jsonl.is_absolute():
        out_jsonl = PROJECT_ROOT / out_jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[4/4] Writing predictions JSONL to: {out_jsonl}")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"  Successfully wrote {len(predictions)} records.")

    # 5. Package candidate submission zip
    out_zip = Path(args.out_zip)
    if not out_zip.is_absolute():
        out_zip = PROJECT_ROOT / out_zip
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nPackaging Codabench candidate submission archive to: {out_zip}")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out_jsonl, arcname="predictions.jsonl")

    # Verify zip integrity
    with zipfile.ZipFile(out_zip, "r") as z:
        namelist = z.namelist()
        info = z.getinfo("predictions.jsonl")
        print(f"  Archive contains: {namelist}")
        print(f"  Uncompressed size: {info.file_size:,} bytes")
        print(f"  Compressed size  : {info.compress_size:,} bytes")
        assert namelist == ["predictions.jsonl"], f"Invalid zip contents: {namelist}"

    print("\n" + "=" * 80)
    print("TEST INFERENCE & PACKAGING COMPLETE")
    print(f"Candidate JSONL: {out_jsonl}")
    print(f"Candidate ZIP  : {out_zip}")
    print("=" * 80)


if __name__ == "__main__":
    main()
