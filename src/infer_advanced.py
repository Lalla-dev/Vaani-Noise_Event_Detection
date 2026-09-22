"""
infer_advanced.py
Inference, validation evaluation, parameter sweeping, and Codabench submission generator
for the advanced Wav2Vec2 + BiGRU Noise Event Detector.

Key features:
- Raw 16kHz audio waveforms directly into Wav2Vec2.
- Fast FP16 GPU inference on RTX 4050.
- Ground-truth evaluation against Codabench validationMetadata.json.
- Threshold & post-processing parameter sweep mode (--sweep) without recomputing forward passes.
- Direct comparison with the 0.8418 Combined baseline.
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

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

from src.advanced_model import Wav2Vec2NoiseDetector, WAV2VEC2_STRIDE, WAV2VEC2_SR
from src.infer import AudioFolderDataset
from src.eval_local import (
    load_ground_truth,
    event_counts,
    merge_intervals,
    total_duration,
    intersection_duration,
)


def parse_args():
    default_model = "models/advanced_wav2vec2_5000_gold_v2.pt"
    if not (PROJECT_ROOT / default_model).exists():
        default_model = "models/advanced_wav2vec2_gold.pt"

    parser = argparse.ArgumentParser(description="Inference and Evaluation for Wav2Vec2 Noise Event Detection")
    parser.add_argument("--model", type=str, default=default_model,
                        help=f"Path to trained model checkpoint (default: {default_model})")
    parser.add_argument("--input-data", type=str, default=None,
                        help="Path to Codabench input_data folder or input_data.zip containing audio files")
    parser.add_argument("--validation-dir", type=str, default=None,
                        help="Path to extracted Codabench validation folder (reads validationMetadata.json + local WAVs)")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, default=False,
                        help="Evaluate against ground-truth validation metadata and report F1/Dice/Combined")
    parser.add_argument("--sweep", action=argparse.BooleanOptionalAction, default=False,
                        help="Run multi-threshold and post-processing sweep on validation set")
    parser.add_argument("--out", type=str, default="submissions/predictions_wav2vec2_5000_gold.jsonl",
                        help="Output predictions JSONL path")
    parser.add_argument("--zip-out", type=str, default="submissions/submission_track1_wav2vec2_5000_gold.zip",
                        help="Explicit destination path for Codabench submission zip package")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Probability threshold for event detection (default: 0.35)")
    parser.add_argument("--median-filter", type=int, default=7,
                        help="Window size for median filtering binary prediction mask (default: 7)")
    parser.add_argument("--min-duration", type=float, default=0.15,
                        help="Minimum event duration in seconds (default: 0.15)")
    parser.add_argument("--merge-gap", type=float, default=0.10,
                        help="Maximum gap in seconds between consecutive events to merge (default: 0.10)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device (auto-detects cuda if available)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="Use FP16 mixed precision for fast GPU inference")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Process at most N samples (for quick testing)")
    parser.add_argument("--create-zip", action=argparse.BooleanOptionalAction, default=False,
                        help="Create Codabench submission zip archive (default: False)")
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    p = Path(ckpt_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found at {p}")
    ckpt = torch.load(str(p), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def frames_to_seconds(start_frame: int, end_frame: int, stride: int, sr: int, duration: float) -> Tuple[float, float]:
    onset = round(start_frame * stride / sr, 3)
    offset = round(min(end_frame * stride / sr, duration), 3)
    return onset, offset


def logits_to_events(
    probs: Union[torch.Tensor, np.ndarray],
    threshold: float,
    stride: int,
    sr: int,
    duration: float,
    med: int = 7,
    merge_gap: float = 0.10,
    min_duration: float = 0.15,
) -> List[Dict]:
    if isinstance(probs, torch.Tensor):
        mask = (probs >= threshold).cpu().numpy().astype(np.uint8)
    else:
        mask = (probs >= threshold).astype(np.uint8)

    if med > 1:
        mask = median_filter(mask, size=med, mode="nearest")

    events = []
    if mask.sum() == 0:
        return events

    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        onset, offset = frames_to_seconds(s, e, stride, sr, duration)
        if offset > onset:
            events.append({"onset": onset, "offset": offset})

    # Merge consecutive events within merge_gap
    if merge_gap > 0.0 and len(events) > 1:
        merged = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            if ev["onset"] - prev["offset"] <= merge_gap:
                merged[-1] = {"onset": prev["onset"], "offset": round(max(prev["offset"], ev["offset"]), 3)}
            else:
                merged.append(ev)
        events = merged

    # Filter by min_duration
    if min_duration > 0.0:
        events = [ev for ev in events if round(ev["offset"] - ev["onset"], 3) >= min_duration]

    return events


def evaluate_predictions(gt_map: dict, predictions_map: dict) -> Tuple[float, float, float, int, int, int]:
    sum_tp = sum_fp = sum_fn = 0
    dice_scores = []
    for stem, gt_info in gt_map.items():
        if stem not in predictions_map:
            continue
        pr_events = [(e["onset"], e["offset"]) for e in predictions_map[stem]]
        gt_events = gt_info["events"]

        tp, fp, fn = event_counts(gt_events, pr_events, tol_ratio=0.20, min_collar=0.05)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

        gt_m = merge_intervals(gt_events)
        pr_m = merge_intervals(pr_events)
        inter = intersection_duration(gt_events, pr_events)
        gt_d = total_duration(gt_m)
        pr_d = total_duration(pr_m)
        clip_dice = 1.0 if (gt_d == 0 and pr_d == 0) else (2.0 * inter / (gt_d + pr_d + 1e-9))
        dice_scores.append(clip_dice)

    prec = sum_tp / (sum_tp + sum_fp + 1e-9)
    rec = sum_tp / (sum_tp + sum_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    comb = f1 + dice
    return f1, dice, comb, sum_tp, sum_fp, sum_fn


def resolve_dataset(args) -> Dataset:
    if args.input_data:
        p = Path(args.input_data)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        print(f"Running inference in Codabench input_data mode: {p}")
        return AudioFolderDataset(p, target_sr=WAV2VEC2_SR)

    if args.validation_dir:
        p = Path(args.validation_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        from src.val_loader import CodabenchValDataset
        print(f"Running in Validation mode: {p}")
        return CodabenchValDataset(validation_dir=str(p), target_sr=WAV2VEC2_SR)

    search_paths = [
        PROJECT_ROOT.parent / "Validation" / "validation",
        PROJECT_ROOT / "Validation" / "validation",
        PROJECT_ROOT / "input_data",
        PROJECT_ROOT.parent / "input_data",
    ]
    for candidate in search_paths:
        if candidate.exists():
            if (candidate / "validationMetadata.json").exists():
                from src.val_loader import CodabenchValDataset
                print(f"Auto-discovered Validation dataset at: {candidate}")
                return CodabenchValDataset(validation_dir=str(candidate), target_sr=WAV2VEC2_SR)
            elif (candidate / "audio").is_dir() or any(candidate.glob("*.wav")):
                print(f"Auto-discovered Codabench input data at: {candidate}")
                return AudioFolderDataset(candidate, target_sr=WAV2VEC2_SR)

    raise RuntimeError("No input dataset found. Please provide --input-data or --validation-dir.")


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("VAANI TRACK 1: ADVANCED WAV2VEC2 INFERENCE & EVALUATION")
    print("=" * 70)
    print(f"Model Checkpoint : {args.model}")
    print(f"Device           : {device} (FP16: {args.fp16 and device.type == 'cuda'})")

    model = Wav2Vec2NoiseDetector()
    model = load_checkpoint(model, args.model, device)

    dataset = resolve_dataset(args)
    if args.max_samples and hasattr(dataset, "file_list"):
        dataset.file_list = dataset.file_list[:args.max_samples]
    elif args.max_samples and hasattr(dataset, "samples"):
        dataset.samples = dataset.samples[:args.max_samples]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning GPU forward pass on {len(dataset)} audio clips...")
    start_time = time.time()
    clip_probs_cache = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        clip_id = sample["clip_id"]
        audio_tensor = sample["audio"]
        sr = sample["sample_rate"]
        duration = float(sample.get("duration", len(audio_tensor) / sr))

        waveform = audio_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(args.fp16 and device.type == "cuda")):
                logits = model(waveform).squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy()

        clip_probs_cache.append({
            "clip_id": clip_id,
            "stem": Path(clip_id).stem,
            "probs": probs,
            "duration": duration,
            "sr": sr,
        })

        if (idx + 1) % 250 == 0 or (idx + 1) == len(dataset):
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(elapsed, 1e-3)
            print(f"[{idx + 1:5d}/{len(dataset)}] clips forwarded ({rate:.1f} clips/s)", flush=True)

    elapsed_forward = time.time() - start_time
    print(f"GPU forward pass completed in {elapsed_forward:.1f}s ({len(dataset) / max(elapsed_forward, 1e-3):.1f} clips/s)\n")

    # Load ground truth if evaluating or sweeping
    val_meta = PROJECT_ROOT.parent / "Validation" / "validation" / "validationMetadata.json"
    gt_map = load_ground_truth(val_meta) if val_meta.exists() else None

    # SWEEP MODE
    if args.sweep and gt_map:
        print("=" * 80)
        print("PARAMETER SWEEP ON VALIDATION SET (2,501 CLIPS)")
        print("=" * 80)
        print(f"{'Thresh':<8} {'Med':<5} {'MinDur':<8} {'Gap':<6} {'F1':<8} {'Dice':<8} {'Combined':<10} {'TP':<6} {'FP':<6} {'FN':<6}")
        print("-" * 80)

        thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        medians = [5, 7]
        min_durs = [0.15]
        gaps = [0.10]

        best_comb = -1.0
        best_cfg = None
        best_preds = None

        for th in thresholds:
            for med in medians:
                for md in min_durs:
                    for gp in gaps:
                        cur_preds = {}
                        for item in clip_probs_cache:
                            evs = logits_to_events(
                                item["probs"],
                                threshold=th,
                                stride=WAV2VEC2_STRIDE,
                                sr=item["sr"],
                                duration=item["duration"],
                                med=med,
                                merge_gap=gp,
                                min_duration=md,
                            )
                            cur_preds[item["clip_id"]] = evs
                            cur_preds[item["stem"]] = evs

                        f1, dice, comb, tp, fp, fn = evaluate_predictions(gt_map, cur_preds)
                        print(f"{th:<8.2f} {med:<5d} {md:<8.2f} {gp:<6.2f} {f1:<8.4f} {dice:<8.4f} {comb:<10.4f} {tp:<6d} {fp:<6d} {fn:<6d}")

                        if comb > best_comb:
                            best_comb = comb
                            best_cfg = (th, med, md, gp, f1, dice, tp, fp, fn)
                            best_preds = cur_preds

        print("=" * 80)
        b_th, b_med, b_md, b_gp, b_f1, b_dice, b_tp, b_fp, b_fn = best_cfg
        print("BEST CONFIGURATION FROM SWEEP:")
        print(f"  Threshold   : {b_th:.2f}")
        print(f"  Median Filt : {b_med}")
        print(f"  Min Duration: {b_md:.2f}s")
        print(f"  Merge Gap   : {b_gp:.2f}s")
        print(f"  Event F1    : {b_f1:.4f} (TP: {b_tp}, FP: {b_fp}, FN: {b_fn})")
        print(f"  Segment Dice: {b_dice:.4f}")
        print(f"  COMBINED    : {best_comb:.4f}")
        print(f"  CRNN Baseline: 0.8418")
        if best_comb > 0.8418:
            print(f"  --> STATUS  : BEATS BASELINE (+{best_comb - 0.8418:.4f})! NEW CHAMPION CANDIDATE!")
        else:
            print(f"  --> STATUS  : Below CRNN Baseline ({best_comb - 0.8418:+.4f}). Baseline ID 938825 remains active.")
        print("=" * 80)

        # Write best predictions
        with open(out_path, "w", encoding="utf-8") as fout:
            for item in clip_probs_cache:
                cid = item["clip_id"]
                fout.write(json.dumps({"clip_id": cid, "events": best_preds[cid]}) + "\n")
        print(f"Predictions for best config saved to: {out_path}")

        if args.create_zip and best_comb > 0.8418:
            zip_path = Path(args.zip_out)
            if not zip_path.is_absolute():
                zip_path = PROJECT_ROOT / zip_path
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(out_path, arcname="predictions.jsonl")
            print(f"Submission zip package created: {zip_path}")
        return

    # STANDARD SINGLE-THRESHOLD EVALUATION / INFERENCE
    predictions_map = {}
    with open(out_path, "w", encoding="utf-8") as fout:
        for item in clip_probs_cache:
            events = logits_to_events(
                item["probs"],
                threshold=args.threshold,
                stride=WAV2VEC2_STRIDE,
                sr=item["sr"],
                duration=item["duration"],
                med=args.median_filter,
                merge_gap=args.merge_gap,
                min_duration=args.min_duration,
            )
            predictions_map[item["clip_id"]] = events
            predictions_map[item["stem"]] = events
            fout.write(json.dumps({"clip_id": item["clip_id"], "events": events}) + "\n")

    print(f"Predictions written to: {out_path}")

    if (args.eval or args.validation_dir) and gt_map:
        f1, dice, comb, tp, fp, fn = evaluate_predictions(gt_map, predictions_map)
        print("=" * 70)
        print("WAV2VEC2 VALIDATION SCORE")
        print("=" * 70)
        print(f"Event F1 (Micro): {f1:.4f} (TP: {tp}, FP: {fp}, FN: {fn})")
        print(f"Segment Dice    : {dice:.4f}")
        print(f"COMBINED SCORE  : {comb:.4f}  (CRNN Baseline to beat: 0.8418)")
        print("=" * 70)

    if args.create_zip:
        zip_path = Path(args.zip_out)
        if not zip_path.is_absolute():
            zip_path = PROJECT_ROOT / zip_path
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_path, arcname="predictions.jsonl")
        print(f"\nCodabench submission package created: {zip_path}")
        print("=" * 70)


if __name__ == "__main__":
    main()
