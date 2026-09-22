"""
sweep_decoding.py
Fast, exact validation sweep over prediction thresholds and median-filter values
for Vaani Noise Event Detection.

Caches frame probabilities from a single model forward pass over all 2,501 validation clips,
then rapidly evaluates different decoding configurations using the official competition metrics.
"""

import io
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

# Ensure Windows-safe console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from scipy.ndimage import median_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.val_loader import CodabenchValDataset
from src.features import LogMelExtractor
from src.model import CRNNNoiseDetector
from src.infer import load_checkpoint, frames_to_seconds
from src.eval_local import (
    load_ground_truth,
    event_counts,
    merge_intervals,
    total_duration,
    intersection_duration,
)


def probs_to_events(
    probs: np.ndarray,
    threshold: float,
    med: int,
    hop_length: int = 512,
    sr: int = 16000,
    duration: float = 10.0,
) -> List[Tuple[float, float]]:
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
        onset, offset = frames_to_seconds(s, e, hop_length, sr, duration)
        if offset > onset:
            events.append((onset, offset))
    return events


def evaluate_configuration(
    cached_probs: List[dict],
    gt_map: dict,
    threshold: float,
    med: int,
    tolerance: float = 0.20,
    min_collar: float = 0.05,
) -> dict:
    sum_tp = sum_fp = sum_fn = 0
    dice_scores = []

    for item in cached_probs:
        stem = item["stem"]
        if stem not in gt_map:
            continue

        gt_info = gt_map[stem]
        gt_events = gt_info["events"]

        pr_events = probs_to_events(
            item["probs"],
            threshold=threshold,
            med=med,
            hop_length=item["hop_length"],
            sr=item["sr"],
            duration=item["duration"],
        )

        # 1. Event F1 collar matching
        tp, fp, fn = event_counts(gt_events, pr_events, tol_ratio=tolerance, min_collar=min_collar)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

        # 2. Segment-level Dice
        gt_merged = merge_intervals(gt_events)
        pr_merged = merge_intervals(pr_events)
        inter = intersection_duration(gt_events, pr_events)
        gt_d = total_duration(gt_merged)
        pr_d = total_duration(pr_merged)

        if gt_d == 0 and pr_d == 0:
            clip_dice = 1.0
        else:
            clip_dice = (2.0 * inter) / (gt_d + pr_d + 1e-9)

        dice_scores.append(clip_dice)

    prec = sum_tp / (sum_tp + sum_fp + 1e-9)
    rec = sum_tp / (sum_tp + sum_fn + 1e-9)
    f1_micro = 2 * prec * rec / (prec + rec + 1e-9)
    dice_macro = float(np.mean(dice_scores)) if dice_scores else 0.0
    combined = f1_micro + dice_macro

    return {
        "threshold": threshold,
        "med": med,
        "f1": f1_micro,
        "dice": dice_macro,
        "combined": combined,
        "tp": sum_tp,
        "fp": sum_fp,
        "fn": sum_fn,
        "precision": prec,
        "recall": rec,
        "total_preds": sum_tp + sum_fp,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Decoding sweep")
    parser.add_argument("--model", type=str, default="models/baseline_crnn_5000_gold.pt",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    val_dir = PROJECT_ROOT.parent / "Validation" / "validation"
    val_meta_path = val_dir / "validationMetadata.json"
    device = torch.device("cpu")

    print("=" * 65)
    print(f"VAANI TRACK 1: VALIDATION DECODING SWEEP")
    print("=" * 65)
    print(f"Checkpoint: {model_path}")
    print(f"Validation: {val_dir}")

    # 1. Load model & ground truth
    model = CRNNNoiseDetector(n_mels=64)
    model = load_checkpoint(model, str(model_path), device)
    model.eval()

    gt_map = load_ground_truth(Path(val_meta_path))
    dataset = CodabenchValDataset(validation_dir=str(val_dir))
    extractor = LogMelExtractor(sr=16000, n_mels=64, n_fft=1024, hop_length=512)

    # 2. Precompute model probabilities for all 2,501 validation clips (one forward pass)
    print(f"\nComputing model frame probabilities for {len(dataset)} clips (single pass)...")
    t0 = time.time()
    cached_probs = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        clip_id = sample["clip_id"]
        stem = Path(clip_id).stem
        audio = sample["audio"].numpy()
        sr = sample["sample_rate"]
        duration = float(sample.get("duration", len(audio) / sr))

        try:
            feats = extractor.extract(audio)
            feats = feats.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(feats).squeeze(0)
                probs = torch.sigmoid(logits).cpu().numpy()
        except Exception as e:
            probs = np.zeros(1, dtype=np.float32)

        cached_probs.append({
            "stem": stem,
            "probs": probs,
            "hop_length": extractor.hop_length,
            "sr": extractor.sr,
            "duration": duration,
        })

    print(f"Forward passes completed in {time.time() - t0:.2f}s. Memory cached for {len(cached_probs)} clips.")

    # 3. Define sweep parameters
    thresholds = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    median_filters = [1, 3, 5, 7]

    print(f"\nSweeping {len(thresholds)} thresholds x {len(median_filters)} median filters ({len(thresholds)*len(median_filters)} total configurations)...")
    results = []
    t_sweep = time.time()

    for thresh in thresholds:
        for med in median_filters:
            t_cfg = time.time()
            res = evaluate_configuration(cached_probs, gt_map, threshold=thresh, med=med)
            results.append(res)
            print(
                f"Thresh: {thresh:.2f} | Med: {med} | F1: {res['f1']:.4f} | "
                f"Dice: {res['dice']:.4f} | Comb: {res['combined']:.4f} | "
                f"TP: {res['tp']:4d} | FP: {res['fp']:5d} | FN: {res['fn']:4d}"
            )

    print(f"\nSweep completed in {time.time() - t_sweep:.2f}s.")

    # Sort results by combined score descending
    results.sort(key=lambda x: x["combined"], reverse=True)
    best = results[0]

    print("\n" + "=" * 80)
    print(f"{'Thresh':<8}{'Med':<6}{'Event F1':<12}{'Dice':<12}{'Combined':<12}{'TP':<8}{'FP':<8}{'FN':<8}{'Preds':<8}")
    print("-" * 80)
    for r in results:
        marker = " <== BEST" if r == best else ""
        print(
            f"{r['threshold']:<8.2f}{r['med']:<6}{r['f1']:<12.4f}{r['dice']:<12.4f}{r['combined']:<12.4f}"
            f"{r['tp']:<8}{r['fp']:<8}{r['fn']:<8}{r['total_preds']:<8}{marker}"
        )
    print("=" * 80)
    print(
        f"\nHighest Combined Score: {best['combined']:.4f} achieved at "
        f"Threshold={best['threshold']:.2f}, Median-Filter={best['med']}"
    )
    print(
        f"Metrics -> Event F1: {best['f1']:.4f}, Segment Dice: {best['dice']:.4f}, "
        f"TP: {best['tp']}, FP: {best['fp']}, FN: {best['fn']}"
    )


if __name__ == "__main__":
    main()
