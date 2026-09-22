"""
sweep_fine_postprocess.py
Finer threshold sweep & post-processing (min event duration, merge gap) optimization
for the 9,426-Gold CRNN model on the 2,501 validation clips.

Evaluates against official competition metrics via src/eval_local.py functions:
- Event-based F1 (collar tolerance: max(20% GT, 50ms))
- Segment-level Dice (macro across clips)
- Combined = Event_F1 + Dice
"""

import io
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Windows console UTF-8 safety
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


def postprocess_events(
    events: List[Tuple[float, float]],
    merge_gap: float = 0.0,
    min_duration: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Apply merge gap bridging and minimum duration pruning to extracted event intervals.
    """
    if not events:
        return []

    # 1. Merge gap: combine consecutive events separated by <= merge_gap
    if merge_gap > 0.0:
        merged = [events[0]]
        for on, off in events[1:]:
            prev_on, prev_off = merged[-1]
            if on - prev_off <= merge_gap:
                merged[-1] = (prev_on, max(prev_off, off))
            else:
                merged.append((on, off))
        events = merged

    # 2. Min duration: remove events shorter than min_duration
    if min_duration > 0.0:
        events = [(on, off) for on, off in events if round(off - on, 3) >= min_duration]

    return events


def probs_to_events_post(
    probs: np.ndarray,
    threshold: float,
    med: int,
    hop_length: int = 512,
    sr: int = 16000,
    duration: float = 10.0,
    merge_gap: float = 0.0,
    min_duration: float = 0.0,
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

    if merge_gap > 0.0 or min_duration > 0.0:
        events = postprocess_events(events, merge_gap=merge_gap, min_duration=min_duration)

    return events


def evaluate_configuration(
    cached_probs: List[dict],
    gt_map: dict,
    threshold: float,
    med: int,
    merge_gap: float = 0.0,
    min_duration: float = 0.0,
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

        pr_events = probs_to_events_post(
            item["probs"],
            threshold=threshold,
            med=med,
            hop_length=item["hop_length"],
            sr=item["sr"],
            duration=item["duration"],
            merge_gap=merge_gap,
            min_duration=min_duration,
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
        "merge_gap": merge_gap,
        "min_duration": min_duration,
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
    model_path = PROJECT_ROOT / "models" / "baseline_crnn_9426_gold.pt"
    val_dir = PROJECT_ROOT.parent / "Validation" / "validation"
    val_meta_path = val_dir / "validationMetadata.json"
    device = torch.device("cpu")

    print("=" * 75)
    print("VAANI TRACK 1: 9,426-GOLD FINER THRESHOLD & POST-PROCESSING SWEEP")
    print("=" * 75)
    print(f"Checkpoint : {model_path}")
    print(f"Validation : {val_dir}")

    # 1. Load model & ground truth
    model = CRNNNoiseDetector(n_mels=64)
    model = load_checkpoint(model, str(model_path), device)
    model.eval()

    gt_map = load_ground_truth(Path(val_meta_path))
    dataset = CodabenchValDataset(validation_dir=str(val_dir))
    extractor = LogMelExtractor(sr=16000, n_mels=64, n_fft=1024, hop_length=512)

    # 2. Precompute model frame probabilities (single pass in ~45s)
    print(f"\nComputing model frame probabilities for {len(dataset)} clips...")
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

    print(f"Forward passes completed in {time.time() - t0:.2f}s. Caching in memory complete.")

    # -----------------------------------------------------------------------
    # Part 1: Fine Threshold Sweep (med=7, merge_gap=0, min_dur=0)
    # Requested: 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46
    # -----------------------------------------------------------------------
    fine_thresholds = [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46]
    print(f"\n--- Part 1: Fine Threshold Sweep ({len(fine_thresholds)} thresholds, med=7) ---")
    part1_results = []
    for th in fine_thresholds:
        res = evaluate_configuration(cached_probs, gt_map, threshold=th, med=7, merge_gap=0.0, min_duration=0.0)
        part1_results.append(res)
        print(
            f"Thresh: {th:.2f} | Med: 7 | F1: {res['f1']:.4f} | "
            f"Dice: {res['dice']:.4f} | Comb: {res['combined']:.4f} | "
            f"TP: {res['tp']:4d} | FP: {res['fp']:5d} | FN: {res['fn']:4d}"
        )

    part1_results.sort(key=lambda x: x["combined"], reverse=True)
    best_th = part1_results[0]["threshold"]
    print(f"\nPart 1 Best: Threshold={best_th:.2f} -> Combined: {part1_results[0]['combined']:.4f} (F1: {part1_results[0]['f1']:.4f}, Dice: {part1_results[0]['dice']:.4f})")

    # -----------------------------------------------------------------------
    # Part 2: Post-processing exploration (min_duration & merge_gap)
    # Test around the top 3 thresholds from Part 1
    # -----------------------------------------------------------------------
    top_thresholds = sorted(list(set([r["threshold"] for r in part1_results[:3]])))
    min_durations = [0.0, 0.05, 0.10, 0.15, 0.20]
    merge_gaps = [0.0, 0.05, 0.10, 0.15, 0.20]

    print(f"\n--- Part 2: Post-Processing Sweep (Thresholds: {top_thresholds}, MinDur: {min_durations}, MergeGap: {merge_gaps}) ---")
    part2_results = []
    t_p2 = time.time()
    for th in top_thresholds:
        for md in min_durations:
            for mg in merge_gaps:
                if md == 0.0 and mg == 0.0:
                    continue  # Already evaluated in Part 1
                res = evaluate_configuration(cached_probs, gt_map, threshold=th, med=7, merge_gap=mg, min_duration=md)
                part2_results.append(res)

    print(f"Part 2 completed ({len(part2_results)} configurations in {time.time() - t_p2:.2f}s).")

    # Combine and analyze all results
    all_results = part1_results + part2_results
    all_results.sort(key=lambda x: x["combined"], reverse=True)

    baseline_score = 0.8325  # from default coarse sweep (th=0.40, med=7)
    best = all_results[0]

    print("\n" + "=" * 95)
    print(f"{'Rank':<5}{'Thresh':<8}{'Med':<5}{'MinDur':<8}{'MergeGap':<10}{'Event F1':<11}{'Dice':<11}{'Combined':<11}{'TP':<7}{'FP':<7}{'FN':<7}")
    print("-" * 95)
    for idx, r in enumerate(all_results[:15], 1):
        marker = " <== TOP" if idx == 1 else ""
        print(
            f"{idx:<5}{r['threshold']:<8.2f}{r['med']:<5}{r['min_duration']:<8.2f}{r['merge_gap']:<10.2f}"
            f"{r['f1']:<11.4f}{r['dice']:<11.4f}{r['combined']:<11.4f}"
            f"{r['tp']:<7}{r['fp']:<7}{r['fn']:<7}{marker}"
        )
    print("=" * 95)

    print("\n" + "=" * 75)
    print("FINAL SUMMARY & COMPARISON")
    print("=" * 75)
    print(f"Previous Baseline (Coarse) : Combined = {baseline_score:.4f} (th=0.40, med=7)")
    print(f"New Best Configuration     : Combined = {best['combined']:.4f}")
    print(f"  - Threshold    : {best['threshold']:.2f}")
    print(f"  - Median Filter: {best['med']}")
    print(f"  - Min Duration : {best['min_duration']:.2f}s")
    print(f"  - Merge Gap    : {best['merge_gap']:.2f}s")
    print(f"  - Event F1     : {best['f1']:.4f} (Micro TP={best['tp']}, FP={best['fp']}, FN={best['fn']})")
    print(f"  - Segment Dice : {best['dice']:.4f}")
    diff = best['combined'] - baseline_score
    if diff > 1e-4:
        print(f"  - Improvement  : +{diff:.4f} over baseline {baseline_score:.4f}")
    else:
        print(f"  - Outcome      : Baseline 0.8325 remains the best configuration (diff: {diff:+.4f})")
    print("=" * 75)


if __name__ == "__main__":
    main()
