"""
eval_local.py
Official Competition-aligned Local Evaluation Scorer for Track 1: Noise Event Detection.

Calculates:
1. Event-based F1 (Micro-averaged across dataset):
   - Collar tolerance: max(20% of GT event duration, 50 ms) on both onset and offset
   - Greedy 1-to-1 matching minimizing edge error
2. Segment-level Dice (Macro-averaged across clips):
   - 2 * Intersection / (GT_total + Pred_total)
3. Combined Score:
   - Combined = Event_F1 + Dice (0.0 to 2.0)

Usage:
    python src/eval_local.py --predictions submissions/predictions.jsonl --validation-meta Validation/validation/validationMetadata.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Interval math & collar matching (exact competition logic)
# ---------------------------------------------------------------------------
def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping intervals into a disjoint, sorted list."""
    if not intervals:
        return []
    itv = sorted(intervals)
    out = [list(itv[0])]
    for a, b in itv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def total_duration(intervals: List[Tuple[float, float]]) -> float:
    """Total non-overlapping duration."""
    return float(sum(max(0.0, b - a) for a, b in intervals))


def intersection_duration(a_list: List[Tuple[float, float]], b_list: List[Tuple[float, float]]) -> float:
    """Total overlap duration between two interval lists (both merged first)."""
    a_list, b_list = merge_intervals(a_list), merge_intervals(b_list)
    i = j = 0
    inter = 0.0
    while i < len(a_list) and j < len(b_list):
        a1, a2 = a_list[i]
        b1, b2 = b_list[j]
        lo, hi = max(a1, b1), min(a2, b2)
        if hi > lo:
            inter += hi - lo
        if a2 < b2:
            i += 1
        else:
            j += 1
    return inter


def event_match(g: Tuple[float, float], p: Tuple[float, float], tol_ratio: float = 0.20, min_collar: float = 0.05) -> bool:
    """Collar match: onset and offset within max(20% of GT duration, 50 ms)."""
    dur = max(1e-6, g[1] - g[0])
    tol = max(tol_ratio * dur, min_collar)
    return abs(g[0] - p[0]) <= tol and abs(g[1] - p[1]) <= tol


def event_counts(gt_events: List[Tuple[float, float]], pr_events: List[Tuple[float, float]],
                 tol_ratio: float = 0.20, min_collar: float = 0.05) -> Tuple[int, int, int]:
    """Greedy 1-to-1 event matching -> (tp, fp, fn)."""
    matched, tp = set(), 0
    for g in gt_events:
        best_j, best_err = -1, 1e18
        for j, p in enumerate(pr_events):
            if j in matched or not event_match(g, p, tol_ratio=tol_ratio, min_collar=min_collar):
                continue
            err = abs(g[0] - p[0]) + abs(g[1] - p[1])
            if err < best_err:
                best_err, best_j = err, j
        if best_j >= 0:
            matched.add(best_j)
            tp += 1
    return tp, len(pr_events) - tp, len(gt_events) - tp


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_predictions(pred_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Load predictions JSONL into a mapping: clip_id -> list of (onset, offset)."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    preds = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"[Warning] Failed to parse prediction line {line_num}: {e}")
                continue

            cid = str(record.get("clip_id", "")).strip()
            events = []
            for ev in record.get("events", []) or []:
                try:
                    on, off = float(ev["onset"]), float(ev["offset"])
                    if off > on:
                        events.append((on, off))
                except (ValueError, KeyError, TypeError):
                    continue
            events = sorted(events)
            preds[cid] = events
            # Also map stem if cid has extension, or with .wav
            stem = Path(cid).stem
            preds[stem] = events
            if not cid.endswith(".wav"):
                preds[f"{cid}.wav"] = events

    return preds


def load_ground_truth(meta_path: Path) -> Dict[str, Dict]:
    """
    Load ground truth metadata from validationMetadata.json or JSONL.
    Returns: dict mapping clip_id -> {events, synthetic, duration, etc.}
    """
    # Handle folder path passed directly
    if meta_path.is_dir():
        if (meta_path / "validationMetadata.json").exists():
            meta_path = meta_path / "validationMetadata.json"
        elif (meta_path / "validation" / "validationMetadata.json").exists():
            meta_path = meta_path / "validation" / "validationMetadata.json"

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    gt = {}
    if content.startswith("["):
        # JSON Array (validationMetadata.json format)
        records = json.loads(content)
        for r in records:
            fname = r.get("segmentFileName", "")
            stem = Path(fname).stem
            events = []
            for ev in r.get("NoiseSubCategoryTimeStamp", []) or []:
                if isinstance(ev, dict) and "start" in ev and "end" in ev:
                    try:
                        on, off = float(ev["start"]), float(ev["end"])
                        if off > on:
                            events.append((on, off))
                    except (ValueError, TypeError):
                        continue
            entry = {
                "events": sorted(events),
                "synthetic": bool(r.get("syntheticData", False)),
                "duration": float(r.get("duration", 0.0)),
                "segmentFileName": fname,
            }
            gt[stem] = entry
            gt[fname] = entry
    else:
        # JSONL format (unified.jsonl / val_split.jsonl)
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cid = str(r.get("clip_id", "")).strip()
            events = []
            for ev in r.get("events", []) or []:
                try:
                    on, off = float(ev["onset"]), float(ev["offset"])
                    if off > on:
                        events.append((on, off))
                except (ValueError, KeyError, TypeError):
                    continue
            entry = {
                "events": sorted(events),
                "synthetic": False,
                "duration": float(r.get("duration", 0.0)),
                "segmentFileName": cid,
            }
            gt[cid] = entry
            gt[Path(cid).stem] = entry

    return gt


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------
def evaluate(predictions_path: str, validation_meta_path: str,
             tolerance: float = 0.20, min_collar: float = 0.05) -> Dict:
    pred_map = load_predictions(Path(predictions_path))
    gt_map = load_ground_truth(Path(validation_meta_path))

    # Evaluate predictions present in predictions_path against their ground truth
    # Get unique evaluated clip stems
    evaluated_stems = set()
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    raw_id = str(r.get("clip_id", ""))
                    evaluated_stems.add(Path(raw_id).stem)
                except Exception:
                    pass

    total_clips = len(evaluated_stems)
    if total_clips == 0:
        raise ValueError(f"No valid predictions found in {predictions_path}")

    # Accumulators
    sum_tp = sum_fp = sum_fn = 0
    dice_scores = []

    # Breakdown accumulators
    nat_tp = nat_fp = nat_fn = 0
    nat_dice = []
    syn_tp = syn_fp = syn_fn = 0
    syn_dice = []

    missing_in_gt = 0

    for stem in sorted(evaluated_stems):
        if stem not in gt_map:
            missing_in_gt += 1
            continue

        gt_info = gt_map[stem]
        gt_events = gt_info["events"]
        pr_events = pred_map.get(stem, [])
        is_syn = gt_info["synthetic"]

        # 1. Event F1 counts (greedy 1-1 collar matching)
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
            clip_dice = 1.0  # Perfect true negative
        else:
            clip_dice = (2.0 * inter) / (gt_d + pr_d + 1e-9)

        dice_scores.append(clip_dice)

        if is_syn:
            syn_tp += tp
            syn_fp += fp
            syn_fn += fn
            syn_dice.append(clip_dice)
        else:
            nat_tp += tp
            nat_fp += fp
            nat_fn += fn
            nat_dice.append(clip_dice)

    # Compute micro metrics
    prec = sum_tp / (sum_tp + sum_fp + 1e-9)
    rec = sum_tp / (sum_tp + sum_fn + 1e-9)
    f1_micro = 2 * prec * rec / (prec + rec + 1e-9)
    dice_macro = float(np.mean(dice_scores)) if dice_scores else 0.0
    combined = f1_micro + dice_macro

    # Print results
    print("=" * 65)
    print("VAANI TRACK 1: LOCAL EVALUATION RESULTS")
    print("=" * 65)
    print(f"Predictions File       : {predictions_path}")
    print(f"Validation Ground Truth: {validation_meta_path}")
    print(f"Clips Evaluated        : {len(dice_scores)} (Missing in GT: {missing_in_gt})")
    print(f"Event Collar Tolerance : max({tolerance*100:.0f}% of GT duration, {min_collar*1000:.0f}ms)")
    print("-" * 65)
    print(f"Micro-Average Counts   : TP={sum_tp}, FP={sum_fp}, FN={sum_fn}")
    print(f"Precision              : {prec:.4f}")
    print(f"Recall                 : {rec:.4f}")
    print(f"Event-based F1 (Micro) : {f1_micro:.4f}")
    print(f"Segment-level Dice     : {dice_macro:.4f}")
    print(f"COMBINED SCORE (F1+Dice: {combined:.4f}  (Max: 2.0000)")
    print("-" * 65)

    if nat_dice:
        n_p = nat_tp / (nat_tp + nat_fp + 1e-9)
        n_r = nat_tp / (nat_tp + nat_fn + 1e-9)
        n_f1 = 2 * n_p * n_r / (n_p + n_r + 1e-9)
        n_dice = float(np.mean(nat_dice))
        print(f"Natural Audio ({len(nat_dice)} clips)  : F1={n_f1:.4f}, Dice={n_dice:.4f}, Comb={n_f1 + n_dice:.4f}")

    if syn_dice:
        s_p = syn_tp / (syn_tp + syn_fp + 1e-9)
        s_r = syn_tp / (syn_tp + syn_fn + 1e-9)
        s_f1 = 2 * s_p * s_r / (s_p + s_r + 1e-9)
        s_dice = float(np.mean(syn_dice))
        print(f"Synthetic Audio ({len(syn_dice)} clips): F1={s_f1:.4f}, Dice={s_dice:.4f}, Comb={s_f1 + s_dice:.4f}")
    print("=" * 65)

    return {
        "clips_evaluated": len(dice_scores),
        "precision": prec,
        "recall": rec,
        "event_f1_micro": f1_micro,
        "dice_macro": dice_macro,
        "combined": combined,
    }


def main():
    parser = argparse.ArgumentParser(description="Local evaluation for Track 1: Noise Event Detection")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions.jsonl file")
    parser.add_argument("--validation-meta", type=str,
                        default="Validation/validation/validationMetadata.json",
                        help="Path to validationMetadata.json or folder")
    parser.add_argument("--tolerance", type=float, default=0.20,
                        help="Collar duration ratio tolerance (default: 0.20 for +-20%)")
    parser.add_argument("--min-collar", type=float, default=0.05,
                        help="Minimum collar in seconds (default: 0.05 for 50ms)")
    args = parser.parse_args()

    evaluate(
        predictions_path=args.predictions,
        validation_meta_path=args.validation_meta,
        tolerance=args.tolerance,
        min_collar=args.min_collar,
    )


if __name__ == "__main__":
    main()
