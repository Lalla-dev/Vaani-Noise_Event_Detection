"""
ensemble_eval.py
Track 1 Ensemble Pipeline: CRNN (9426 Gold) + Wav2Vec2 (9426 Gold).

Objectives:
1. Extract frame-level probabilities from both models across all 2,501 validation clips.
2. Save CRNN and Wav2Vec2 probabilities separately into data/cache/ensemble/.
3. Align the Wav2Vec2 20 ms probability grid with the CRNN grid (supporting 10 ms, 20 ms, and 32 ms grids).
4. Evaluate weighted combinations: P_ens = alpha * P_w2v + (1 - alpha) * P_crnn
   for alpha in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0].
5. Tune threshold, median filter, minimum event duration, and merge gap.
6. Evaluate against official local metrics: Micro Event F1, Macro Segment Dice, Combined Score (overall, natural, synthetic).
7. Save predictions to submissions/predictions_ensemble_crnn_wav2vec2.jsonl without touching protected files.
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import median_filter

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
from src.val_loader import CodabenchValDataset
from src.eval_local import (
    load_ground_truth,
    event_counts,
    merge_intervals,
    total_duration,
    intersection_duration,
)


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: Union[str, Path], device: torch.device):
    p = Path(ckpt_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {p}")
    ckpt = torch.load(str(p), map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def precompute_crnn_probs(
    dataset: CodabenchValDataset,
    crnn_ckpt: str,
    device: torch.device,
    cache_path: Path,
    force_recompute: bool = False,
) -> Dict[str, dict]:
    """Extract and cache frame probabilities for CRNN on 2,501 validation clips."""
    if cache_path.exists() and not force_recompute:
        print(f"[CRNN] Loading cached probabilities from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    print(f"\n[CRNN] Running forward pass on {len(dataset)} clips using {crnn_ckpt}...")
    model = CRNNNoiseDetector(n_mels=64)
    model = load_model_checkpoint(model, crnn_ckpt, device)
    extractor = LogMelExtractor(sr=16000, n_mels=64, n_fft=1024, hop_length=512)

    t0 = time.time()
    cache = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        clip_id = sample["clip_id"]
        stem = Path(clip_id).stem
        audio = sample["audio"].numpy()
        sr = sample["sample_rate"]
        duration = float(sample.get("duration", len(audio) / sr))

        try:
            feats = extractor.extract(audio)
            feats = feats.unsqueeze(0).to(device)  # [1, T, 64]
            with torch.no_grad():
                logits = model(feats).squeeze(0)  # [T]
                probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[Warning] CRNN extraction failed for {clip_id}: {e}")
            probs = np.zeros(1, dtype=np.float32)

        cache[stem] = {
            "clip_id": clip_id,
            "stem": stem,
            "probs": probs,
            "hop_length": extractor.hop_length,
            "sr": extractor.sr,
            "duration": duration,
        }

        if (idx + 1) % 500 == 0 or (idx + 1) == len(dataset):
            rate = (idx + 1) / max(time.time() - t0, 1e-3)
            print(f"  CRNN: [{idx + 1:5d}/{len(dataset)}] clips forwarded ({rate:.1f} clips/s)", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[CRNN] Probabilities saved to {cache_path} ({len(cache)} clips, {time.time() - t0:.1f}s)\n")
    return cache


def precompute_wav2vec2_probs(
    dataset: CodabenchValDataset,
    w2v_ckpt: str,
    device: torch.device,
    cache_path: Path,
    fp16: bool = True,
    force_recompute: bool = False,
) -> Dict[str, dict]:
    """Extract and cache frame probabilities for Wav2Vec2 on 2,501 validation clips."""
    if cache_path.exists() and not force_recompute:
        print(f"[Wav2Vec2] Loading cached probabilities from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    print(f"\n[Wav2Vec2] Running forward pass on {len(dataset)} clips using {w2v_ckpt}...")
    model = Wav2Vec2NoiseDetector()
    model = load_model_checkpoint(model, w2v_ckpt, device)

    t0 = time.time()
    cache = {}

    for idx in range(len(dataset)):
        sample = dataset[idx]
        clip_id = sample["clip_id"]
        stem = Path(clip_id).stem
        audio_tensor = sample["audio"]
        sr = sample["sample_rate"]
        duration = float(sample.get("duration", len(audio_tensor) / sr))

        waveform = audio_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(fp16 and device.type == "cuda")):
                logits = model(waveform).squeeze(0)
            probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

        cache[stem] = {
            "clip_id": clip_id,
            "stem": stem,
            "probs": probs,
            "stride": WAV2VEC2_STRIDE,
            "sr": WAV2VEC2_SR,
            "duration": duration,
        }

        if (idx + 1) % 250 == 0 or (idx + 1) == len(dataset):
            rate = (idx + 1) / max(time.time() - t0, 1e-3)
            print(f"  Wav2Vec2: [{idx + 1:5d}/{len(dataset)}] clips forwarded ({rate:.1f} clips/s)", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[Wav2Vec2] Probabilities saved to {cache_path} ({len(cache)} clips, {time.time() - t0:.1f}s)\n")
    return cache


def align_and_blend_probs(
    crnn_item: dict,
    w2v_item: dict,
    alpha: float,
    grid_mode: str = "10ms",
) -> Tuple[np.ndarray, float]:
    """
    Align CRNN and Wav2Vec2 probabilities to a common temporal grid and compute weighted blend:
    P_blend = alpha * P_w2v + (1 - alpha) * P_crnn

    Returns:
        blend_probs: 1D np.ndarray of probabilities
        frame_step_sec: duration of 1 frame in seconds (e.g. 0.010 for 10ms, 0.020 for 20ms)
    """
    duration = max(crnn_item["duration"], w2v_item["duration"], 0.01)
    p_crnn = crnn_item["probs"]
    p_w2v = w2v_item["probs"]

    T_crnn = len(p_crnn)
    T_w2v = len(p_w2v)

    # Center timestamps of native frames
    t_crnn = (np.arange(T_crnn) + 0.5) * (crnn_item["hop_length"] / crnn_item["sr"])
    t_w2v = (np.arange(T_w2v) + 0.5) * (w2v_item["stride"] / w2v_item["sr"])

    if grid_mode == "10ms":
        frame_step_sec = 0.010
        M = max(1, int(round(duration / frame_step_sec)))
        t_target = (np.arange(M) + 0.5) * frame_step_sec

        interp_crnn = np.interp(t_target, t_crnn, p_crnn)
        interp_w2v = np.interp(t_target, t_w2v, p_w2v)
        blend = alpha * interp_w2v + (1.0 - alpha) * interp_crnn
        return blend.astype(np.float32), frame_step_sec

    elif grid_mode == "20ms":
        # Native Wav2Vec2 grid (20 ms, stride=320 @ 16kHz)
        frame_step_sec = w2v_item["stride"] / w2v_item["sr"]
        interp_crnn = np.interp(t_w2v, t_crnn, p_crnn)
        blend = alpha * p_w2v + (1.0 - alpha) * interp_crnn
        return blend.astype(np.float32), frame_step_sec

    elif grid_mode == "32ms":
        # Native CRNN grid (32 ms, hop=512 @ 16kHz)
        frame_step_sec = crnn_item["hop_length"] / crnn_item["sr"]
        interp_w2v = np.interp(t_crnn, t_w2v, p_w2v)
        blend = alpha * interp_w2v + (1.0 - alpha) * p_crnn
        return blend.astype(np.float32), frame_step_sec

    else:
        raise ValueError(f"Unknown grid_mode: {grid_mode}")


def probs_to_events(
    probs: np.ndarray,
    threshold: float,
    frame_step_sec: float,
    duration: float,
    med: int = 7,
    merge_gap: float = 0.10,
    min_duration: float = 0.15,
) -> List[Tuple[float, float]]:
    """Convert 1D frame probabilities to event intervals with post-processing."""
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
        onset = round(s * frame_step_sec, 3)
        offset = round(min(e * frame_step_sec, duration), 3)
        if offset > onset:
            events.append((onset, offset))

    # Merge gap
    if merge_gap > 0.0 and len(events) > 1:
        merged = [events[0]]
        for on, off in events[1:]:
            prev_on, prev_off = merged[-1]
            if on - prev_off <= merge_gap:
                merged[-1] = (prev_on, max(prev_off, off))
            else:
                merged.append((on, off))
        events = merged

    # Min duration
    if min_duration > 0.0:
        events = [(on, off) for on, off in events if round(off - on, 3) >= min_duration]

    return events


def evaluate_predictions_dict(
    gt_map: dict,
    predictions: Dict[str, List[Tuple[float, float]]],
    tolerance: float = 0.20,
    min_collar: float = 0.05,
) -> dict:
    """Evaluate predictions against ground truth, computing micro F1, macro Dice, and subset metrics."""
    sum_tp = sum_fp = sum_fn = 0
    dice_scores = []

    nat_tp = nat_fp = nat_fn = 0
    nat_dice = []
    syn_tp = syn_fp = syn_fn = 0
    syn_dice = []

    for stem, gt_info in gt_map.items():
        if stem not in predictions:
            continue
        gt_events = gt_info["events"]
        pr_events = predictions[stem]
        is_syn = gt_info["synthetic"]

        tp, fp, fn = event_counts(gt_events, pr_events, tol_ratio=tolerance, min_collar=min_collar)
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

    prec = sum_tp / (sum_tp + sum_fp + 1e-9)
    rec = sum_tp / (sum_tp + sum_fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    comb = f1 + dice

    # Natural subset
    n_p = nat_tp / (nat_tp + nat_fp + 1e-9)
    n_r = nat_tp / (nat_tp + nat_fn + 1e-9)
    n_f1 = 2 * n_p * n_r / (n_p + n_r + 1e-9)
    n_dice = float(np.mean(nat_dice)) if nat_dice else 0.0

    # Synthetic subset
    s_p = syn_tp / (syn_tp + syn_fp + 1e-9)
    s_r = syn_tp / (syn_tp + syn_fn + 1e-9)
    s_f1 = 2 * s_p * s_r / (s_p + s_r + 1e-9)
    s_dice = float(np.mean(syn_dice)) if syn_dice else 0.0

    return {
        "clips": len(dice_scores),
        "f1": f1,
        "dice": dice,
        "combined": comb,
        "tp": sum_tp,
        "fp": sum_fp,
        "fn": sum_fn,
        "precision": prec,
        "recall": rec,
        "nat_f1": n_f1,
        "nat_dice": n_dice,
        "nat_comb": n_f1 + n_dice,
        "syn_f1": s_f1,
        "syn_dice": s_dice,
        "syn_comb": s_f1 + s_dice,
    }


def main():
    parser = argparse.ArgumentParser(description="Ensemble CRNN and Wav2Vec2 models on Track 1")
    parser.add_argument("--crnn-model", type=str, default="models/baseline_crnn_9426_gold.pt")
    parser.add_argument("--w2v-model", type=str, default="models/advanced_wav2vec2_9426_gold.pt")
    parser.add_argument("--validation-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recompute-probs", action="store_true", default=False)
    parser.add_argument("--out", type=str, default="submissions/predictions_ensemble_crnn_wav2vec2.jsonl")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Locate validation folder
    search_paths = [
        PROJECT_ROOT.parent / "Validation" / "validation",
        PROJECT_ROOT / "Validation" / "validation",
        PROJECT_ROOT.parent / "Validation",
    ]
    val_dir = None
    if args.validation_dir:
        val_dir = Path(args.validation_dir)
    else:
        for p in search_paths:
            if p.exists() and ((p / "validationMetadata.json").exists() or (p / "validation" / "validationMetadata.json").exists()):
                val_dir = p
                break

    if not val_dir:
        raise FileNotFoundError("Validation directory not found.")

    val_meta = val_dir / "validationMetadata.json" if (val_dir / "validationMetadata.json").exists() else val_dir / "validation" / "validationMetadata.json"
    gt_map = load_ground_truth(val_meta)
    dataset = CodabenchValDataset(validation_dir=str(val_dir))

    print("=" * 80)
    print("VAANI TRACK 1: CRNN + WAV2VEC2 ENSEMBLE EVALUATION")
    print("=" * 80)
    print(f"CRNN Model Checkpoint     : {args.crnn_model}")
    print(f"Wav2Vec2 Model Checkpoint : {args.w2v_model}")
    print(f"Validation Clips          : {len(dataset)}")
    print(f"Device                    : {device}")
    print("=" * 80)

    # 1. Precompute / Load Frame Probabilities
    cache_dir = PROJECT_ROOT / "data" / "cache" / "ensemble"
    crnn_cache_path = cache_dir / "probs_crnn_9426_gold.pt"
    w2v_cache_path = cache_dir / "probs_wav2vec2_9426_gold.pt"

    crnn_cache = precompute_crnn_probs(dataset, args.crnn_model, device, crnn_cache_path, force_recompute=args.recompute_probs)
    w2v_cache = precompute_wav2vec2_probs(dataset, args.w2v_model, device, w2v_cache_path, force_recompute=args.recompute_probs)

    # Verify overlap of stems
    common_stems = sorted(list(set(crnn_cache.keys()) & set(w2v_cache.keys()) & set(gt_map.keys())))
    print(f"Successfully aligned probabilities for {len(common_stems)} validation clips.")
    assert len(common_stems) == 2501, f"Expected 2,501 validation clips, got {len(common_stems)}"

    # -------------------------------------------------------------------------
    # 2. Evaluate Individual Baselines on identical validation set
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STANDALONE MODEL VALIDATION CALIBRATION")
    print("=" * 80)

    # Standalone CRNN (known best: th=0.40, med=7)
    crnn_preds = {}
    for stem in common_stems:
        c_item = crnn_cache[stem]
        crnn_preds[stem] = probs_to_events(
            c_item["probs"],
            threshold=0.40,
            frame_step_sec=c_item["hop_length"] / c_item["sr"],
            duration=c_item["duration"],
            med=7,
            merge_gap=0.0,
            min_duration=0.0,
        )
    crnn_eval = evaluate_predictions_dict(gt_map, crnn_preds)
    print(f"CRNN Standalone (th=0.40, med=7, hop=512):")
    print(f"  Event F1: {crnn_eval['f1']:.4f} | Segment Dice: {crnn_eval['dice']:.4f} | COMBINED: {crnn_eval['combined']:.4f}")
    print(f"  Natural: F1={crnn_eval['nat_f1']:.4f}, Dice={crnn_eval['nat_dice']:.4f}, Comb={crnn_eval['nat_comb']:.4f}")
    print(f"  Synthetic: F1={crnn_eval['syn_f1']:.4f}, Dice={crnn_eval['syn_dice']:.4f}, Comb={crnn_eval['syn_comb']:.4f}")

    # Standalone Wav2Vec2 (known best: th=0.45, med=5, md=0.15, mg=0.10)
    w2v_preds = {}
    for stem in common_stems:
        w_item = w2v_cache[stem]
        w2v_preds[stem] = probs_to_events(
            w_item["probs"],
            threshold=0.45,
            frame_step_sec=w_item["stride"] / w_item["sr"],
            duration=w_item["duration"],
            med=5,
            merge_gap=0.10,
            min_duration=0.15,
        )
    w2v_eval = evaluate_predictions_dict(gt_map, w2v_preds)
    print(f"\nWav2Vec2 Standalone (th=0.45, med=5, md=0.15, mg=0.10, stride=320):")
    print(f"  Event F1: {w2v_eval['f1']:.4f} | Segment Dice: {w2v_eval['dice']:.4f} | COMBINED: {w2v_eval['combined']:.4f}")
    print(f"  Natural: F1={w2v_eval['nat_f1']:.4f}, Dice={w2v_eval['nat_dice']:.4f}, Comb={w2v_eval['nat_comb']:.4f}")
    print(f"  Synthetic: F1={w2v_eval['syn_f1']:.4f}, Dice={w2v_eval['syn_dice']:.4f}, Comb={w2v_eval['syn_comb']:.4f}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 3. Grid Alignment Exploration & Weight Sweeps
    # We test grid alignments: "10ms" (primary requested), "20ms" (W2V native)
    # Weights for Wav2Vec2: [0.3, 0.4, 0.5, 0.6, 0.7]
    # -------------------------------------------------------------------------
    weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
    medians_10ms = [5, 7, 9]
    medians_20ms = [5, 7]

    print("\n" + "=" * 95)
    print("STAGE 1: WEIGHTED ENSEMBLE SWEEP ON 10 MS GRID (W2V weights 0.3 to 0.7)")
    print("=" * 95)
    print(f"{'Alpha(W2V)':<11}{'Thresh':<8}{'Med':<5}{'MinDur':<8}{'MergeGap':<10}{'Event F1':<11}{'Dice':<11}{'Combined':<11}{'TP':<7}{'FP':<7}{'FN':<7}")
    print("-" * 95)

    all_stage1_results = []
    t_sweep_start = time.time()

    for alpha in weights:
        # Precompute blended probabilities for all clips for this alpha on 10ms grid
        blended_clips_10ms = []
        for stem in common_stems:
            b_probs, step_sec = align_and_blend_probs(
                crnn_cache[stem],
                w2v_cache[stem],
                alpha=alpha,
                grid_mode="10ms",
            )
            blended_clips_10ms.append((stem, b_probs, step_sec, crnn_cache[stem]["duration"]))

        for th in thresholds:
            for med in medians_10ms:
                for md in [0.10, 0.15]:
                    for mg in [0.05, 0.10]:
                        cur_preds = {}
                        for stem, b_probs, step_sec, dur in blended_clips_10ms:
                            cur_preds[stem] = probs_to_events(
                                b_probs,
                                threshold=th,
                                frame_step_sec=step_sec,
                                duration=dur,
                                med=med,
                                merge_gap=mg,
                                min_duration=md,
                            )
                        ev = evaluate_predictions_dict(gt_map, cur_preds)
                        res_entry = {
                            "grid": "10ms",
                            "alpha": alpha,
                            "threshold": th,
                            "med": med,
                            "min_duration": md,
                            "merge_gap": mg,
                            **ev,
                        }
                        all_stage1_results.append(res_entry)

        # Print the best result for this alpha
        alpha_res = [r for r in all_stage1_results if r["alpha"] == alpha]
        alpha_res.sort(key=lambda x: x["combined"], reverse=True)
        top_a = alpha_res[0]
        print(
            f"{top_a['alpha']:<11.2f}{top_a['threshold']:<8.2f}{top_a['med']:<5}{top_a['min_duration']:<8.2f}{top_a['merge_gap']:<10.2f}"
            f"{top_a['f1']:<11.4f}{top_a['dice']:<11.4f}{top_a['combined']:<11.4f}"
            f"{top_a['tp']:<7}{top_a['fp']:<7}{top_a['fn']:<7}"
        )

    print(f"Stage 1 (10ms grid) completed in {time.time() - t_sweep_start:.1f}s.")

    # Also test 20ms grid alignment for comparison
    print("\n" + "=" * 95)
    print("STAGE 1B: WEIGHTED ENSEMBLE SWEEP ON 20 MS GRID (for comparison)")
    print("=" * 95)
    print(f"{'Alpha(W2V)':<11}{'Thresh':<8}{'Med':<5}{'MinDur':<8}{'MergeGap':<10}{'Event F1':<11}{'Dice':<11}{'Combined':<11}{'TP':<7}{'FP':<7}{'FN':<7}")
    print("-" * 95)

    all_stage1b_results = []
    for alpha in [0.4, 0.5, 0.6]:
        blended_clips_20ms = []
        for stem in common_stems:
            b_probs, step_sec = align_and_blend_probs(
                crnn_cache[stem],
                w2v_cache[stem],
                alpha=alpha,
                grid_mode="20ms",
            )
            blended_clips_20ms.append((stem, b_probs, step_sec, crnn_cache[stem]["duration"]))

        for th in [0.40, 0.45, 0.50, 0.55]:
            for med in medians_20ms:
                for md in [0.10, 0.15]:
                    for mg in [0.05, 0.10]:
                        cur_preds = {}
                        for stem, b_probs, step_sec, dur in blended_clips_20ms:
                            cur_preds[stem] = probs_to_events(
                                b_probs,
                                threshold=th,
                                frame_step_sec=step_sec,
                                duration=dur,
                                med=med,
                                merge_gap=mg,
                                min_duration=md,
                            )
                        ev = evaluate_predictions_dict(gt_map, cur_preds)
                        res_entry = {
                            "grid": "20ms",
                            "alpha": alpha,
                            "threshold": th,
                            "med": med,
                            "min_duration": md,
                            "merge_gap": mg,
                            **ev,
                        }
                        all_stage1b_results.append(res_entry)

        alpha_res_b = [r for r in all_stage1b_results if r["alpha"] == alpha]
        alpha_res_b.sort(key=lambda x: x["combined"], reverse=True)
        top_b = alpha_res_b[0]
        print(
            f"{top_b['alpha']:<11.2f}{top_b['threshold']:<8.2f}{top_b['med']:<5}{top_b['min_duration']:<8.2f}{top_b['merge_gap']:<10.2f}"
            f"{top_b['f1']:<11.4f}{top_b['dice']:<11.4f}{top_b['combined']:<11.4f}"
            f"{top_b['tp']:<7}{top_b['fp']:<7}{top_b['fn']:<7}"
        )

    # -------------------------------------------------------------------------
    # 4. Global Ranking across All Ensembles
    # -------------------------------------------------------------------------
    all_results = all_stage1_results + all_stage1b_results
    all_results.sort(key=lambda x: x["combined"], reverse=True)

    print("\n" + "=" * 105)
    print("TOP 15 ENSEMBLE CONFIGURATIONS ACROSS ALL SWEEPS")
    print("=" * 105)
    print(f"{'Rank':<5}{'Grid':<7}{'Alpha':<7}{'Thresh':<8}{'Med':<5}{'MinDur':<8}{'MergeGap':<10}{'Event F1':<11}{'Dice':<11}{'Combined':<11}{'Nat Comb':<10}{'Syn Comb':<10}")
    print("-" * 105)
    for idx, r in enumerate(all_results[:15], 1):
        marker = " <== BEST" if idx == 1 else ""
        print(
            f"{idx:<5}{r['grid']:<7}{r['alpha']:<7.2f}{r['threshold']:<8.2f}{r['med']:<5}{r['min_duration']:<8.2f}{r['merge_gap']:<10.2f}"
            f"{r['f1']:<11.4f}{r['dice']:<11.4f}{r['combined']:<11.4f}{r['nat_comb']:<10.4f}{r['syn_comb']:<10.4f}{marker}"
        )
    print("=" * 105)

    best = all_results[0]

    # -------------------------------------------------------------------------
    # 5. Fine Post-Processing Sweep around the Best Ensemble Configuration
    # -------------------------------------------------------------------------
    print(f"\n--- Running Fine Tuning around best configuration (Grid: {best['grid']}, Alpha: {best['alpha']}) ---")
    b_grid = best["grid"]
    b_alpha = best["alpha"]

    blended_clips_fine = []
    for stem in common_stems:
        b_probs, step_sec = align_and_blend_probs(
            crnn_cache[stem],
            w2v_cache[stem],
            alpha=b_alpha,
            grid_mode=b_grid,
        )
        blended_clips_fine.append((stem, b_probs, step_sec, crnn_cache[stem]["duration"]))

    fine_thresholds = [best["threshold"] - 0.04, best["threshold"] - 0.02, best["threshold"], best["threshold"] + 0.02, best["threshold"] + 0.04]
    fine_medians = [best["med"] - 2, best["med"], best["med"] + 2] if best["med"] > 3 else [3, 5, 7]
    fine_min_durs = [0.08, 0.10, 0.12, 0.15, 0.18]
    fine_merge_gaps = [0.05, 0.08, 0.10, 0.12]

    fine_results = []
    for th in fine_thresholds:
        for med in fine_medians:
            for md in fine_min_durs:
                for mg in fine_merge_gaps:
                    cur_preds = {}
                    for stem, b_probs, step_sec, dur in blended_clips_fine:
                        cur_preds[stem] = probs_to_events(
                            b_probs,
                            threshold=th,
                            frame_step_sec=step_sec,
                            duration=dur,
                            med=med,
                            merge_gap=mg,
                            min_duration=md,
                        )
                    ev = evaluate_predictions_dict(gt_map, cur_preds)
                    fine_results.append({
                        "grid": b_grid,
                        "alpha": b_alpha,
                        "threshold": round(th, 2),
                        "med": med,
                        "min_duration": round(md, 2),
                        "merge_gap": round(mg, 2),
                        **ev,
                    })

    fine_results.sort(key=lambda x: x["combined"], reverse=True)
    if fine_results[0]["combined"] > best["combined"]:
        best = fine_results[0]
        print(f"Fine tuning improved Combined score to: {best['combined']:.4f}")

    # -------------------------------------------------------------------------
    # 6. Generate and Save Predictions for the Best Ensemble
    # -------------------------------------------------------------------------
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating final predictions using best ensemble configuration...")
    final_preds_map = {}
    clips_with_events = 0
    total_events = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for stem in common_stems:
            b_probs, step_sec = align_and_blend_probs(
                crnn_cache[stem],
                w2v_cache[stem],
                alpha=best["alpha"],
                grid_mode=best["grid"],
            )
            evs = probs_to_events(
                b_probs,
                threshold=best["threshold"],
                frame_step_sec=step_sec,
                duration=crnn_cache[stem]["duration"],
                med=best["med"],
                merge_gap=best["merge_gap"],
                min_duration=best["min_duration"],
            )
            final_preds_map[stem] = evs
            if evs:
                clips_with_events += 1
                total_events += len(evs)
            formatted = [{"onset": on, "offset": off} for on, off in evs]
            fout.write(json.dumps({"clip_id": crnn_cache[stem]["clip_id"], "events": formatted}) + "\n")

    print(f"Predictions successfully written to: {out_path}")
    print(f"  Clips processed    : {len(common_stems)}")
    print(f"  Clips with events  : {clips_with_events} / {len(common_stems)} ({clips_with_events / len(common_stems) * 100:.1f}%)")
    print(f"  Total events       : {total_events}")

    # Re-verify written file using eval_local logic
    re_eval = evaluate_predictions_dict(gt_map, final_preds_map)
    assert abs(re_eval["combined"] - best["combined"]) < 1e-4

    # -------------------------------------------------------------------------
    # 7. Final Comprehensive Comparison Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: STANDALONE CRNN vs STANDALONE WAV2VEC2 vs ENSEMBLE")
    print("=" * 80)
    print(f"{'Model':<28}{'Event F1':<12}{'Segment Dice':<14}{'Combined Score':<16}{'vs CRNN':<10}{'vs W2V':<10}")
    print("-" * 80)
    print(f"{'CRNN (9426 Gold)':<28}{crnn_eval['f1']:<12.4f}{crnn_eval['dice']:<14.4f}{crnn_eval['combined']:<16.4f}{'--':<10}{crnn_eval['combined'] - w2v_eval['combined']:<+10.4f}")
    print(f"{'Wav2Vec2 (9426 Gold)':<28}{w2v_eval['f1']:<12.4f}{w2v_eval['dice']:<14.4f}{w2v_eval['combined']:<16.4f}{w2v_eval['combined'] - crnn_eval['combined']:<+10.4f}{'--':<10}")
    print(f"{'Ensemble (CRNN+W2V)':<28}{best['f1']:<12.4f}{best['dice']:<14.4f}{best['combined']:<16.4f}{best['combined'] - crnn_eval['combined']:<+10.4f}{best['combined'] - w2v_eval['combined']:<+10.4f}")
    print("-" * 80)
    print("Subsets Breakdown for Best Ensemble:")
    print(f"  Natural Audio  (1,806 clips) : Event F1 = {best['nat_f1']:.4f}, Segment Dice = {best['nat_dice']:.4f}, Combined = {best['nat_comb']:.4f}")
    print(f"  Synthetic Audio (695 clips)  : Event F1 = {best['syn_f1']:.4f}, Segment Dice = {best['syn_dice']:.4f}, Combined = {best['syn_comb']:.4f}")
    print("-" * 80)
    print("Optimal Ensemble Parameters:")
    print(f"  Grid Alignment        : {best['grid']}")
    print(f"  Wav2Vec2 Weight (alpha): {best['alpha']:.2f} (CRNN weight: {1.0 - best['alpha']:.2f})")
    print(f"  Probability Threshold : {best['threshold']:.2f}")
    print(f"  Median Filter Size    : {best['med']}")
    print(f"  Min Event Duration    : {best['min_duration']:.2f}s")
    print(f"  Merge Gap             : {best['merge_gap']:.2f}s")
    print("=" * 80)

    # Save summary report to JSON
    report_path = PROJECT_ROOT / "reports" / "ensemble_validation_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "crnn_baseline": crnn_eval,
        "wav2vec2_standalone": w2v_eval,
        "best_ensemble": best,
        "top_10_configurations": all_results[:10],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Full ensemble report saved to: {report_path}")


if __name__ == "__main__":
    main()
