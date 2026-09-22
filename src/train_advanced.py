"""
train_advanced.py
Corrected Advanced fine-tuning pipeline for Vaani Noise Event Detection.

Features:
- VaaniNoiseDataset + collate_fn_vaani (Gold tier, locally cached)
- Wav2Vec2NoiseDetector (pretrained Wav2Vec2 encoder + BiGRU detection head)
- Exact build_targets() frame masking to eliminate zero-padding silence from loss
- Automated / configurable BCEWithLogitsLoss pos_weight based on exact frame positive/negative ratio
- Frozen Wav2Vec2 encoder with dedicated higher LR for BiGRU head (optional differential LR if unfrozen)
- AdamW optimizer with gradient clipping
- Live in-training distribution logging (positive-frame ratio, logit range, sigmoid probabilities)
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time
from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import VaaniNoiseDataset, collate_fn_vaani
from src.advanced_model import Wav2Vec2NoiseDetector, WAV2VEC2_STRIDE, WAV2VEC2_SR

# Windows UTF-8 console output
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Corrected Wav2Vec2 fine-tuning for Vaani Noise Event Detection"
    )
    parser.add_argument("--train-jsonl", type=str, default="data/processed/train_split.jsonl")
    parser.add_argument("--cache-dir", type=str, default="data/cache/audio_clips")
    parser.add_argument("--save-path", type=str, default="models/advanced_wav2vec2_5000_gold_v2.pt",
                        help="Checkpoint destination path (never overwrite baseline/prior runs)")
    parser.add_argument("--pretrained", type=str, default="facebook/wav2vec2-base",
                        help="HuggingFace Wav2Vec2 encoder id")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default 4 fits comfortably in 6GB RTX 4050 with FP16)")
    parser.add_argument("--pos-weight", type=str, default="auto",
                        help="pos_weight for BCEWithLogitsLoss ('auto' calculates exact ratio from data, or float e.g. 2.625)")
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze Wav2Vec2 encoder weights and train only BiGRU + Linear head")
    parser.add_argument("--lr-head", type=float, default=5e-4,
                        help="Learning rate for BiGRU + classifier head (default: 5e-4)")
    parser.add_argument("--lr-encoder", type=float, default=3e-5,
                        help="Learning rate for Wav2Vec2 encoder when unfrozen (default: 3e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max training samples (e.g. 5000 for 5k Gold experiment)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Compute device (auto-detects cuda if available)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="Use mixed precision training (FP16)")
    parser.add_argument("--tier-filter", type=str, nargs="*", default=["gold"],
                        help="Tiers to include (e.g. gold)")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Print training progress every N batches")
    parser.add_argument("--cached-first", action=argparse.BooleanOptionalAction, default=True,
                        help="Prioritize locally cached clips to avoid HF streaming delays")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_targets(
    batch: dict,
    stride: int = WAV2VEC2_STRIDE,
    sr: int = WAV2VEC2_SR,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convert event annotations into per-frame binary targets aligned to the
    Wav2Vec2 output grid (one frame per `stride` input samples).

    Returns:
        targets: FloatTensor [B, num_frames]
        frame_mask: FloatTensor [B, num_frames] (1.0 for valid audio, 0.0 for padding)
        num_frames: int
    """
    audio_padded = batch["audio"]      # [B, max_samples]
    durations = batch["durations"]     # [B]
    events_batch = batch["events"]     # List[List[Dict]]
    clip_ids = batch["clip_ids"]       # List[str]

    batch_size, max_samples = audio_padded.shape
    num_frames = max(1, max_samples // stride)

    targets = torch.zeros(batch_size, num_frames, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, num_frames, dtype=torch.float32)

    for i in range(batch_size):
        dur = float(durations[i].item() if isinstance(durations[i], torch.Tensor) else durations[i])
        valid_frames = min(num_frames, max(1, int(np.floor(dur * sr / stride))))
        frame_mask[i, :valid_frames] = 1.0

        sample_events = events_batch[i]
        if not isinstance(sample_events, list):
            continue
        for ev in sample_events:
            if not isinstance(ev, dict):
                continue
            onset = ev.get("onset")
            offset = ev.get("offset")
            if onset is None or offset is None:
                continue
            try:
                onset = float(onset)
                offset = float(offset)
            except (ValueError, TypeError):
                continue
            if onset < 0 or offset < onset:
                continue
            start_frame = max(0, int(np.floor(onset * sr / stride)))
            end_frame = min(valid_frames, int(np.ceil(offset * sr / stride)))
            if end_frame > start_frame:
                targets[i, start_frame:end_frame] = 1.0
            elif start_frame < valid_frames:
                targets[i, start_frame:start_frame + 1] = 1.0

    return targets, frame_mask, num_frames


def align_length(
    logits: torch.Tensor,
    targets: torch.Tensor,
    frame_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Trim logits, targets, and frame mask to a common frame length."""
    T = min(logits.shape[1], targets.shape[1], frame_mask.shape[1])
    return logits[:, :T], targets[:, :T], frame_mask[:, :T]


def calculate_frame_class_ratio(samples: list, stride: int = WAV2VEC2_STRIDE, sr: int = WAV2VEC2_SR) -> Dict[str, float]:
    """
    Calculate the exact frame-level positive, negative, and pos_weight ratio
    across a list of dataset sample dicts using the exact build_targets() logic.
    """
    total_pos = 0
    total_neg = 0
    total_valid = 0

    for s in samples:
        dur = float(s.get("duration", 0.0))
        if dur <= 0:
            continue
        valid_frames = max(1, int(np.floor(dur * sr / stride)))
        clip_targets = np.zeros(valid_frames, dtype=np.float32)

        for ev in s.get("events", []):
            try:
                onset = float(ev["onset"])
                offset = float(ev["offset"])
            except (KeyError, ValueError, TypeError):
                continue
            if onset < 0 or offset < onset:
                continue
            start_frame = max(0, int(np.floor(onset * sr / stride)))
            end_frame = min(valid_frames, int(np.ceil(offset * sr / stride)))
            if end_frame > start_frame:
                clip_targets[start_frame:end_frame] = 1.0
            elif start_frame < valid_frames:
                clip_targets[start_frame:start_frame + 1] = 1.0

        pos = int(np.sum(clip_targets == 1.0))
        neg = valid_frames - pos
        total_pos += pos
        total_neg += neg
        total_valid += valid_frames

    pos_ratio = total_pos / max(total_valid, 1)
    neg_pos_ratio = total_neg / max(total_pos, 1)

    return {
        "total_valid_frames": total_valid,
        "total_positive_frames": total_pos,
        "total_negative_frames": total_neg,
        "pos_frame_ratio": pos_ratio,
        "neg_to_pos_ratio": neg_pos_ratio,
    }


def train():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("VAANI NOISE EVENT DETECTION - CORRECTED WAV2VEC2 TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device           : {device}")
    print(f"Encoder          : {args.pretrained} (Freeze: {args.freeze_encoder})")
    print(f"Batch Size       : {args.batch_size}")
    print(f"Mixed Precision  : {args.fp16 and device.type == 'cuda'}")
    print(f"Save Path        : {args.save_path}")

    # 1. Dataset loading
    dataset = VaaniNoiseDataset(
        jsonl_path=args.train_jsonl,
        cache_dir=args.cache_dir,
        target_sr=WAV2VEC2_SR,
        tier_filter=args.tier_filter,
        transform=None,
    )

    if args.cached_first:
        cache_path = Path(args.cache_dir)
        if cache_path.exists():
            cached_files = set(os.listdir(cache_path))
            cached, uncached = [], []
            for s in dataset.samples:
                (cached if f"{s['clip_id']}.wav" in cached_files else uncached).append(s)
            dataset.samples = cached + uncached
            if cached:
                print(f"Prioritized {len(cached)} locally cached clips at start of dataset.")

    if args.max_samples is not None and args.max_samples > 0:
        dataset.samples = dataset.samples[:args.max_samples]

    print(f"Dataset samples  : {len(dataset)} ({args.tier_filter})")
    if len(dataset) == 0:
        raise RuntimeError("No training samples found matching criteria.")

    # 2. Calculate exact frame-level positive/negative ratio on training set
    print("\nCalculating exact frame-level positive/negative ratio on training set...")
    stats = calculate_frame_class_ratio(dataset.samples, stride=WAV2VEC2_STRIDE, sr=WAV2VEC2_SR)
    print(f"  - Total Valid Audio Frames : {stats['total_valid_frames']:,}")
    print(f"  - Positive (Noise) Frames  : {stats['total_positive_frames']:,} ({stats['pos_frame_ratio']*100:.2f}%)")
    print(f"  - Negative (Clean) Frames  : {stats['total_negative_frames']:,} ({(1-stats['pos_frame_ratio'])*100:.2f}%)")
    print(f"  - Exact Negative/Positive  : {stats['neg_to_pos_ratio']:.4f}")

    if str(args.pos_weight).strip().lower() == "auto":
        pos_weight_val = stats["neg_to_pos_ratio"]
        print(f"-> Selected pos_weight (Auto from data): {pos_weight_val:.4f}")
    else:
        try:
            pos_weight_val = float(args.pos_weight)
            print(f"-> Selected pos_weight (User override): {pos_weight_val:.4f}")
        except ValueError:
            pos_weight_val = stats["neg_to_pos_ratio"]
            print(f"-> Unrecognized pos_weight, falling back to auto: {pos_weight_val:.4f}")

    # 3. Model setup
    model = Wav2Vec2NoiseDetector(
        pretrained_name=args.pretrained,
        freeze_feature_encoder=True,
    ).to(device)

    # Freezing & Differential Learning Rates
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print(f"Wav2Vec2 encoder frozen: training BiGRU head and classifier ({model.count_parameters():,} trainable params).")
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_head,
            weight_decay=args.weight_decay,
        )
    else:
        print("Wav2Vec2 encoder unfrozen: using differential learning rates.")
        head_params = list(model.gru.parameters()) + list(model.classifier.parameters())
        optimizer = optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": args.lr_encoder},
                {"params": head_params, "lr": args.lr_head},
            ],
            weight_decay=args.weight_decay,
        )

    pos_weight_tensor = torch.tensor([pos_weight_val], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device.type == "cuda"))

    # 4. DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(len(dataset) > args.batch_size),
        collate_fn=collate_fn_vaani,
        num_workers=args.num_workers,
    )

    print("\nStarting Training...")
    print(f"  - Epochs: {args.epochs} | Batches per epoch: {len(train_loader)}")
    print(f"  - Loss: BCEWithLogitsLoss(pos_weight={pos_weight_val:.4f}) with frame-level padding masking")
    print("=" * 70)

    first_batch_logged = False
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        t_epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            waveforms = batch["audio"].to(device)
            targets, frame_mask, _ = build_targets(batch, stride=WAV2VEC2_STRIDE, sr=WAV2VEC2_SR)
            targets = targets.to(device)
            frame_mask = frame_mask.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(args.fp16 and device.type == "cuda")):
                logits = model(waveforms)
                logits, targets, frame_mask = align_length(logits, targets, frame_mask)

                # Masked BCE loss: padded frames do NOT contribute to gradients
                per_frame_loss = criterion(logits, targets)
                loss = (per_frame_loss * frame_mask).sum() / (frame_mask.sum() + 1e-9)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            if not first_batch_logged:
                print(f"First batch dimensions: Audio {list(waveforms.shape)} -> Logits {list(logits.shape)} | Mask Sum: {frame_mask.sum().item():.0f}")
                first_batch_logged = True

            running_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - t_epoch_start
                rate = (batch_idx + 1) / max(elapsed, 1e-3)
                cur_loss = running_loss / num_batches

                # Distribution diagnostics on valid frames only
                valid = frame_mask.bool()
                if valid.any():
                    batch_pos_ratio = (targets[valid] == 1.0).float().mean().item() * 100.0
                    valid_logits = logits[valid].detach()
                    l_min = valid_logits.min().item()
                    l_mean = valid_logits.mean().item()
                    l_max = valid_logits.max().item()
                    valid_probs = torch.sigmoid(valid_logits)
                    p_min = valid_probs.min().item()
                    p_mean = valid_probs.mean().item()
                    p_max = valid_probs.max().item()
                    p_gt_30 = (valid_probs >= 0.30).float().mean().item() * 100.0
                    p_gt_50 = (valid_probs >= 0.50).float().mean().item() * 100.0

                    print(
                        f"Epoch [{epoch + 1}/{args.epochs}] Batch [{batch_idx + 1:4d}/{len(train_loader):4d}] "
                        f"({rate:.1f} b/s) | Loss: {cur_loss:.4f} | Pos: {batch_pos_ratio:.1f}% | "
                        f"Logits: [{l_min:+.2f}, {l_mean:+.2f}, {l_max:+.2f}] | "
                        f"Probs: [{p_min:.2f}, {p_mean:.2f}, {p_max:.2f}] (>=0.30: {p_gt_30:.1f}%, >=0.50: {p_gt_50:.1f}%)",
                        flush=True,
                    )

        epoch_loss = running_loss / max(num_batches, 1)
        print(f"--> Epoch [{epoch + 1}/{args.epochs}] Finished in {time.time() - t_epoch_start:.1f}s - Final Epoch Loss: {epoch_loss:.4f}\n")

    # 5. Save checkpoint
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "pretrained": args.pretrained,
            "stride": WAV2VEC2_STRIDE,
            "sample_rate": WAV2VEC2_SR,
            "epochs": args.epochs,
            "pos_weight": pos_weight_val,
            "freeze_encoder": args.freeze_encoder,
            "lr_head": args.lr_head,
            "lr_encoder": args.lr_encoder,
            "final_loss": epoch_loss,
            "dataset_stats": stats,
            "args": vars(args),
        },
        save_path,
    )
    print("=" * 70)
    print(f"Model checkpoint successfully saved to: {save_path}")
    print("Wav2Vec2 fine-tuning completed.")
    print("=" * 70)


if __name__ == "__main__":
    train()
