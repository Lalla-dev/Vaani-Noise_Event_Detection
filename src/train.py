"""
train.py
Baseline training pipeline for Vaani Noise Event Detection.

Connects:
- VaaniNoiseDataset (filtered to Gold + Silver) + collate_fn_vaani
- LogMelExtractor (16kHz, 64 mels, hop_length 512)
- CRNNNoiseDetector (bidirectional CRNN)
- BCEWithLogitsLoss + Adam optimizer
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import VaaniNoiseDataset, collate_fn_vaani
from src.features import LogMelExtractor
from src.model import CRNNNoiseDetector

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


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline CRNN Training Pipeline for Vaani Noise Event Detection")
    parser.add_argument("--train-jsonl", type=str, default="data/processed/train_split.jsonl",
                        help="Path to training jsonl split")
    parser.add_argument("--cache-dir", type=str, default="data/cache/audio_clips",
                        help="Path to local audio cache directory")
    parser.add_argument("--save-path", type=str, default="models/baseline_crnn.pt",
                        help="Destination path for trained model checkpoint")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training DataLoader")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max number of training samples to load (default: 100 for fast baseline iteration)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device ('cpu' or 'cuda')")
    parser.add_argument("--cached-first", action=argparse.BooleanOptionalAction, default=True,
                        help="Prioritize locally cached audio clips to eliminate HF streaming delays (default: True)")
    parser.add_argument("--tier-filter", type=str, nargs="*", default=["gold", "silver"],
                        help="Tiers to include (e.g. gold silver)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (default: 0 for stable CPU execution)")
    return parser.parse_args()


def process_batch(
    batch: dict,
    extractor: LogMelExtractor,
    sr: int = 16000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process raw waveform batch from collate_fn_vaani into padded log-mel features
    and aligned binary frame-level targets.

    Returns:
        padded_features: FloatTensor of shape [B, max_T, 64]
        padded_targets:  FloatTensor of shape [B, max_T]
    """
    audio_padded = batch["audio"]       # [B, max_audio_len]
    durations = batch["durations"]      # [B]
    events_batch = batch["events"]      # List[List[Dict]]
    clip_ids = batch["clip_ids"]        # List[str]
    batch_size = audio_padded.shape[0]

    feature_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []

    for i in range(batch_size):
        dur = float(durations[i].item()) if hasattr(durations[i], "item") else float(durations[i])
        clip_id = clip_ids[i]

        # Extract actual clip waveform up to its duration
        expected_len = int(round(dur * sr)) if dur > 0 else audio_padded.shape[1]
        expected_len = min(expected_len, audio_padded.shape[1])
        wav = audio_padded[i, :expected_len]

        # Waveform validity check
        if len(wav) == 0:
            print(f"[Warning] Empty waveform for clip {clip_id}, substituting silent frame.")
            wav = torch.zeros(max(int(sr * 1.0), 16000), dtype=torch.float32)
        elif torch.isnan(wav).any() or torch.isinf(wav).any():
            print(f"[Warning] NaN/Inf in waveform for clip {clip_id}, replacing with zeros.")
            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

        # Extract log-mel features [T_i, 64]
        mel = extractor.extract(wav)

        # Feature tensor shape validation check
        if mel.ndim != 2 or mel.shape[1] != extractor.n_mels:
            raise ValueError(
                f"Feature tensor must have shape [T, {extractor.n_mels}], got {mel.shape} for clip {clip_id}"
            )

        T_i = mel.shape[0]

        # Generate binary frame-level targets for this clip
        target = torch.zeros(T_i, dtype=torch.float32)
        sample_events = events_batch[i]

        if isinstance(sample_events, list):
            for ev in sample_events:
                if not isinstance(ev, dict):
                    print(f"[Warning] Malformed event (not dict) in clip {clip_id}: {ev}; skipping.")
                    continue
                onset = ev.get("onset")
                offset = ev.get("offset")
                if onset is None or offset is None:
                    print(f"[Warning] Malformed event (missing onset/offset) in clip {clip_id}: {ev}; skipping.")
                    continue
                try:
                    onset = float(onset)
                    offset = float(offset)
                except (ValueError, TypeError):
                    print(f"[Warning] Malformed event timestamps in clip {clip_id}: {ev}; skipping.")
                    continue

                if onset < 0 or offset < onset:
                    print(f"[Warning] Invalid event interval [{onset}, {offset}] in clip {clip_id}; skipping.")
                    continue

                # Align time to frames: frame t spans [t * hop / sr, (t + 1) * hop / sr]
                start_frame = max(0, int(np.floor(onset * sr / extractor.hop_length)))
                end_frame = min(T_i, int(np.ceil(offset * sr / extractor.hop_length)))

                if end_frame > start_frame:
                    target[start_frame:end_frame] = 1.0
                elif offset > onset and start_frame < T_i:
                    target[start_frame:min(T_i, start_frame + 1)] = 1.0
        else:
            print(f"[Warning] Unexpected events format in clip {clip_id}: {type(sample_events)}")

        feature_list.append(mel)
        target_list.append(target)

    # Dynamic batch padding to max_T
    max_T = max(feat.shape[0] for feat in feature_list)
    padded_features = torch.zeros(batch_size, max_T, extractor.n_mels, dtype=torch.float32)
    padded_targets = torch.zeros(batch_size, max_T, dtype=torch.float32)

    for i in range(batch_size):
        T_curr = feature_list[i].shape[0]
        padded_features[i, :T_curr, :] = feature_list[i]
        padded_targets[i, :T_curr] = target_list[i]

    return padded_features, padded_targets


def train():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 60)
    print("VAANI NOISE EVENT DETECTION - BASELINE TRAINING")
    print("=" * 60)
    print(f"Device: {device}")

    # 1. Load Dataset (transform=None)
    dataset = VaaniNoiseDataset(
        jsonl_path=args.train_jsonl,
        cache_dir=args.cache_dir,
        target_sr=16000,
        tier_filter=args.tier_filter,
        transform=None,
    )

    # Optional: prioritize cached clips to avoid streaming latency during smoke tests
    if args.cached_first:
        cache_path = Path(args.cache_dir)
        if cache_path.exists():
            cached_files = set(os.listdir(cache_path))
            cached_samples = []
            uncached_samples = []
            for s in dataset.samples:
                fname = f"{s['clip_id']}.wav"
                if fname in cached_files:
                    cached_samples.append(s)
                else:
                    uncached_samples.append(s)
            dataset.samples = cached_samples + uncached_samples
            if cached_samples:
                print(f"Prioritized {len(cached_samples)} locally cached clips at start of dataset.")

    # Apply max-samples limit
    if args.max_samples is not None and args.max_samples > 0:
        dataset.samples = dataset.samples[:args.max_samples]

    print(f"Number of training samples: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")

    if len(dataset) == 0:
        raise RuntimeError("No training samples found matching the criteria.")

    # 2. DataLoader with collate_fn_vaani
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(len(dataset) > args.batch_size),
        collate_fn=collate_fn_vaani,
        num_workers=args.num_workers,
    )

    # 3. LogMelExtractor
    extractor = LogMelExtractor(
        sr=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        fmin=20.0,
        fmax=8000.0,
    )

    # 4. Model, Loss, Optimizer
    model = CRNNNoiseDetector(n_mels=64).to(device)
    print(f"Model parameter count: {model.count_parameters():,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 5. Training Loop
    first_batch_logged = False

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            features, targets = process_batch(batch, extractor, sr=16000)
            features = features.to(device)
            targets = targets.to(device)

            if not first_batch_logged:
                print(f"Feature shape (batch): {list(features.shape)}")
                first_batch_logged = True

            optimizer.zero_grad()
            logits = model(features)

            # Target length matches model output length validation check
            assert targets.shape == logits.shape, (
                f"Target shape {targets.shape} does not match model output shape {logits.shape}"
            )

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Loss: {epoch_loss:.4f}")

    # 6. Save Model Checkpoint
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "final_loss": epoch_loss,
            "args": vars(args),
        },
        save_path,
    )
    print(f"Model checkpoint saved successfully to {save_path}")
    print("Baseline training completed successfully.")


if __name__ == "__main__":
    train()
