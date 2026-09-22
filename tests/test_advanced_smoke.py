"""
Offline smoke test for the advanced Wav2Vec2 pipeline.

Verifies model + target-building + loss + backprop wire together WITHOUT
needing the HF-gated dataset or an HF_TOKEN. Uses synthetic waveforms and
events shaped exactly like collate_fn_vaani output.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_model import Wav2Vec2NoiseDetector, WAV2VEC2_SR
from src.train_advanced import build_targets, align_length


def main():
    print("=== Advanced pipeline offline smoke test ===\n")

    # Synthetic batch mimicking collate_fn_vaani output: 2 clips, 2s @ 16kHz
    B, samples = 2, WAV2VEC2_SR * 2
    batch = {
        "audio": torch.randn(B, samples),
        "durations": torch.tensor([2.0, 1.5]),
        "events": [
            [{"onset": 0.5, "offset": 1.2, "category": "animal"}],
            [{"onset": 0.0, "offset": 0.4, "category": "vehicle_traffic"}],
        ],
        "clip_ids": ["synthetic_0", "synthetic_1"],
    }

    targets, num_frames = build_targets(batch)
    print(f"Targets shape: {tuple(targets.shape)}  (num_frames={num_frames})")
    assert targets.shape[0] == B
    assert targets.sum() > 0, "expected some positive frames from the events"

    model = Wav2Vec2NoiseDetector()
    model.train()
    logits = model(batch["audio"])
    logits, targets = align_length(logits, targets)
    print(f"Logits shape:  {tuple(logits.shape)}")
    assert logits.shape == targets.shape

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, targets)
    loss.backward()

    grad_ok = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad
    )
    print(f"Loss: {loss.item():.4f}  |  finite grads present: {grad_ok}")
    assert grad_ok, "no finite gradients produced"

    print("\n[OK] Advanced pipeline wires together end-to-end (offline).")


if __name__ == "__main__":
    main()
