"""
smoke_gpu_advanced.py
Fast GPU smoke test for the Wav2Vec2 + BiGRU model on NVIDIA RTX 4050 (6GB VRAM).

Validates:
1. PyTorch CUDA capability and RTX 4050 detection.
2. Wav2Vec2 model loading into GPU memory.
3. FP16 autocast forward + backward pass on a full batch of 10-second clips.
4. Peak VRAM measurement to confirm safe execution within 6GB limit.
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_model import Wav2Vec2NoiseDetector, WAV2VEC2_STRIDE, WAV2VEC2_SR


def main():
    print("=" * 65)
    print("RTX 4050 GPU SMOKE TEST FOR WAV2VEC2 ADVANCED PIPELINE")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available in current PyTorch build!")
        print(f"PyTorch version: {torch.__version__}")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    print(f"GPU Device  : {gpu_name}")
    print(f"Total VRAM  : {total_mem:.1f} MiB ({total_mem / 1024:.2f} GB)")
    print(f"PyTorch     : {torch.__version__} (CUDA: {torch.version.cuda})")

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()

    # 1. Load model onto GPU
    print("\n[1/4] Initializing Wav2Vec2NoiseDetector on GPU...")
    model = Wav2Vec2NoiseDetector().to("cuda")
    params = model.count_parameters()
    mem_after_model = torch.cuda.memory_allocated(0) / (1024 ** 2)
    print(f"  - Trainable parameters : {params:,}")
    print(f"  - Model VRAM footprint : {mem_after_model:.1f} MiB ({time.time() - t0:.2f}s)")

    # 2. Synthetic batch: 4 clips of 10 seconds each (batch size 4)
    print("\n[2/4] Generating synthetic batch (4 clips x 10 seconds @ 16kHz)...")
    batch_size = 4
    clip_seconds = 10
    waveforms = torch.randn(batch_size, WAV2VEC2_SR * clip_seconds, device="cuda")
    expected_frames = model.frames_for_samples(WAV2VEC2_SR * clip_seconds)
    targets = torch.randint(0, 2, (batch_size, expected_frames), dtype=torch.float32, device="cuda")

    # 3. FP16 Forward + Backward pass
    print("\n[3/4] Running FP16 mixed precision forward + backward pass...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    t_pass = time.time()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=True):
        logits = model(waveforms)
        T = min(logits.shape[1], targets.shape[1])
        loss = criterion(logits[:, :T], targets[:, :T])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    step_duration = time.time() - t_pass
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

    print(f"  - Forward + Backward time : {step_duration:.3f} seconds ({batch_size / step_duration:.1f} clips/s)")
    print(f"  - Loss value              : {loss.item():.4f}")
    print(f"  - Peak VRAM allocated     : {peak_vram:.1f} MiB ({peak_vram / 1024:.2f} GB / {total_mem / 1024:.2f} GB)")

    # 4. Assessment
    print("\n[4/4] VRAM Safety Assessment:")
    free_headroom = (total_mem - peak_vram) / 1024
    if peak_vram < 4000:
        print(f"  [PASS] Peak VRAM is only {peak_vram / 1024:.2f} GB ({free_headroom:.2f} GB headroom remaining).")
        print("  Batch size 4 with FP16 is safe and optimal for training on RTX 4050!")
    else:
        print(f"  [WARNING] High VRAM usage ({peak_vram / 1024:.2f} GB). Consider batch size 2.")

    print("=" * 65)
    print("GPU SMOKE TEST COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
