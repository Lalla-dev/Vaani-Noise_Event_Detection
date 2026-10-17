"""Smoke test for Track 2 SepFormer enhancement on Phase 2 test clips.

Verifies:
1. Audio loading and enhancement on GPU.
2. Exact frame count and duration preservation.
3. 16kHz mono PCM_16 WAV format compliance.
4. Signal integrity (no NaNs, no Infs, peak within limits).
5. Processing speed (RTF) and projected full dataset runtime.
"""

from pathlib import Path
import soundfile as sf
import torch
import numpy as np

from track2.audio_io import load_audio, verify_audio_file
from track2.enhance import SepFormerEnhancer


def run_smoke_test():
    phase2_dir = Path(r"C:\Users\Shubham\Desktop\vaani project\input_data_track-2_phase-2\audio")
    smoke_out_dir = Path(r"C:\Users\Shubham\Desktop\vaani project\Vaani-Noise_Event_Detection\data\smoke_track2")
    smoke_out_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(list(phase2_dir.glob("*.wav")))[:3]
    print("=" * 60)
    print("TRACK 2 PHASE 2 SMOKE TEST: SepFormer Speech Enhancement")
    print("=" * 60)
    print(f"Input Directory:  {phase2_dir}")
    print(f"Output Directory: {smoke_out_dir}")
    print(f"Number of test clips: {len(test_files)}")

    enhancer = SepFormerEnhancer()

    results = []
    for idx, f in enumerate(test_files, 1):
        orig_info = sf.info(str(f))
        out_f = smoke_out_dir / f.name

        print(f"\n--- Clip [{idx}/{len(test_files)}]: {f.name} ---")
        print(f"  Input:  {orig_info.duration:.3f}s | {orig_info.frames} frames | {orig_info.samplerate}Hz | {orig_info.channels}ch | {orig_info.subtype}")

        res = enhancer.enhance_file(f, out_f)
        enh_info = sf.info(str(out_f))

        # Signal checks
        enh_audio, sr = load_audio(out_f)
        peak = np.max(np.abs(enh_audio))
        has_nan = np.isnan(enh_audio).any()
        has_inf = np.isinf(enh_audio).any()
        frames_match = (orig_info.frames == enh_info.frames)
        sr_match = (enh_info.samplerate == 16000)
        ch_match = (enh_info.channels == 1)
        subtype_match = (enh_info.subtype == "PCM_16")

        print(f"  Output: {enh_info.duration:.3f}s | {enh_info.frames} frames | {enh_info.samplerate}Hz | {enh_info.channels}ch | {enh_info.subtype}")
        print(f"  Inference time: {res['elapsed']:.2f}s | RTF: {res['rtf']:.3f}x")
        print(f"  Signal checks: Peak={peak:.3f} | NaN={has_nan} | Inf={has_inf}")
        print(f"  Format checks: FramesMatch={frames_match} | 16kHz={sr_match} | Mono={ch_match} | PCM_16={subtype_match}")

        assert frames_match, f"Frame count mismatch for {f.name}!"
        assert sr_match, f"Sample rate mismatch for {f.name}!"
        assert ch_match, f"Channel count mismatch for {f.name}!"
        assert subtype_match, f"Subtype mismatch for {f.name}!"
        assert not has_nan, f"NaNs detected in enhanced audio for {f.name}!"
        assert not has_inf, f"Infs detected in enhanced audio for {f.name}!"
        assert peak <= 1.0, f"Clipping detected for {f.name}!"

        results.append(res)

    avg_rtf = np.mean([r["rtf"] for r in results])
    total_audio_sec = 3178 * 12.36 # ~10.91 hours
    projected_runtime_sec = total_audio_sec * avg_rtf
    projected_runtime_min = projected_runtime_sec / 60.0

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED SUCCESSFULLY!")
    print(f"Average RTF on RTX 4050: {avg_rtf:.3f}x (processes 1s audio in {avg_rtf:.3f}s)")
    print(f"Projected total runtime for all 3,178 clips (10.9 hours):")
    print(f"  ~{projected_runtime_min:.1f} minutes ({projected_runtime_min / 60.0:.2f} hours)")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
