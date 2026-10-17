"""SepFormer Audio Enhancement Pipeline for Track 2.

Uses pre-trained speechbrain/sepformer-dns4-16k-enhancement on GPU.
Ensures exact length preservation, RMS level matching with peak limiting,
and fully resumable batch processing.
"""

import os
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from speechbrain.inference.separation import SepformerSeparation

from track2.audio_io import load_audio, save_audio, verify_audio_file, TARGET_SAMPLE_RATE


class SepFormerEnhancer:
    """Wrapper for SpeechBrain SepFormer DNS4 16kHz speech enhancement model."""

    def __init__(
        self,
        source: str = "speechbrain/sepformer-dns4-16k-enhancement",
        savedir: str = "pretrained_models/sepformer-dns4-16k-enhancement",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading SepFormer model from '{source}' onto {self.device}...")
        self.model = SepformerSeparation.from_hparams(
            source=source,
            savedir=savedir,
            run_opts={"device": self.device},
        )
        self.model.eval()
        print("SepFormer model loaded successfully.")

    def enhance_waveform(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SAMPLE_RATE,
        match_energy: bool = True,
        peak_limit: float = 0.95,
        dry_blend: float = 0.0,
    ) -> np.ndarray:
        """Enhance a single 1D audio waveform.

        Args:
            audio: 1D numpy array of audio samples (float32).
            sample_rate: Expected sample rate (must be 16,000 Hz).
            match_energy: Whether to scale enhanced speech to match input RMS energy.
            peak_limit: Maximum allowed absolute amplitude (e.g. 0.95 to avoid clipping).
            dry_blend: Optional blend fraction of original noisy input (0.0 = full enhancement).

        Returns:
            enhanced_audio: 1D numpy array of enhanced audio with exact same length.
        """
        orig_len = len(audio)
        tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)  # [1, T]

        with torch.no_grad():
            # separate_batch returns [1, T, 1]
            separated = self.model.separate_batch(tensor)
            enhanced = separated[:, :, 0].squeeze(0).cpu().numpy()

        # Enforce exact length preservation
        if len(enhanced) < orig_len:
            enhanced = np.pad(enhanced, (0, orig_len - len(enhanced)), mode="constant")
        elif len(enhanced) > orig_len:
            enhanced = enhanced[:orig_len]

        # Optional dry blend
        if dry_blend > 0.0:
            enhanced = (1.0 - dry_blend) * enhanced + dry_blend * audio

        # Level normalization
        if match_energy:
            rms_orig = np.sqrt(np.mean(audio**2) + 1e-8)
            rms_enh = np.sqrt(np.mean(enhanced**2) + 1e-8)
            scale = rms_orig / rms_enh
            enhanced = enhanced * scale

        # Peak limiting
        peak = np.max(np.abs(enhanced))
        if peak > peak_limit:
            enhanced = enhanced / (peak + 1e-8) * peak_limit

        return enhanced.astype(np.float32)

    def enhance_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        match_energy: bool = True,
        peak_limit: float = 0.95,
        dry_blend: float = 0.0,
    ) -> dict:
        """Enhance a single audio file and save the output.

        Args:
            input_path: Path to input WAV.
            output_path: Path to output enhanced WAV.
            match_energy: Whether to match input RMS energy.
            peak_limit: Peak limit ceiling.
            dry_blend: Dry blend factor.

        Returns:
            dict containing file metrics (duration, frames, elapsed_time, rtf).
        """
        in_path = Path(input_path)
        out_path = Path(output_path)

        audio, sr = load_audio(in_path)
        orig_frames = len(audio)
        duration = orig_frames / sr

        t0 = time.time()
        enhanced = self.enhance_waveform(
            audio,
            sample_rate=sr,
            match_energy=match_energy,
            peak_limit=peak_limit,
            dry_blend=dry_blend,
        )
        t1 = time.time()
        elapsed = t1 - t0
        rtf = elapsed / duration if duration > 0 else 0.0

        save_audio(out_path, enhanced, sample_rate=sr)

        # Verify output
        v = verify_audio_file(out_path, expected_frames=orig_frames, expected_sr=sr)
        if not v["valid"]:
            raise RuntimeError(f"Enhanced audio verification failed for {out_path}: {v['errors']}")

        return {
            "filename": in_path.name,
            "frames": orig_frames,
            "duration": duration,
            "elapsed": elapsed,
            "rtf": rtf,
        }


def run_batch_enhancement(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    enhancer: Optional[SepFormerEnhancer] = None,
    log_interval: int = 50,
) -> dict:
    """Run resumable speech enhancement across all WAV files in input_dir.

    Args:
        input_dir: Path to directory containing input WAV files.
        output_dir: Path to directory where enhanced WAV files will be saved.
        enhancer: Existing SepFormerEnhancer instance, or None to create one.
        log_interval: How often to print progress and ETA.

    Returns:
        Summary dictionary with processing metrics.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list(in_dir.glob("*.wav")))
    total_files = len(wav_files)
    print(f"Found {total_files} total WAV files in {in_dir}")

    if total_files == 0:
        raise ValueError(f"No WAV files found in {in_dir}")

    if enhancer is None:
        enhancer = SepFormerEnhancer()

    # Check for existing completed files for resumption
    pending_files = []
    skipped_count = 0
    for f in wav_files:
        out_f = out_dir / f.name
        if out_f.exists() and out_f.stat().st_size > 44:
            # Check validity
            try:
                info = sf.info(str(out_f))
                if info.samplerate == TARGET_SAMPLE_RATE and info.subtype == "PCM_16":
                    skipped_count += 1
                    continue
            except Exception:
                pass
        pending_files.append(f)

    print(
        f"Resumption check: {skipped_count} files already completed, {len(pending_files)} pending."
    )

    if not pending_files:
        print("All files already completed!")
        return {
            "total_files": total_files,
            "processed_files": 0,
            "skipped_files": skipped_count,
            "total_audio_hours": 0.0,
            "total_runtime_seconds": 0.0,
        }

    total_proc_duration = 0.0
    total_proc_time = 0.0
    start_time = time.time()

    for idx, in_file in enumerate(pending_files, 1):
        out_file = out_dir / in_file.name
        res = enhancer.enhance_file(in_file, out_file)

        total_proc_duration += res["duration"]
        total_proc_time += res["elapsed"]

        if idx % log_interval == 0 or idx == len(pending_files):
            wall_elapsed = time.time() - start_time
            avg_rtf = total_proc_time / total_proc_duration if total_proc_duration > 0 else 0.0
            avg_sec_per_clip = wall_elapsed / idx
            remaining_clips = len(pending_files) - idx
            estimated_remaining_sec = remaining_clips * avg_sec_per_clip
            eta_min = estimated_remaining_sec / 60.0

            print(
                f"[{idx}/{len(pending_files)}] "
                f"Audio processed: {total_proc_duration / 60:.1f}m | "
                f"Wall time: {wall_elapsed / 60:.1f}m | "
                f"Avg RTF: {avg_rtf:.3f}x | "
                f"ETA: {eta_min:.1f} min ({eta_min / 60:.2f} h)",
                flush=True,
            )

    wall_total = time.time() - start_time
    print(
        f"\nBatch Enhancement Finished! Processed {len(pending_files)} clips in {wall_total / 60:.2f} minutes."
    )

    return {
        "total_files": total_files,
        "processed_files": len(pending_files),
        "skipped_files": skipped_count,
        "total_audio_hours": total_proc_duration / 3600.0,
        "total_runtime_minutes": wall_total / 60.0,
        "avg_rtf": total_proc_time / total_proc_duration if total_proc_duration > 0 else 0.0,
    }
