"""Audio I/O utilities for Track 2 Speech Enhancement.

Enforces 16kHz mono PCM_16 WAV format, exact frame preservation,
and audio integrity checks.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import torch


TARGET_SAMPLE_RATE = 16000


def load_audio(
    file_path: Union[str, Path],
    target_sr: int = TARGET_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Load an audio file as float32 1D numpy array at the target sample rate.

    Args:
        file_path: Path to the audio file.
        target_sr: Expected sample rate (default 16,000 Hz).

    Returns:
        audio: 1D float32 numpy array with range [-1.0, 1.0].
        sample_rate: Actual sample rate of the audio.
    """
    path_str = str(file_path)
    audio, sr = sf.read(path_str, dtype="float32")

    if audio.ndim > 1:
        # Convert stereo/multichannel to mono by averaging channels
        audio = np.mean(audio, axis=-1)

    if sr != target_sr:
        raise ValueError(
            f"Audio sample rate mismatch: expected {target_sr} Hz, got {sr} Hz for {path_str}"
        )

    return audio, sr


def save_audio(
    file_path: Union[str, Path],
    audio: Union[np.ndarray, torch.Tensor],
    sample_rate: int = TARGET_SAMPLE_RATE,
    subtype: str = "PCM_16",
) -> None:
    """Save an audio signal as 16-bit PCM WAV.

    Performs safety checks for NaN, Inf, and clipping.

    Args:
        file_path: Output path for the WAV file.
        audio: 1D numpy array or torch Tensor of audio samples.
        sample_rate: Sample rate (must be 16,000 Hz for Track 2).
        subtype: Soundfile subtype, defaults to 'PCM_16'.
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    if audio.ndim > 1:
        audio = np.squeeze(audio)

    # Sanity checks
    if np.isnan(audio).any():
        raise ValueError(f"NaN values detected in audio tensor to be saved to {file_path}")
    if np.isinf(audio).any():
        raise ValueError(f"Inf values detected in audio tensor to be saved to {file_path}")

    # Ensure peak <= 1.0 to avoid PCM_16 wrap-around or hard clipping distortion
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / (peak + 1e-8) * 0.95

    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, sample_rate, subtype=subtype, format="WAV")


def verify_audio_file(
    file_path: Union[str, Path],
    expected_frames: int = None,
    expected_sr: int = TARGET_SAMPLE_RATE,
) -> dict:
    """Verify that a saved audio file is valid and meets competition specifications.

    Args:
        file_path: Path to the WAV file.
        expected_frames: If provided, checks that frames match exactly.
        expected_sr: Expected sample rate (16,000 Hz).

    Returns:
        dict with verification results.
    """
    path_str = str(file_path)
    info = sf.info(path_str)

    valid = True
    errors = []

    if info.samplerate != expected_sr:
        valid = False
        errors.append(f"Sample rate is {info.samplerate}, expected {expected_sr}")

    if info.channels != 1:
        valid = False
        errors.append(f"Channels count is {info.channels}, expected 1 (mono)")

    if info.subtype != "PCM_16":
        valid = False
        errors.append(f"Subtype is {info.subtype}, expected PCM_16")

    if expected_frames is not None and info.frames != expected_frames:
        valid = False
        errors.append(
            f"Frame count mismatch: got {info.frames}, expected {expected_frames} (diff: {info.frames - expected_frames})"
        )

    return {
        "valid": valid,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
        "subtype": info.subtype,
        "errors": errors,
    }
