from typing import Union
import librosa
import numpy as np
import torch


class LogMelExtractor:
    def __init__(
        self,
        sr: int = 16_000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 512,
        fmin: float = 20.0,
        fmax: float = 8000.0,
        top_db: float = 80.0,
    ) -> None:
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.top_db = top_db

# Core extraction
    def extract(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.FloatTensor:
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        waveform = np.asarray(waveform, dtype=np.float32).squeeze()

        if waveform.ndim != 1:
            raise ValueError(
                f"Expected a 1D waveform, got shape {waveform.shape}"
            )

        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.top_db)
        log_mel = log_mel.T
        return torch.from_numpy(log_mel).float()

# Callable interface (compatible with VaaniNoiseDataset.transform)
    def __call__(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.FloatTensor:
        """Alias for :meth:`extract` so the extractor can be used as a callable."""
        return self.extract(waveform)

    def __repr__(self) -> str:
        return (
            f"LogMelExtractor(sr={self.sr}, n_mels={self.n_mels}, "
            f"n_fft={self.n_fft}, hop_length={self.hop_length}, "
            f"fmin={self.fmin}, fmax={self.fmax})"
        )

# Quick sanity check
if __name__ == "__main__":
    SR = 16_000
    DURATION = 5  

    print("=== LogMelExtractor Test ===\n")

    extractor = LogMelExtractor(sr=SR)
    print(f"Extractor: {extractor}\n")

    # NumPy inp
    np_waveform = np.random.randn(SR * DURATION).astype(np.float32)
    features_np = extractor(np_waveform)
    print(f"NumPy  input  : waveform shape = {np_waveform.shape}")
    print(f"               feature  shape = {features_np.shape}  (dtype={features_np.dtype})")

    # Torch input 
    torch_waveform = torch.randn(SR * DURATION)
    features_torch = extractor(torch_waveform)
    print(f"\nTorch  input  : waveform shape = {tuple(torch_waveform.shape)}")
    print(f"               feature  shape = {tuple(features_torch.shape)}  (dtype={features_torch.dtype})")

    # Basic assertions
    expected_frames = 1 + (SR * DURATION) // extractor.hop_length
    assert features_np.shape[1] == extractor.n_mels, "n_mels mismatch"
    assert features_np.shape[0] == expected_frames, (
        f"Expected {expected_frames} frames, got {features_np.shape[0]}"
    )
    assert features_np.dtype == torch.float32, "dtype should be float32"

    print(f"\n[OK] All checks passed.  Expected frames: {expected_frames}")
