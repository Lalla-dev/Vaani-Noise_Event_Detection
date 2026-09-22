"""
advanced_model.py
Advanced model for Vaani Noise Event Detection.

Instead of hand-crafted log-mel features + a small CRNN (the baseline), this
fine-tunes a pretrained Wav2Vec2 encoder (HuggingFace transformers) into a
frame-level noise-event detector.

Pipeline:
    raw waveform [B, samples] (16kHz)
        -> Wav2Vec2 encoder -> frame embeddings [B, T, H]
        -> detection head    -> per-frame logits  [B, T]

Wav2Vec2's convolutional feature encoder downsamples 16kHz audio to ~50
frames/sec (a 320-sample stride, i.e. 20ms per frame). We expose that stride
so training targets can be aligned to the encoder's frame grid the same way
the baseline aligns to its hop_length.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


# Wav2Vec2 (base, 16kHz) total downsampling factor: product of conv strides
# [5,2,2,2,2,2,2] = 320 samples per output frame -> 50 fps at 16kHz.
WAV2VEC2_STRIDE = 320
WAV2VEC2_SR = 16000


class Wav2Vec2NoiseDetector(nn.Module):
    """
    Fine-tunable Wav2Vec2 encoder with a lightweight frame-level detection head.

    Args:
        pretrained_name: HuggingFace model id for the Wav2Vec2 encoder.
        head_hidden: hidden size of the BiGRU detection head.
        dropout: dropout applied before the classifier.
        freeze_feature_encoder: freeze the CNN feature encoder (standard and
            recommended for Wav2Vec2 fine-tuning; keeps low-level features
            stable and cuts memory/compute).
    """

    def __init__(
        self,
        pretrained_name: str = "facebook/wav2vec2-base",
        head_hidden: int = 128,
        dropout: float = 0.1,
        freeze_feature_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_name)
        if freeze_feature_encoder:
            self.encoder.freeze_feature_encoder()

        enc_hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=enc_hidden,
            hidden_size=head_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(head_hidden * 2, 1)

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            waveforms: FloatTensor [B, num_samples] at 16kHz.
            attention_mask: optional LongTensor [B, num_samples] marking valid
                (1) vs padded (0) samples.

        Returns:
            logits: FloatTensor [B, T] of per-frame event logits.
        """
        outputs = self.encoder(waveforms, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state          # [B, T, H]
        hidden = self.dropout(hidden)
        hidden, _ = self.gru(hidden)                # [B, T, 2*head_hidden]
        logits = self.classifier(hidden).squeeze(-1)  # [B, T]
        return logits

    def frames_for_samples(self, num_samples: int) -> int:
        """Number of encoder output frames produced for a given sample count."""
        return self.encoder._get_feat_extract_output_lengths(
            torch.tensor(num_samples)
        ).item()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== Wav2Vec2NoiseDetector Test ===\n")
    model = Wav2Vec2NoiseDetector()
    print(f"Trainable parameters: {model.count_parameters():,}\n")

    dummy = torch.randn(2, WAV2VEC2_SR)  # 2 clips, 1 second each
    model.eval()
    with torch.no_grad():
        logits = model(dummy)
    expected_T = model.frames_for_samples(WAV2VEC2_SR)
    print(f"Input  shape: {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(logits.shape)}  (expected T ~= {expected_T})")
    assert logits.shape[0] == 2 and logits.ndim == 2
    print("\n[OK] Forward pass works.")
