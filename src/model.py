import torch
import torch.nn as nn
from typing import Optional


class CRNNNoiseDetector(nn.Module):

    def __init__(
        self,
        n_mels: int = 64,
        cnn_channels: tuple = (16, 32),
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels

        ch1, ch2 = cnn_channels

        self.cnn = nn.Sequential(

            nn.Conv2d(1, ch1, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  
            nn.Conv2d(ch1, ch2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(ch2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )

        cnn_freq_out = n_mels // 4
        lstm_input_size = ch2 * cnn_freq_out
        self.dropout = nn.Dropout(dropout)

        # RNN: small bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )


        self.head = nn.Linear(lstm_hidden * 2, 1)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        B, T, F = x.shape
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, -1)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.head(x).squeeze(-1)

        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



# Quick sanity check
if __name__ == "__main__":
    print("=== CRNNNoiseDetector Test ===\n")

    model = CRNNNoiseDetector()
    print(model)
    print(f"\nTrainable parameters: {model.count_parameters():,}\n")


    dummy_input = torch.randn(2, 157, 64)
    print(f"Input  shape: {tuple(dummy_input.shape)}")

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Output shape: {tuple(logits.shape)}")

    # Verify
    assert logits.shape == (2, 157), (
        f"Expected output shape (2, 157), got {tuple(logits.shape)}"
    )
    assert logits.dtype == torch.float32, "dtype should be float32"

    print(f"\n[OK] All checks passed.  Output: [B={2}, T={157}]")
