"""Causal 1D conv encoder for invariant sequences (streaming)."""

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:

    class CausalConv1d(nn.Module):
        """Causal 1D convolution (no future context)."""

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
            super().__init__()
            self.padding = (kernel_size - 1, 0)
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.nn.functional.pad(x, self.padding)
            return self.conv(x)

    class CausalConv1dEncoder(nn.Module):
        """Stack of causal convs: (B, T, C) -> (B, T, D)."""

        def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            num_layers: int = 2,
            kernel_size: int = 3,
        ):
            super().__init__()
            layers = []
            c_in = in_dim
            for _ in range(num_layers - 1):
                layers.append(CausalConv1d(c_in, hidden_dim, kernel_size))
                layers.append(nn.ReLU())
                c_in = hidden_dim
            layers.append(CausalConv1d(c_in, out_dim, kernel_size))
            self.net = nn.Sequential(*layers)
            self.out_dim = out_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, C) -> (B, C, T)
            x = x.transpose(1, 2)
            out = self.net(x)
            return out.transpose(1, 2)

else:
    CausalConv1d = None
    CausalConv1dEncoder = None
