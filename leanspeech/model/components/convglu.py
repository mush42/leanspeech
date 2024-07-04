import torch
import torch.nn as nn

from .conv_sep import ConvSeparable


class ConvGLU(nn.Module):
    """Dropout - Conv1d - GLU and optional batch normalization.
    """
    def __init__(self,
                 channels: int,
                 kernel_size: 7,
                 batchnorm: bool=False,
                 dropout: float=0.0
             ):
        """
        Args:
            channels: size of the input channels.
            kernel_size: size of the convolutional kernels.
            batchnorm: use batch normalization
            dropout: dropout rate.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            ConvSeparable(channels, channels * 2, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels * 2),
            nn.GLU(dim=1)
        )

    def forward(self, inputs: torch.Tensor):
        """Transform the inputs with given conditions.
        Args:
            inputs: [torch.float32; [B, channels, T]], input channels.
        Returns:
            [torch.float32; [B, channels, T]], transformed.
        """
        # [B, channels, T]
        return inputs + self.conv(inputs)


class PreConv(nn.Module):
    def __init__(self, c_in, c_mid, c_out):
        super(PreConv, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(c_in, c_mid, kernel_size=1, dilation=1),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv1d(c_mid, c_mid, kernel_size=1, dilation=1),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Conv1d(c_mid, c_out, kernel_size=1, dilation=1),
        )

    def forward(self, x):
        y = self.network(x)
        return y
