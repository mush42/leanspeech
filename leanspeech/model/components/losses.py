import torch
import torch.nn as nn
from torch.nn import functional as F


class DurationPredictorLoss(torch.nn.Module):
    """
    Loss function module for duration predictor.
    The loss value is Calculated in log domain to make it Gaussian.
    """

    def __init__(self, offset=1.0, reduction="none"):
        """
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.
        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets, nonpadding):
        """
        Args:
            outputs (Tensor): Batch of prediction durations in log domain (B, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (B, T)
        Returns:
            Tensor: Mean squared error loss value.
        Note:
            `outputs` is in log domain but `targets` is in linear domain.
        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets.float())
        loss = (loss * nonpadding).sum() / nonpadding.sum()
        return loss


class STFTMagnitudeLoss(nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Log-magnitudes are calculated with `log(log_fac*x + log_eps)`, where `log_fac` controls the
    compression strength (larger value results in more compression), and `log_eps` can be used
    to control the range of the compressed output values (e.g., `log_eps>=1` ensures positive
    output values). The default values `log_fac=1` and `log_eps=0` correspond to plain log-compression.

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        log_eps (float, optional): Constant value added to the magnitudes before evaluating the logarithm.
            Default: 0.0
        log_fac (float, optional): Constant multiplication factor for the magnitudes before evaluating the logarithm.
            Default: 1.0
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(self, log=True, log_eps=0.0, log_fac=1.0, distance="L1", reduction="mean"):
        super(STFTMagnitudeLoss, self).__init__()

        self.log = log
        self.log_eps = log_eps
        self.log_fac = log_fac

        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag, y_mag):
        if self.log:
            x_mag = torch.log(self.log_fac * x_mag + self.log_eps)
            y_mag = torch.log(self.log_fac * y_mag + self.log_eps)
        return self.distance(x_mag, y_mag)

