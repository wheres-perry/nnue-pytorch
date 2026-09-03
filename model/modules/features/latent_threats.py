import torch
from torch import nn

from .input_feature import InputFeature


class LatentThreats(InputFeature):
    HASH = 0x29A7ACC2
    FEATURE_NAME = "LatentThreats"
    INPUT_FEATURE_NAME = "LatentThreats"
    MAX_ACTIVE_FEATURES = 8

    NUM_INPUTS = 12288
    NUM_REAL_FEATURES = 12288
    EXPORT_WEIGHT_DTYPE = torch.int8

    def __init__(self, num_outputs: int):
        super().__init__()

        self.num_outputs = num_outputs
        self.weight = nn.Parameter(
            torch.empty(self.NUM_INPUTS, num_outputs, dtype=torch.float32)
        )

        self.reset_parameters()

    def merged_weight(self) -> torch.Tensor:
        return self.weight

    @torch.no_grad()
    def coalesce(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        pass  # no virtual weights

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        """Threats have no piece values, so PSQT columns are zero."""
        L1 = self.num_outputs - num_psqt_buckets
        for i in range(num_psqt_buckets):
            self.weight[:, L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        return self.weight.data.clone()

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        self.weight.data.copy_(export_weight)

    def clip_weights(self, quantization) -> None:
        """Clamp threat weights to quantization-safe range."""
        self.weight.data.clamp_(
            quantization.min_threat_weight, quantization.max_threat_weight
        )
