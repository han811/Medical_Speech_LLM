import gin
from torch import nn

from .registry import register
from .base import BaseConnector


@gin.configurable()
@register("linear")
class LinearConnector(BaseConnector):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        x = self.layer(x)
        return x
