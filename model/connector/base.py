from abc import abstractmethod

from torch import nn


class BaseConnector(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented forward")
