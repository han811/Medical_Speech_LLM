from abc import abstractmethod

import torch
from torch import nn


class BaseLLM(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented forward")
