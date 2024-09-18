from abc import abstractmethod

import torch
from torch import nn


class BaseSpeechEncoder(nn.Module):
    def __init__(self, finetune: bool = False):
        super().__init__()

        self.finetune = finetune

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented forward")
