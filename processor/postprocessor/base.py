from abc import abstractmethod

from torch import nn


class BasePostProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError
