from abc import abstractmethod

from torch import nn


class BaseSpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented forward")


class BaseSpeechPreProcessor:
    def __init__(self, sampling_rate: int):
        self.sampling_rate = sampling_rate

    @abstractmethod
    def __call__(self, waveform):
        raise NotImplementedError


class BaseSpeechPostProcessor:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        raise NotImplementedError
