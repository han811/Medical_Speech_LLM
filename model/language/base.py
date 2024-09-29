from abc import abstractmethod

from torch import nn


class BaseLLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = None
        self.preprocessor: BaseLLMPreProcessor = None
        self.postprocessor: BaseLLMPostProcessor = None

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented forward")


class BaseLLMPreProcessor:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, words):
        raise NotImplementedError


class BaseLLMPostProcessor:
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, words):
        raise NotImplementedError
