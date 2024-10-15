from abc import abstractmethod

from torch import nn


class BaseVoiceCLF(nn.Module):
    def __init__(self, speech_encoder_name: str):
        super().__init__()

        self.speech_encoder_name = speech_encoder_name

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented forward")

    @abstractmethod
    def encode(self, batch):
        raise NotImplementedError("Not implemented encode")

    @abstractmethod
    def preprocess(self, batch):
        raise NotImplementedError("Not implemented preprocess")


class BaseVoiceLLM(nn.Module):
    def __init__(
        self,
        connector_name: str,
        speech_encoder_name: str,
        llm_model_name: str,
    ):
        super().__init__()

        self.connector_name = connector_name
        self.speech_encoder_name = speech_encoder_name
        self.llm_model_name = llm_model_name

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Not implemented forward")

    @abstractmethod
    def encode(self, batch):
        raise NotImplementedError("Not implemented encode")

    @abstractmethod
    def preprocess(self, batch):
        raise NotImplementedError("Not implemented preprocess")
