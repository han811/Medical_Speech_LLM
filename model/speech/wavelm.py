from typing import Optional

import gin
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoFeatureExtractor

from ..utils.lora_config import CustomLoraConfig
from .registry import register
from .base import BaseSpeechEncoder, BaseSpeechPreProcessor, BaseSpeechPostProcessor


@gin.configurable()
@register("wavelm")
class WaveLMEncoder(BaseSpeechEncoder):
    __model_names: list = [
        "microsoft/wavlm-large",
        "microsoft/wavlm-base",
        "microsoft/wavlm-base-plus",
    ]

    def __init__(
        self,
        model_name,
        use_lora: bool = False,
        lora_config: Optional[CustomLoraConfig] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_config = lora_config.get_lora_config()

        self.load_model()

    def load_model(self):
        self.encoder: nn.Module = AutoModel.from_pretrained(self.model_name)

        if self.use_lora:
            if self.lora_config is None:
                raise ValueError(
                    "lora_config should be initialized if you want to use lora"
                )
            self.encoder = get_peft_model(self.encoder, self.lora_config)
            self.encoder.print_trainable_parameters()

        self.preprocessor = WaveLMPreProcessor(model_name=self.model_name)

    def forward(self, x):
        hidden_state = self.encoder(x).last_hidden_state
        return hidden_state

    def card_list(self):
        print(WaveLMEncoder.__model_names)


@gin.configurable()
class WaveLMPreProcessor(BaseSpeechPreProcessor):
    def __init__(
        self,
        model_name: str,
        sampling_rate: int,
    ):
        super().__init__(sampling_rate)

        self.model_name = model_name

        self.preprocessor = AutoFeatureExtractor.from_pretrained(self.model_name)

    def __call__(self, waveform):
        return self.preprocessor(
            waveform.squeeze(),
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
        ).input_values
