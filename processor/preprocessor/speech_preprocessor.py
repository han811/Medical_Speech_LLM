from transformers import AutoFeatureExtractor

from .registry import register
from .base import BasePreProcessor


@register("wavelm")
class WaveLMPreProcessor(BasePreProcessor):
    __model_names: list = [
        "microsoft/wavlm-large",
        "microsoft/wavlm-base",
        "microsoft/wavlm-base-plus",
    ]

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        sampling_rate: int = 16000,
    ):
        super().__init__()

        self.model_name = model_name
        self.sampling_rate = sampling_rate

        self.preprocessor = AutoFeatureExtractor.from_pretrained(self.model_name)

    def __call__(self, waveform):
        return self.preprocessor(
            waveform.squeeze(),
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
        ).input_values
