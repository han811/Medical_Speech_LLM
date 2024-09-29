from typing import Optional

import gin
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel

from .registry import register
from .base import BaseSpeechEncoder


@gin.configurage()
@register("hubert")
class HubertEncoder(BaseSpeechEncoder):
    __model_names: list = [
        "facebook/hubert-xlarge-ll60k",
        "facebook/hubert-large-ll60k",
        "facebook/hubert-base-ls960",
    ]

    def __init__(
        self,
        model_name: str = "facebook/hubert-xlarge-ll60k",
        use_lora: bool = False,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_config = lora_config

        self.encoder = None

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

    def forward(self, x):
        hidden_state = self.encoder(x).last_hidden_state
        return hidden_state

    def card_list(self):
        print(HubertEncoder.__model_names)
