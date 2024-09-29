from typing import Optional

import gin
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .registry import register
from .base import BaseLLM


@gin.configurage()
@register("llama")
class LLAMAModel(BaseLLM):
    __model_names: list = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_lora: bool = False,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_config = lora_config

        self.tokenizer = None
        self.model = None

        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.use_lora:
            if self.lora_config is None:
                raise ValueError(
                    "lora_config should be initialized if you want to use lora"
                )
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()

    def forward(self, x):
        out = self.tokenizer(x)
        out = self.model(out)
        return out

    def card_list(self):
        print(LLAMAModel.__model_names)
