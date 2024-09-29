from typing import Optional

import gin
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.lora_config import CustomLoraConfig
from .registry import register
from .base import BaseLLM, BaseLLMPreProcessor, BaseLLMPostProcessor


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
        lora_config: Optional[CustomLoraConfig] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_config = lora_config.get_lora_config()

        self.load_model()

    def load_model(self):
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

        self.preprocessor = LLAMAPreProcessor(model_name=self.model_name)

    def forward(self, x):
        out = self.model(out)
        return out

    def card_list(self):
        print(LLAMAModel.__model_names)


@gin.configurable()
class LLAMAPreProcessor(BaseLLMPreProcessor):
    def __init__(self, model_name: str):
        super().__init__()

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def __call__(self, words):
        return self.tokenizer(words)
