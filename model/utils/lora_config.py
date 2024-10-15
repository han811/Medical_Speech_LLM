import gin
from peft import LoraConfig


@gin.configurable
class CustomLoraConfig:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        target_modules: str = "all-linear",
        lora_dropout: float = 0.05,
        task_type: str = "CAUSAL_LM",
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
        self.task_type = task_type

    def get_lora_config(self):
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            task_type=self.task_type,
        )
