from torch import nn

from transformers import AutoModel

from .registry import register
from .base import BaseSpeechEncoder


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
        finetune: bool = False,
    ):
        super().__init__(finetune)

        self.model_name = model_name
        self.encoder = None

        self.load_model()

    def load_model(self):
        self.encoder: nn.Module = AutoModel.from_pretrained(self.model_name)
        for param in self.encoder.parameters():
            param.requires_grad = self.finetune

    def forward(self, x):
        return self.encoder(x).last_hidden_state

    def card_list(self):
        print(HubertEncoder.__model_names)
