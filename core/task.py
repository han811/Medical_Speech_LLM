import gin
import pytorch_lightning as pl
from torch.optim import Adam

from model import VoiceLLM


@gin.configurable
class Task(pl.LightningModule):
    def __init__(
        self,
        connector_name: str,
        speech_encoder_name: str,
        llm_model_name: str,
        max_lr: float = 3e-4,
        warmup_step: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        ### key components ###
        self.voice_llm_model = VoiceLLM(
            connector_name, speech_encoder_name, llm_model_name
        )

        ### learning parameters ###
        self.max_lr = max_lr
        self.warmup_step = warmup_step

    def configure_optimizers(self):
        print(self.voice_llm_model.speech_encoder.parameters())
        opt = [
            {"params": self.voice_llm_model.speech_encoder.parameters(), "lr": 1e-5},
            {"params": self.voice_llm_model.connector.parameters(), "lr": self.max_lr},
            {"params": self.voice_llm_model.llm_model.parameters(), "lr": self.max_lr},
        ]
        optimizer = Adam(opt, lr=self.max_lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        (
            speech_features,
            pre_tokenized_ids,
            post_tokenized_ids,
            output_tokenized_ids,
        ) = self.voice_llm_model.preprocess(batch=batch)
        embeds, atts, label_ids = self.voice_llm_model.encode(
            speech_features, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids
        )
        outputs = self.voice_llm_model(embeds, atts, label_ids)
        loss = outputs["loss"]
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        (
            speech_features,
            pre_tokenized_ids,
            post_tokenized_ids,
            output_tokenized_ids,
        ) = self.voice_llm_model.preprocess(batch=batch)
        embeds, atts, label_ids = self.voice_llm_model.encode(
            speech_features, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids
        )
        outputs = self.voice_llm_model(embeds, atts, label_ids)
        loss = outputs["loss"]
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
