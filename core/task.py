import gin
import pytorch_lightning as pl
from torch.optim import Adam

from model import VoiceLLM
from dataset import BaseDataset


@gin.configurable()
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
        self.model = VoiceLLM(connector_name, speech_encoder_name, llm_model_name)
        self.dataset = BaseDataset()  # TODO: Not implemented

        ### learning parameters ###
        self.max_lr = max_lr
        self.warmup_step = warmup_step

    def configure_optimizers(self):
        opt = [
            {"params": self.model.speech_encoder.parameters(), "lr": 1e-5},
            {"params": self.model.connector.parameters(), "lr": self.max_lr},
            {"params": self.model.llm_model.parameters(), "lr": self.max_lr},
        ]
        optimizer = Adam(opt, lr=self.max_lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        (
            speech_features,
            pre_tokenized_ids,
            post_tokenized_ids,
            output_tokenized_ids,
        ) = self.model.preprocess(batch=batch)
        embeds, atts, label_ids = self.model.encode(
            speech_features, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids
        )
        outputs = self.model(embeds, atts, label_ids)
        loss = outputs["loss"]
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(
            mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids
        )
        outputs = self.forward(embeds, atts, label_ids)
        loss = outputs["loss"]
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
