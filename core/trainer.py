import gin
import pytorch_lightning as pl

import model


@gin.configurable()
class TrainerLightning(pl.LightningModule):
    def __init__(
        self,
        model: model.BaseVoiceLLM,
        max_lr: float = 3e-4,
        total_training_step: int = 50000,
        warmup_step: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.max_lr = max_lr
        self.total_training_step = total_training_step
        self.warmup_step = warmup_step

    def training_step(self, batch, batch_idx):
        mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids = batch
        embeds, atts, label_ids = self.encode(
            mel, pre_tokenized_ids, post_tokenized_ids, output_tokenized_ids
        )
        outputs = self.forward(embeds, atts, label_ids)
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
