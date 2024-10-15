import gin
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy

from .task import Task
import dataset


@gin.configurable
class TrainerWrapper:
    def __init__(
        self,
        task: Task,
        log_path: str,
        exp_name: str,
        dataset_name: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        max_epochs: int = 50000,
        device: str = "cpu",
        gpus: int = 2,
        limit_train_batches: int = 10000,
        limit_val_batches: int = 10000,
        log_every_n_steps: int = 10000,
        grad_accumulate_steps: int = 8,
    ):
        self.task = task
        self.dataset_factory: dataset.DatasetFactory = dataset.make(dataset_name)()
        self.train_dataset = self.dataset_factory.get_dataset()
        self.val_dataset = self.dataset_factory.get_dataset(is_train=False)
        speech_preprocessor, llm_preprocessor = (
            self.task.voice_llm_model.get_preprocesor()
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.dataset_factory.get_collator_fn(
                speech_preprocessor=speech_preprocessor,
                llm_preprocessor=llm_preprocessor,
                device=device,
            ),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.dataset_factory.get_collator_fn(
                speech_preprocessor=speech_preprocessor,
                llm_preprocessor=llm_preprocessor,
                device=device,
            ),
        )

        logger = CSVLogger(log_path, name=exp_name)
        # logger = WandbLogger(project="mmllm", name=log_path)

        self.trainer = Trainer(
            accelerator="gpu",
            devices=gpus,
            max_epochs=max_epochs,
            strategy=DDPStrategy(find_unused_parameters=True),
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            log_every_n_steps=log_every_n_steps,
            enable_checkpointing=True,
            callbacks=[
                ModelCheckpoint(
                    dirpath=f"checkpoints",
                    filename=log_path + "-{epoch}",
                    save_top_k=1,
                    monitor="val/loss",
                    save_last=True,
                )
            ],
            fast_dev_run=False,
            logger=logger,
            accumulate_grad_batches=grad_accumulate_steps,
            resume_from_checkpoint=None,
        )

    def train(self):
        self.trainer.fit(self.task, self.train_loader, self.val_loader)
