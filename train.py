import argparse
import logging

from clearml import Task

from torch import set_float32_matmul_precision

from scripts.constants import CONTROL
from scripts.model import LSTMAttention
from scripts.datamodule import AdBannerDataModule
from scripts.lightningmodule import AdBannerLightningModule
from scripts.config import Config

from lightning.pytorch import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor

from lightning.pytorch.loggers import TensorBoardLogger


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def main():
    args = arg_parse()
    config = Config.from_yaml(path=args.config_file)
    logging.basicConfig(level=logging.INFO)
    seed_everything(seed=config.train.seed, workers=True)

    set_float32_matmul_precision('high')

    Task.set_credentials(
        api_host='https://api.clear.ml',
        web_host='https://app.clear.ml',
        files_host='https://files.clear.ml',
        key='28Y1LI2GEA9JB1KIFEZT',
        secret='yyAurXE95FyDkxKnWyBBh2GDBtQlaTsjHvUg20DUWyAYFT6qpY'
    )
    task = Task.init(
        project_name='Deep Neural Net with Attention for Multi-channel Multi-touch Attribution',
        task_name='multi-touch-attribution',
        auto_connect_frameworks=True,
    )
    task.connect(config.model_dump())

    datamodule = AdBannerDataModule(
        path=config.data.path,
        path_controls=config.data.path_controls,
        train_fraction=config.data.train_fraction,
        train_batch_size=config.data.train_batch_size,
        valid_batch_size=config.data.valid_batch_size,
        num_workers=config.data.num_workers,
    )
    model = LSTMAttention(
        mapping=CONTROL if config.model.controls else None,
        attention_non_linearity=config.model.non_linearity,
        time_decay=config.model.time_decay,
        num_embeddings=config.model.num_embeddings,
        embedding_dim=config.model.embedding_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    )
    module = AdBannerLightningModule(
        model=model,
        lr=config.train.lr,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        monitor=config.train.monitor_metric,
        mode=config.train.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{config.train.monitor_metric}_{{{config.train.monitor_metric}:.3f}}',
    )
    trainer = Trainer(
        max_epochs=config.train.n_epochs,
        accelerator=config.train.accelerator,
        log_every_n_steps=1,
        logger=TensorBoardLogger(save_dir='.', name='logs'),
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.train.monitor_metric, patience=3, mode='max'),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        deterministic=True,
    )
    trainer.fit(module, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)


if __name__ == '__main__':
    main()
