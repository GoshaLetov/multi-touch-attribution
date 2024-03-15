from omegaconf import OmegaConf
from pydantic import BaseModel


class TrainConfig(BaseModel):
    monitor_metric: str
    monitor_mode: str
    accelerator: str
    device: int
    seed: int
    n_epochs: int
    lr: float


class ModelConfig(BaseModel):
    backbone: str
    num_embeddings: int
    embedding_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    non_linearity: str
    time_decay: float
    controls: bool


class DataConfig(BaseModel):
    path: str
    path_controls: str
    num_workers: int
    train_fraction: float
    train_batch_size: int
    valid_batch_size: int


class Config(BaseModel):
    experiment_name: str
    train: TrainConfig
    model: ModelConfig
    data: DataConfig

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
