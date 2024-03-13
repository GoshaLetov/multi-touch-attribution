import pandas as pd

from typing import Optional
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from scripts.dataset import AdBannerDataset
from scripts.dataset import collate_fn


MEDIA = {
    'video_mobile_app': 1,
    'banner_mobile_app': 2,
    'video_site': 3,
    'banner_site': 4,
    'banner_marketplace': 5,
    'video_marketplace': 0,
}


def read_events(path: str) -> pd.DataFrame:
    events = pd.read_csv(filepath_or_buffer=path, parse_dates=['tsEvent_datetime'])

    events = events[[
        'tsEvent',
        'hid',
        'event',
        'media',
        'tsEvent_datetime',
    ]].assign(
        media=events.media.map(MEDIA),
    ).dropna(

    ).sort_values(
        by=[
            'tsEvent',
            'hid',
        ],
    )
    return events


def train_valid_test_split(events: pd.DataFrame, train_fraction: float) -> pd.DataFrame:
    hid = events.hid.drop_duplicates().reset_index().set_index('hid').drop(columns='index')

    train_hid = hid.sample(frac=train_fraction, replace=False, random_state=42).index
    valid_hid = hid.drop(train_hid).sample(frac=0.5, replace=False, random_state=42).index
    test_hid = hid.drop(train_hid).drop(valid_hid).index

    events.loc[events.hid.isin(train_hid), 'part'] = 'train'
    events.loc[events.hid.isin(valid_hid), 'part'] = 'valid'
    events.loc[events.hid.isin(test_hid), 'part'] = 'test'

    print(events.part.value_counts(normalize=True))

    return events


class AdBannerDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        train_fraction: float = 0.8,
        train_batch_size: int = 32,
        valid_batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self._path = path
        self._train_fraction = train_fraction
        self._train_batch_size = train_batch_size
        self._valid_batch_size = valid_batch_size
        self._num_workers = num_workers

        self.train_dataset: Optional[AdBannerDataset] = None
        self.valid_dataset: Optional[AdBannerDataset] = None
        self.test_dataset: Optional[AdBannerDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        events = train_valid_test_split(
            events=read_events(path=self._path),
            train_fraction=self._train_fraction,
        )

        self.train_dataset = AdBannerDataset(
            data=events[events.part == 'train'],
            max_seq_length=35,
            min_seq_length=2,
            augmented=False,
        )
        self.valid_dataset = AdBannerDataset(
            data=events[events.part == 'valid'],
            max_seq_length=35,
            min_seq_length=2,
            augmented=False,
        )
        self.test_dataset = AdBannerDataset(
            data=events[events.part == 'test'],
            max_seq_length=35,
            min_seq_length=2,
            augmented=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._train_batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._valid_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._valid_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._valid_batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=self._num_workers,
        )
