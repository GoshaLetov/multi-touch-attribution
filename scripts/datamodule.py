import pandas as pd

from typing import Optional
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from scripts.dataset import AdBannerDataset
from scripts.dataset import collate_fn
from scripts.constants import MEDIA
from scripts.constants import CONTROL


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


def read_controls(path: str) -> pd.DataFrame:
    controls = pd.read_csv(filepath_or_buffer=path).drop(columns=['input_id'])
    controls = controls.set_index('hid').sort_index().reset_index().drop_duplicates(subset=['hid'])

    controls = controls.fillna(9999999).set_index(['hid', 'long_interests']).astype(int).reset_index().set_index('hid')
    controls.long_interests = controls.long_interests.apply(eval).apply(lambda x: [9999999] if x == [] else x)

    for column in CONTROL.keys():
        if column in ['long_interests']:
            controls[column] = controls[column].apply(lambda x: [CONTROL[column][i] for i in x])
        else:
            controls[column] = controls[column].map(CONTROL[column])

    controls = controls.reset_index()
    controls = controls.drop(columns=['long_interests', 'city'])

    return controls


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
        path_controls: str,
        train_fraction: float = 0.8,
        train_batch_size: int = 32,
        valid_batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self._path = path
        self._path_controls = path_controls
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

        controls = read_controls(self._path_controls)

        self.train_dataset = AdBannerDataset(
            events=events[events.part == 'train'],
            controls=controls,
            max_seq_length=35,
            min_seq_length=2,
            augmented=False,
        )
        self.valid_dataset = AdBannerDataset(
            events=events[events.part == 'valid'],
            controls=controls,
            max_seq_length=35,
            min_seq_length=2,
            augmented=False,
        )
        self.test_dataset = AdBannerDataset(
            events=events[events.part == 'test'],
            controls=controls,
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
