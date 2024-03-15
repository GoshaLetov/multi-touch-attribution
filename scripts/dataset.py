import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scripts.constants import CONTROL


def collate_fn(batch) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    sequence, controls, targets = zip(*batch)

    sequence_out = {}
    for item in sequence[0].keys():
        sequence_out[item] = [obs[item] for obs in sequence]

        if item in ['sequence', 'time']:
            sequence_out[item] = pad_sequence(
                sequences=sequence_out[item],
                batch_first=True,
                padding_value=0,
            )
        else:
            sequence_out[item] = torch.stack(sequence_out[item])

    controls_out = {}
    for item in controls[0].keys():
        controls_out[item] = [obs[item] for obs in controls]

        if item in ['long_interests']:
            controls_out[item] = pad_sequence(
                sequences=controls_out[item],
                batch_first=True,
                padding_value=0,
            )
        else:
            controls_out[item] = torch.stack(controls_out[item])

    targets_out = torch.stack(targets)

    return sequence_out, controls_out, targets_out


class AdBannerDataset(Dataset):
    def __init__(self,
                 events: pd.DataFrame,
                 controls: pd.DataFrame,
                 max_seq_length: int = 10,
                 min_seq_length: int = 3,
                 augmented: bool = False):
        self.events = events
        self.controls = controls
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.augmented = augmented
        self.sequences = self._generate_sequences()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        hid, sequence_idx, target = self.sequences[idx]

        target = torch.tensor(target)

        sequence = self._get_sequence(hid, sequence_idx)
        controls = self._get_controls(hid)

        return sequence, controls, target

    def _get_sequence(self, hid: str, sequence_idx: list[int]) -> dict[str, torch.Tensor]:
        sequence = self.events[self.events['hid'] == hid]

        max_time = sequence.iloc[max(sequence_idx) + 1].tsEvent_datetime
        sequence = sequence.iloc[sequence_idx]

        sequence = {
            'sequence': torch.tensor(sequence.media.tolist()).long(),
            'lengths': torch.tensor(len(sequence.media.tolist())).long(),
            'time': torch.from_numpy(np.log1p((max_time - sequence.tsEvent_datetime).dt.seconds.to_numpy() / 60 / 60)),
        }

        return sequence

    def _get_controls(self, hid: str) -> dict[str, torch.Tensor]:
        controls = self.controls[self.controls['hid'] == hid].drop(columns='hid')

        if controls.empty:
            return {key: torch.tensor(CONTROL[key][9999999]) for key in controls.columns}

        controls = controls.to_dict(orient='records')[0]

        for key, value in controls.items():
            controls[key] = torch.tensor(value)

        return controls

    def _generate_sequences(self) -> list[tuple[str, list[int], int]]:
        sequences = []
        for hid, group in self.events.groupby('hid'):

            hid: str
            group: pd.DataFrame

            events = group.event.tolist()

            if len(events) < self.min_seq_length:
                continue

            if self.augmented:
                for start in range(len(events)):
                    for end in range(
                            start + self.min_seq_length,
                            min(start + self.max_seq_length + 1, len(events) + 1) - 1,
                    ):
                        sequence_idx = list(range(start, end))
                        target = int(events[end] == 'click')
                        sequences.append((hid, sequence_idx, target))
            else:
                clicks_indexes = [i for i, event in enumerate(events) if event == 'click']

                if len(clicks_indexes) > 0:
                    for end in clicks_indexes:
                        if end > self.min_seq_length:
                            # TODO: add click media
                            sequence_idx = list(range(
                                max(0, end - self.max_seq_length),
                                end, # + 1
                            ))
                            target = 1
                            sequences.append((hid, sequence_idx, target))
                else:
                    end = group.shape[0] - 1
                    sequence_idx = list(range(
                        max(0, end - self.max_seq_length),
                        end,
                    ))
                    target = 0
                    sequences.append((hid, sequence_idx, target))

        return sequences
