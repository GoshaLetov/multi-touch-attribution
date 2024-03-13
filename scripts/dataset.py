import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    sequences, targets, lengths, time_delta = zip(*batch)
    padded_sequences = pad_sequence(sequences=sequences, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    lengths = torch.stack(lengths)
    time_delta = pad_sequence(sequences=time_delta, batch_first=True, padding_value=0)
    return padded_sequences, targets, lengths, time_delta


class AdBannerDataset(Dataset):
    def __init__(self, data: pd.DataFrame, max_seq_length: int = 10, min_seq_length: int = 3, augmented: bool = False):
        self.data = data
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.augmented = augmented
        self.sequences = self._generate_sequences()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hid, sequence_idx, target = self.sequences[idx]

        sequence = self._get_sequence(hid, sequence_idx)

        time_delta = torch.log(torch.tensor(sequence.time_delta.tolist()).float())
        sequence = torch.tensor(sequence.media.tolist()).long()
        targets = torch.tensor(target)
        lengths = torch.tensor(len(sequence)).long()

        return sequence, targets, lengths, time_delta

    def _get_sequence(self, hid: str, sequence_idx: list[int]) -> pd.DataFrame:
        sequence = self.data[self.data['hid'] == hid]

        max_time = sequence.iloc[max(sequence_idx) + 1].tsEvent_datetime
        sequence = sequence.iloc[sequence_idx]

        sequence = sequence.assign(
            time_delta=(max_time - sequence.tsEvent_datetime).dt.seconds / 60 / 60 + 1
        )

        return sequence

    def _generate_sequences(self) -> list[tuple[str, list[int], int]]:
        sequences = []
        for hid, group in self.data.groupby('hid'):

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
