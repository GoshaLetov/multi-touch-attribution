import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, attention_size: int, non_linearity: str, time_decay: float, batch_first=False):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = nn.Parameter(torch.FloatTensor(attention_size), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.time_decay = time_decay

        if non_linearity == 'identity':
            self.non_linearity = nn.Identity()
        elif non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        else:
            raise ValueError(f'You must select identity, tanh or relu. Got {non_linearity}')

        nn.init.uniform_(self.attention_weights.data, a=0.005, b=0.005)

    def get_mask(self, attentions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = torch.ones(attentions.size())

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self,
                inputs: torch.Tensor,
                lengths: torch.Tensor,
                time_delta: torch.Tensor,
                ) -> tuple[torch.Tensor, torch.Tensor]:

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights)) - self.time_decay * time_delta
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores


class MultiTouchAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int, non_linearity: str, time_decay: float):
        super(MultiTouchAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=True),
            nn.Tanh(),
        )
        self.attention = SelfAttention(attention_size=out_features, non_linearity=non_linearity, time_decay=time_decay)

    def forward(self, x: torch.Tensor, length: torch.Tensor, time_delta: torch.Tensor) -> torch.Tensor:
        return self.attention(self.mlp(x), length, time_delta)


class ControlsFCN(nn.Module):
    def __init__(self, mapping: dict[str, dict[int, int]]):
        super(ControlsFCN, self).__init__()
        self.mapping = mapping
        self.columns = sorted(mapping.keys())
        self.output_len = 0

        self.embeddings = {}
        for key, value in mapping.items():
            self.embeddings[key] = nn.Embedding(num_embeddings=len(value), embedding_dim=len(value))
            self.output_len += len(value)

    def forward(self, controls: dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []

        for column in self.columns:
            if column in controls:
                embeddings.append(self.embeddings[column](controls[column]))
        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings


class LSTMAttention(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        mapping: dict[str, dict[int, int]] = None,
        embedding_dim: int = 32,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
        attention_non_linearity: str = 'identity',
        time_decay: float = 0.,
    ) -> None:
        super(LSTMAttention, self).__init__()

        self.controls = mapping is not None

        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False,
        )

        self.attention = MultiTouchAttention(
            in_features=hidden_size,
            out_features=hidden_size,
            non_linearity=attention_non_linearity,
            time_decay=time_decay,
        )

        in_features = hidden_size

        if mapping:
            self.controls = ControlsFCN(
                mapping=mapping,
            )
            in_features += self.controls.output_len

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=1,
            ),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0, end_dim=-1),
        )

    def forward(self,
                sequence: dict[str, torch.Tensor],
                controls: dict[str, torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, _ = self.lstm(self.embeddings(sequence['sequence']))
        representations, scores = self.attention(outputs, sequence['lengths'], sequence['time'])

        if self.controls:
            representations = torch.cat([representations, self.controls(controls)], dim=-1)

        return self.classifier(representations.to(torch.float32)), scores
