from typing import Any

import torch
import torchmetrics
import lightning.pytorch as pl


class AdBannerLightningModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float):
        super().__init__()

        self._model = model
        self._metrics_valid = torchmetrics.MetricCollection(metrics={
            'Accuracy': torchmetrics.classification.BinaryAccuracy(threshold=0.5),
            'AUC': torchmetrics.classification.BinaryAUROC(),
        }).clone(prefix='valid.')
        self._metrics_test = torchmetrics.MetricCollection(metrics={
            'Accuracy': torchmetrics.classification.BinaryAccuracy(threshold=0.5),
            'AUC': torchmetrics.classification.BinaryAUROC(),
        }).clone(prefix='test.')
        self._loss = torch.nn.BCELoss()
        self._lr = lr

    def forward(self,
                sequence: dict[str, torch.Tensor],
                controls: dict[str, torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:

        for key, value in sequence.items():
            sequence[key] = value.to(self.device)

        if controls:
            for key, value in controls.items():
                controls[key] = value.to(self.device)

        return self._model(sequence, controls)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'valid.AUC',
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        sequence, controls, labels = batch
        probas, _ = self.forward(sequence, controls)

        loss = self._loss(probas, labels.to(dtype=torch.float, device=self.device))
        self.log(name=f'train.BCELoss', value=loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        sequence, controls, labels = batch
        probas, _ = self.forward(sequence, controls)
        loss = self._loss(probas, labels.to(dtype=torch.float, device=self.device))
        self.log(name=f'valid.BCELoss', value=loss.item())
        self._metrics_valid(probas, labels)

    def test_step(self, batch, batch_idx):
        sequence, controls, labels = batch
        probas, _ = self.forward(sequence, controls)
        self._metrics_test(probas, labels)

    def predict_step(self, batch, batch_idx):
        sequence, controls, labels = batch
        probas, scores = self.forward(sequence, controls)
        return probas, scores

    def on_validation_epoch_start(self) -> None:
        self._metrics_valid.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._metrics_valid.compute(), on_epoch=True, on_step=False)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._metrics_test.compute(), on_epoch=True, on_step=False)
