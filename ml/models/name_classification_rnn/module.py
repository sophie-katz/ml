# Copyright (c) 2023 Sophie Katz
#
# This file is part of Sophie's ML Monorepo.
#
# Sophie's ML Monorepo is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later version.
#
# Sophie's ML Monorepo is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Sophie's
# ML Monorepo. If not, see <https://www.gnu.org/licenses/>.

from ml.models.name_classification_rnn.model import NameClassificationRNN
from typing import cast, Optional, Tuple
import torch as T
import torch.nn as nn
import torch.optim as O
import lightning.pytorch as pl

LEARNING_RATE = 0.005


class NameClassificationRNNModule(pl.LightningModule):  # type: ignore
    def __init__(self, model: NameClassificationRNN) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = nn.NLLLoss()

    def training_step(
        self, batch: Tuple[T.Tensor, T.Tensor], batch_index: int
    ) -> T.Tensor:
        name, culture_label = batch
        culture_predicted: Optional[T.Tensor] = None

        hidden = self.model.create_hidden_initial().to(self.device)

        for i in range(name.shape[0]):
            culture_predicted, hidden = self.model(name[:, i], hidden)

        assert culture_predicted is not None

        loss = self.loss_fn(culture_predicted.squeeze(0), culture_label.squeeze(0))

        self.log("train_loss", loss)

        return cast(T.Tensor, loss)

    def configure_optimizers(self) -> O.Optimizer:
        return O.SGD(self.model.parameters(), lr=LEARNING_RATE)

    def backward(self, loss: T.Tensor) -> None:
        loss.backward(retain_graph=True)  # type: ignore
