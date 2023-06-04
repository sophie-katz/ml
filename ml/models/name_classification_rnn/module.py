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

from ml.data.pytorch_name_classification.pytorch import PytorchNameCategorization
from ml.models.name_classification_rnn.model import NameClassificationRNN
from typing import cast, Dict, Optional, Tuple
import torch as T
import torch.nn as nn
import torch.optim as O
import lightning.pytorch as pl

LEARNING_RATE = 0.005


class NameClassificationRNNModule(pl.LightningModule):  # type: ignore
    def __init__(
        self, dataset: PytorchNameCategorization, model: NameClassificationRNN
    ) -> None:
        super().__init__()

        self.model = model
        self.loss_fn = nn.NLLLoss(weight=dataset.culture_name_weights())
        self.culture_name_counts_label: Dict[int, int] = {}
        self.culture_name_counts_predicted: Dict[int, int] = {}

    def training_step(
        self, batch: Tuple[T.Tensor, T.Tensor], batch_index: int
    ) -> T.Tensor:
        name, culture_label = batch
        culture_predicted: Optional[T.Tensor] = None

        hidden = self.model.create_hidden_initial().to(self.device)

        for i in range(name.shape[0]):
            culture_predicted, hidden = self.model(name[i], hidden)

        assert culture_predicted is not None

        loss = self.loss_fn(culture_predicted, culture_label)

        self.log("train_loss", loss)

        culture_name_index_label = culture_label.item()
        if not culture_name_index_label in self.culture_name_counts_label:
            self.culture_name_counts_label[int(culture_name_index_label)] = 1
        else:
            self.culture_name_counts_label[int(culture_name_index_label)] += 1

        culture_name_index_predicted = culture_predicted.argmax().item()
        if not culture_name_index_predicted in self.culture_name_counts_predicted:
            self.culture_name_counts_predicted[int(culture_name_index_predicted)] = 1
        else:
            self.culture_name_counts_predicted[int(culture_name_index_predicted)] += 1

        return cast(T.Tensor, loss)

    def configure_optimizers(self) -> O.Optimizer:
        return O.SGD(self.model.parameters(), lr=LEARNING_RATE)

    def backward(self, loss: T.Tensor) -> None:
        loss.backward(retain_graph=True)  # type: ignore

    def on_train_end(self) -> None:
        print("Culture name counts label:")
        print(self.culture_name_counts_label)

        print("Culture name counts predicted:")
        print(self.culture_name_counts_predicted)
