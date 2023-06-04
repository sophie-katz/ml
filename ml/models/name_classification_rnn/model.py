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

from typing import cast, Tuple
import torch as T
import torch.nn as nn
import pathlib
from ml.data.pytorch_name_classification.pytorch import PytorchNameCategorization


HIDDEN_SIZE = 128


class NameClassificationRNN(nn.Module):
    @staticmethod
    def load(path: pathlib.Path) -> "NameClassificationRNN":
        return cast(NameClassificationRNN, T.load(path))

    def __init__(
        self,
        hidden_size: int,
        alphabet_count: int,
        culture_name_count: int,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(alphabet_count + hidden_size, hidden_size)
        self.input_to_output = nn.Linear(
            alphabet_count + hidden_size,
            culture_name_count,
        )
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(
        self, input_tensor: T.Tensor, hidden_tensor: T.Tensor
    ) -> Tuple[T.Tensor, T.Tensor]:
        combined_tensor = T.cat((input_tensor, hidden_tensor), 0)
        hidden_tensor = self.input_to_hidden(combined_tensor)
        output_tensor = self.input_to_output(combined_tensor)
        output_tensor = self.softmax(output_tensor)

        return output_tensor, hidden_tensor

    def create_hidden_initial(self) -> T.Tensor:
        return T.zeros(self.hidden_size)

    def save(self, path: pathlib.Path) -> None:
        T.save(self, path)

    def infer(self, dataset: PytorchNameCategorization, name: str) -> str:
        input_tensor = dataset.encode_name(name)
        hidden_tensor = self.create_hidden_initial()

        training = self.training

        try:
            self.eval()
            for i in range(input_tensor.shape[0]):
                output_tensor, hidden_tensor = self.forward(
                    input_tensor[i], hidden_tensor
                )
        finally:
            if training:
                self.train()

        return dataset.decode_culture_name(output_tensor)
