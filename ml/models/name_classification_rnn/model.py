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

from typing import cast, Dict, Tuple
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
        index_to_alphabet_mapping: Dict[int, str],
        culture_name_count: int,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(
            len(index_to_alphabet_mapping) + hidden_size, hidden_size
        )
        self.input_to_output = nn.Linear(
            len(index_to_alphabet_mapping) + hidden_size,
            culture_name_count,
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self, input_tensor: T.Tensor, hidden_tensor: T.Tensor
    ) -> Tuple[T.Tensor, T.Tensor]:
        combined_tensor = T.cat((input_tensor, hidden_tensor), 1)
        hidden_tensor = self.input_to_hidden(combined_tensor)
        output_tensor = self.input_to_output(combined_tensor)
        output_tensor = self.softmax(output_tensor)

        return output_tensor, hidden_tensor

    def create_hidden_initial(self) -> T.Tensor:
        return T.zeros(1, self.hidden_size)

    def save(self, path: pathlib.Path) -> None:
        T.save(self, path)

    def infer(self, dataset: PytorchNameCategorization, name: str) -> str:
        # nn.functional.one_hot(T.tensor(dataset.alphabet_mapping()[name[0]]))

        # input_tensor = dataset[0][0]
        # label_tensor = dataset[0][1]
        # hidden_tensor = model.create_hidden_initial()

        # output_tensor, hidden_tensor_next = model(
        #     input_tensor[0].unsqueeze(0), hidden_tensor
        # )

        # print(
        #     f"Name: {''.join(dataset.alphabet_mapping()[i.item()] for i in input_tensor.argmax(1))!r}, Label: {dataset.culture_name_mapping()[int(label_tensor.argmax().item())]!r}, Prediction: {dataset.culture_name_mapping()[int(output_tensor.argmax().item())]!r}",
        # )
        raise NotImplementedError()
