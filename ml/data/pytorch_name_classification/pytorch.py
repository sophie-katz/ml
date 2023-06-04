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

from torch.utils.data import Dataset
from ml.data.pytorch_name_classification.shared import (
    download_and_extract,
    load_name_tuples_from_dir,
)
from typing import cast, Dict, List, Tuple
import torch as T
import torch.nn as nn


class PytorchNameCategorization(Dataset[Tuple[T.Tensor, T.Tensor]]):
    def __init__(self) -> None:
        self._data: List[Tuple[T.Tensor, T.Tensor]] = []
        self._alphabet_to_index_mapping: Dict[str, int] = {}
        self._culture_name_index_to_mapping: Dict[str, int] = {}
        self._index_to_culture_name_mapping: Dict[int, str] = {}
        self._culture_name_counts: Dict[str, int] = {}
        self._loaded = False

    def alphabet_count(self) -> int:
        if not self._loaded:
            self._load()

        return len(self._alphabet_to_index_mapping)

    def culture_name_count(self) -> int:
        if not self._loaded:
            self._load()

        return len(self._culture_name_index_to_mapping)

    def culture_name_weights(self) -> T.Tensor:
        if not self._loaded:
            self._load()

        return T.tensor(
            [
                len(self._data)
                / self._culture_name_counts[self._index_to_culture_name_mapping[index]]
                for index in range(len(self._culture_name_index_to_mapping))
            ]
        )

    def encode_name(self, name: str) -> T.Tensor:
        if not self._loaded:
            self._load()

        alphabet_indices = [
            self._alphabet_to_index_mapping[character] for character in name
        ]

        return cast(
            T.Tensor,
            nn.functional.one_hot(
                T.tensor(alphabet_indices),
                num_classes=len(self._alphabet_to_index_mapping),
            ).to(T.float32),
        )

    def encode_culture_name(self, culture_name: str) -> T.Tensor:
        if not self._loaded:
            self._load()

        culture_index = self._culture_name_index_to_mapping[culture_name]

        # return cast(
        #     T.Tensor,
        #     nn.functional.one_hot(
        #         T.tensor(culture_index),
        #         num_classes=len(self._culture_name_index_to_mapping),
        #     ).to(T.float32),
        # )

        return T.tensor(culture_index, dtype=T.long)

    def decode_culture_name(self, tensor: T.Tensor) -> str:
        if not self._loaded:
            self._load()

        return self._index_to_culture_name_mapping[int(tensor.argmax().item())]

    def __len__(self) -> int:
        if not self._loaded:
            self._load()

        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[T.Tensor, T.Tensor]:
        if not self._loaded:
            self._load()

        return self._data[index]

    def _load(self) -> None:
        names_dir = download_and_extract()
        names = []

        alphabet = set()
        culture_names = set()

        for element in load_name_tuples_from_dir(names_dir):
            names.append(element)
            for character in element[1]:
                alphabet.add(character)
            culture_names.add(element[0])

            if element[0] in self._culture_name_counts:
                self._culture_name_counts[element[0]] += 1
            else:
                self._culture_name_counts[element[0]] = 1

        for index, character in enumerate(sorted(alphabet)):
            self._alphabet_to_index_mapping[character] = index

        for index, culture_name in enumerate(sorted(culture_names)):
            self._culture_name_index_to_mapping[culture_name] = index
            self._index_to_culture_name_mapping[index] = culture_name

        # Needs to be set here because of the subsequent calls to encode_name and
        # encode_culture_name.
        self._loaded = True

        for culture_name, name in names:
            name_encoding = self.encode_name(name)
            culture_encoding = self.encode_culture_name(culture_name)

            if culture_name != "Arabic":
                self._data.append((name_encoding, culture_encoding))
