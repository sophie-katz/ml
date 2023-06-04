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


def test_len() -> None:
    dataset = PytorchNameCategorization()
    assert len(dataset) == 20074


def test_items() -> None:
    dataset = PytorchNameCategorization()

    assert dataset[0][0].shape == (6, 87)
    assert dataset[0][1].shape == (18,)

    assert dataset[100][0].shape == (5, 87)
    assert dataset[100][1].shape == (18,)
