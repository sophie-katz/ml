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

from ml.data.pytorch_name_classification.shared import (
    download_and_extract,
    convert_to_ascii,
    get_culture_name_from_file_path,
    load_name_tuples_from_dir,
)
import pathlib
import itertools


def test_convert_to_ascii() -> None:
    assert convert_to_ascii("Ég er að læra íslansku") == "Eg er að læra islansku"
    assert convert_to_ascii("a\xa0b") == "a b"


def test_get_culture_name_from_file_path() -> None:
    assert (
        get_culture_name_from_file_path(
            pathlib.Path("ml/data/pytorch_name_classification/data/names/Arabic.txt")
        )
        == "Arabic"
    )


def test_load_name_tuples_from_dir() -> None:
    names_dir = download_and_extract()

    tuples = load_name_tuples_from_dir(names_dir)
    tuples_iter = iter(tuples)

    for i in range(10):
        element = next(tuples_iter)

        assert element[0] == "Arabic"
        assert len(element[1]) > 0
