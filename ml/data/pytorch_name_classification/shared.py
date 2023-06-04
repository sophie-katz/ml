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

import pathlib
import unicodedata
from typing import Iterable, Tuple
from ml.core.download import download_http
from ml.core.extract import extract_archive
from ml.core.repo_paths import (
    get_dir_artifacts_data_raw,
    get_dir_artifacts_data_intermediate,
)

DATA_NAME = "pytorch_name_classification"
DOWNLOAD_URL = "https://download.pytorch.org/tutorial/data.zip"
DOWNLOAD_FILENAME = "data.zip"


def download_and_extract() -> pathlib.Path:
    """
    Downloads and extracts the data for the PyTorch name classification tutorial.

    Returns
    -------

        A ``pathlib.Path`` path to the directory containing the name files.
    """

    path_dir_artifacts_data_raw = get_dir_artifacts_data_raw(DATA_NAME, create=True)
    path_dir_artifacts_data_intermediate = get_dir_artifacts_data_intermediate(
        DATA_NAME, create=True
    )

    path_data = path_dir_artifacts_data_raw / DOWNLOAD_FILENAME

    download_http(DOWNLOAD_URL, path_data)

    extract_archive(path_data, path_dir_artifacts_data_intermediate)

    return path_dir_artifacts_data_intermediate / "data" / "names"


def _convert_to_ascii_normalize_str(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def _convert_to_ascii_normalize_chr(c: str) -> str:
    if c == "\xa0":
        return " "
    else:
        return c


def convert_to_ascii(text: str) -> str:
    """
    Converts a string with unicode characters to ASCII by converting non-ASCII
    characters to similar ASCII characters. For example, é becomes e.
    """
    return "".join(
        [
            _convert_to_ascii_normalize_chr(c)
            for c in _convert_to_ascii_normalize_str(text)
            if unicodedata.category(c) != "Mn"
        ]
    )


def get_culture_name_from_file_path(path: pathlib.Path) -> str:
    """
    Gets the culture name from the file path of a name file.

    Parameters
    ----------
    ``path``
        The path to the name file.

    Returns
    -------
        The culture name.
    """

    return path.name.split(".")[0]


def load_name_tuples_from_dir(path: pathlib.Path) -> Iterable[Tuple[str, str]]:
    """
    Loads the name tuples from the name directory.

    The name tuples are in the format ``(culture_name, name)`` where ``culture_name``
    is the name of the culture and ``name`` is the name from that culture.

    Parameters
    ----------
    ``path``
        The path to the name directory.

    Returns
    -------
        A generator of name tuples.
    """

    assert path.is_dir(), "path must be a directory"

    for name_path in path.iterdir():
        culture_name = get_culture_name_from_file_path(name_path)
        with open(name_path, "r", encoding="utf-8") as file:
            for line in file.readlines():
                yield (culture_name, line.strip())
