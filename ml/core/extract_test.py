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

import os
from . import extract
from . import repo_paths
from . import download


def test_extract_archive() -> None:
    url = "https://download.pytorch.org/tutorial/data.zip"
    download_dir = repo_paths.get_dir_artifacts_data_raw(
        "test_extract_archive", create=True
    )
    archive_path = download_dir / "data.zip"

    download.download_http(url, archive_path)

    assert os.path.exists(archive_path)
    assert os.stat(archive_path).st_size == 2882130

    extract_dir = repo_paths.get_dir_artifacts_data_intermediate(
        "test_extract_archive", create=True
    )

    extract.extract_archive(archive_path, extract_dir)

    assert os.path.exists(extract_dir / "data")
    assert os.path.exists(extract_dir / "data/names")
    assert os.path.exists(extract_dir / "data/names/Arabic.txt")
