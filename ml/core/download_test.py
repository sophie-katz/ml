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
from . import repo_paths
from . import download


def test_download_http() -> None:
    url = "https://sherlock-holm.es/stories/plain-text/cano.txt"
    output_dir = repo_paths.get_dir_artifacts_data_raw("test_download_http")
    output_path = output_dir / "cano.txt"

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    assert not os.path.exists(output_path)

    download.download_http(url, output_path)

    assert os.path.exists(output_path)
    assert os.stat(output_path).st_size == 3868223
