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

from .repo_paths import *
import pathlib
import pytest


def test_get_repo_root_path_good() -> None:
    result = get_repo_root_path()
    assert (result / "ml").is_dir()


def test_get_repo_root_path_bad() -> None:
    with pytest.raises(Exception):
        get_repo_root_path(pathlib.Path.home())
