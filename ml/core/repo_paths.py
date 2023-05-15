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
import os
from typing import Optional
import re

_PROJECT_NAME_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _is_git_repo(path: pathlib.Path) -> bool:
    if not path.is_dir():
        return False

    for child in path.iterdir():
        if child.name == ".git":
            return True

    return False


def _is_ml_repo(path: pathlib.Path) -> bool:
    if not path.is_dir():
        return False

    has_core = False
    has_dot_vscode = False
    has_readme = False

    for child in path.iterdir():
        if child.name == "ml":
            has_core = True
        elif child.name == ".vscode":
            has_dot_vscode = True
        elif child.name.lower() == "readme.md":
            with open(child, "r") as file:
                if "Sophie's ML Monorepo" in file.read():
                    has_readme = True

    return has_core and has_dot_vscode and has_readme


def _validate_project_name(project_name: str) -> None:
    if not _PROJECT_NAME_PATTERN.match(project_name):
        raise Exception(
            f"invalid project name: {project_name!r} (must contain only letters, numbers, and '_', and cannot begin with a number)"
        )


def _optionally_create_and_return(path: pathlib.Path, create: bool) -> pathlib.Path:
    if create:
        os.makedirs(path, exist_ok=True)

    return path


def get_repo_root_path(
    cwd: Optional[pathlib.Path] = None, create: bool = False
) -> pathlib.Path:
    start: pathlib.Path

    if cwd is None:
        start = pathlib.Path(os.getcwd())
    else:
        start = cwd

    result = start

    while not _is_git_repo(result) and result != result.parent:
        result = result.parent

    if not _is_git_repo(result):
        raise Exception(
            f"no Git repository found in cwd or parent directories (cwd: {start})"
        )

    if not _is_ml_repo(result):
        raise Exception(
            f'Git repository found in cwd or parent directories does not contain expected children (required: .vscode/, core/, README.md containing "Sophie\'s ML Monorepo", cwd: {start})'
        )

    return _optionally_create_and_return(result, create)


def get_dir_artifacts_data_raw(
    project_name: str, cwd: Optional[pathlib.Path] = None, create: bool = False
) -> pathlib.Path:
    _validate_project_name(project_name)

    return _optionally_create_and_return(
        get_repo_root_path(cwd) / "artifacts" / "data" / project_name / "raw", create
    )


def get_dir_artifacts_data_intermediate(
    project_name: str, cwd: Optional[pathlib.Path] = None, create: bool = False
) -> pathlib.Path:
    _validate_project_name(project_name)

    return _optionally_create_and_return(
        get_repo_root_path(cwd) / "artifacts" / "data" / project_name / "intermediate",
        create,
    )


def get_dir_artifacts_data_cache(
    project_name: str, cwd: Optional[pathlib.Path] = None, create: bool = False
) -> pathlib.Path:
    _validate_project_name(project_name)

    return _optionally_create_and_return(
        get_repo_root_path(cwd) / "artifacts" / "data" / project_name / "cache", create
    )


def get_dir_checkpoints(
    project_name: str, cwd: Optional[pathlib.Path] = None, create: bool = False
) -> pathlib.Path:
    _validate_project_name(project_name)

    return _optionally_create_and_return(
        get_repo_root_path(cwd) / "artifacts" / "checkpoints" / project_name, create
    )


def get_dir_models(
    project_name: str, cwd: Optional[pathlib.Path] = None, create: bool = False
) -> pathlib.Path:
    _validate_project_name(project_name)

    return _optionally_create_and_return(
        get_repo_root_path(cwd) / "artifacts" / "models" / project_name, create
    )
