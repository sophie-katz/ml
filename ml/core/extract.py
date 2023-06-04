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

from tqdm import tqdm
from typing import Union
import pathlib
import zipfile


def extract_archive(
    archive_path: Union[str, pathlib.Path], extract_dir: Union[str, pathlib.Path]
) -> None:
    print(f"Extracting {archive_path} to {extract_dir}...")

    # Extract archive_path zip to extract_dir with tqdm progress bar by byte count
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        total_size = sum(file.file_size for file in zip_file.infolist())
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

        for file in zip_file.infolist():
            if not file.is_dir():
                extracted_path = pathlib.Path(extract_dir, file.filename)

                if (
                    not extracted_path.exists()
                    or extracted_path.stat().st_size != file.file_size
                ):
                    zip_file.extract(file, extract_dir)

                progress_bar.update(file.file_size)

        progress_bar.close()

    print("  Extraction complete.")
