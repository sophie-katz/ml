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

from ml.core.simulation import SimulationBase
import numpy.typing as npt
import numpy as np
import dataclasses
import math
from typing import cast, List, Tuple, Optional

SampleType = np.int8


@dataclasses.dataclass
class SimulationMineLearning2DSolid(
    SimulationBase[None, npt.ArrayLike, npt.NDArray[SampleType]],
):
    """
    A simulation for MineLearning which outputs an image made up of regular squares of
    solid color.

    It takes a 2D array as input containing integers that are indexes of the colors in
    ``colors`` to use for a given grid square.

    Parameters
    ----------
    width: int
        The width of the image in pixels
    height: int
        The height of the image in pixels
    square_size: int
        The size of the grid squares in pixels
    colors: npt.ArrayLike
        An array of colors where each color is a list of channel values. For example, to
        represent red and blue in RGB colors would be:
        ``[(0xff, 0x00, 0x00), (0x00, 0x00, 0xff)]``

    Examples
    --------

    Let's say we have a 5x5 pixel image that we want to cover in 2x2 pixel squares of
    solid colors.

    ```python
    simulation = SimulationMineLearning2DSolid(
        width = 5,
        height = 5,
        square_size = 2,
        colors = [
            (0xff, 0x00, 0x00), # red
            (0x00, 0xff, 0x00), # green
            (0x00, 0x00, 0xff), # blue
        ]
    )
    ```

    Let's say we want to have a checkerboard of red and blue:

    ```python
    output, _ = simulation.step(input=[
        [0, 2, 0],
        [2, 0, 2],
        [0, 2, 0],
    ])
    ```

    Output should give us the pixels we want:

    ```python
    assert (
        output
        == [
            # y = 0
            [
                # x = 0
                [0xff, 0x00, 0x00], # red

                # x = 1
                [0xff, 0x00, 0x00], # red

                # x = 2
                [0x00, 0x00, 0xff], # blue

                # x = 3
                [0x00, 0x00, 0xff], # blue

                # x = 4
                [0xff, 0x00, 0x00], # red
            ],
            # y = 1
            [
                # x = 0
                [0xff, 0x00, 0x00], # red

                # x = 1
                [0xff, 0x00, 0x00], # red

                # x = 2
                [0x00, 0x00, 0xff], # blue

                # x = 3
                [0x00, 0x00, 0xff], # blue

                # x = 4
                [0xff, 0x00, 0x00], # red
            ],
            # y = 2
            [
                # x = 0
                [0x00, 0x00, 0xff], # blue

                # x = 1
                [0x00, 0x00, 0xff], # blue

                # x = 2
                [0xff, 0x00, 0x00], # red

                # x = 3
                [0xff, 0x00, 0x00], # red

                # x = 4
                [0x00, 0x00, 0xff], # blue
            ],
            # y = 3
            [
                # x = 0
                [0x00, 0x00, 0xff], # blue

                # x = 1
                [0x00, 0x00, 0xff], # blue

                # x = 2
                [0xff, 0x00, 0x00], # red

                # x = 3
                [0xff, 0x00, 0x00], # red

                # x = 4
                [0x00, 0x00, 0xff], # blue
            ],
            # y = 4
            [
                # x = 0
                [0xff, 0x00, 0x00], # red

                # x = 1
                [0xff, 0x00, 0x00], # red

                # x = 2
                [0x00, 0x00, 0xff], # blue

                # x = 3
                [0x00, 0x00, 0xff], # blue

                # x = 4
                [0xff, 0x00, 0x00], # red
            ],
        ]
    ).all()
    ```
    """

    width: int
    height: int
    square_size: int
    colors: npt.ArrayLike

    def start(self) -> None:
        pass

    def step(
        self, state: None = None, input: Optional[npt.ArrayLike] = None
    ) -> Tuple[npt.NDArray[SampleType], None]:
        assert input is not None, "input is expected to be a 2D array of color indexes"

        # Ensure that properties are valid
        self._validate()

        # Create empty result
        result: List[List[npt.NDArray[SampleType]]] = []

        # Populate result
        for y in range(self.height):
            result.append([])
            for x in range(self.width):
                pixel = self._get_pixel(x, y, input)
                assert len(np.shape(pixel)) == 1
                result[-1].append(pixel)

        # Return result
        return np.array(result), None

    def _validate(self) -> None:
        # Validate properties
        assert self.width > 0
        assert self.height > 0
        assert self.square_size > 0
        assert len(np.shape(self.colors)) == 2

    def _get_pixel(
        self, x: int, y: int, input: npt.ArrayLike
    ) -> npt.NDArray[SampleType]:
        # Validate parameters
        assert x >= 0
        assert x < self.width
        assert y >= 0
        assert y < self.height
        assert len(np.shape(input)) == 2

        # Calculate the grid index
        grid_x = x // self.square_size
        grid_y = y // self.square_size

        # Get the color index from the input
        assert grid_y < np.shape(input)[0]
        assert grid_x < np.shape(input)[1]
        color_index: int = np.array(input)[grid_y, grid_x]

        # Return the color at the given index
        assert color_index < np.shape(self.colors)[0]
        return cast(
            npt.NDArray[SampleType],
            np.array(self.colors)[color_index],
        )
