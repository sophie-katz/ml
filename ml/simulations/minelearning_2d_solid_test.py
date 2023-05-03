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

from .minelearning_2d_solid import SimulationMineLearning2DSolid


def test_width_2_height_2_square_size_1_colors_1() -> None:
    simulation = SimulationMineLearning2DSolid(2, 2, 1, [(1, 2, 3)])

    output, _ = simulation.step(input=[[0, 0], [0, 0]])

    assert (
        output
        == [
            # y = 0
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [1, 2, 3],
            ],
            # y = 1
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [1, 2, 3],
            ],
        ]
    ).all()


def test_width_2_height_2_square_size_1_colors_2() -> None:
    simulation = SimulationMineLearning2DSolid(2, 2, 1, [(1, 2, 3), (4, 5, 6)])

    output, _ = simulation.step(input=[[0, 1], [0, 1]])

    assert (
        output
        == [
            # y = 0
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [4, 5, 6],
            ],
            # y = 1
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [4, 5, 6],
            ],
        ]
    ).all()


def test_width_2_height_2_square_size_1_colors_3() -> None:
    simulation = SimulationMineLearning2DSolid(
        2, 2, 1, [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    )

    output, _ = simulation.step(input=[[0, 1], [2, 0]])

    assert (
        output
        == [
            # y = 0
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [4, 5, 6],
            ],
            # y = 1
            [
                # x = 0
                [7, 8, 9],
                # x = 1
                [1, 2, 3],
            ],
        ]
    ).all()


def test_width_2_height_2_square_size_2_colors_1() -> None:
    simulation = SimulationMineLearning2DSolid(2, 2, 2, [(1, 2, 3)])

    output, _ = simulation.step(input=[[0]])

    assert (
        output
        == [
            # y = 0
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [1, 2, 3],
            ],
            # y = 1
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [1, 2, 3],
            ],
        ]
    ).all()


def test_width_2_height_2_square_size_2_colors_2() -> None:
    simulation = SimulationMineLearning2DSolid(2, 2, 2, [(1, 2, 3), (4, 5, 6)])

    output, _ = simulation.step(input=[[0]])

    assert (
        output
        == [
            # y = 0
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [1, 2, 3],
            ],
            # y = 1
            [
                # x = 0
                [1, 2, 3],
                # x = 1
                [1, 2, 3],
            ],
        ]
    ).all()


def test_width_5_height_5_square_size_2_colors_3() -> None:
    simulation = SimulationMineLearning2DSolid(
        width=5,
        height=5,
        square_size=2,
        colors=[
            (0xFF, 0x00, 0x00),  # red
            (0x00, 0xFF, 0x00),  # green
            (0x00, 0x00, 0xFF),  # blue
        ],
    )

    output, _ = simulation.step(
        input=[
            [0, 2, 0],
            [2, 0, 2],
            [0, 2, 0],
        ]
    )

    assert (
        output
        == [
            # y = 0
            [
                # x = 0
                [0xFF, 0x00, 0x00],  # red
                # x = 1
                [0xFF, 0x00, 0x00],  # red
                # x = 2
                [0x00, 0x00, 0xFF],  # blue
                # x = 3
                [0x00, 0x00, 0xFF],  # blue
                # x = 4
                [0xFF, 0x00, 0x00],  # red
            ],
            # y = 1
            [
                # x = 0
                [0xFF, 0x00, 0x00],  # red
                # x = 1
                [0xFF, 0x00, 0x00],  # red
                # x = 2
                [0x00, 0x00, 0xFF],  # blue
                # x = 3
                [0x00, 0x00, 0xFF],  # blue
                # x = 4
                [0xFF, 0x00, 0x00],  # red
            ],
            # y = 2
            [
                # x = 0
                [0x00, 0x00, 0xFF],  # blue
                # x = 1
                [0x00, 0x00, 0xFF],  # blue
                # x = 2
                [0xFF, 0x00, 0x00],  # red
                # x = 3
                [0xFF, 0x00, 0x00],  # red
                # x = 4
                [0x00, 0x00, 0xFF],  # blue
            ],
            # y = 3
            [
                # x = 0
                [0x00, 0x00, 0xFF],  # blue
                # x = 1
                [0x00, 0x00, 0xFF],  # blue
                # x = 2
                [0xFF, 0x00, 0x00],  # red
                # x = 3
                [0xFF, 0x00, 0x00],  # red
                # x = 4
                [0x00, 0x00, 0xFF],  # blue
            ],
            # y = 4
            [
                # x = 0
                [0xFF, 0x00, 0x00],  # red
                # x = 1
                [0xFF, 0x00, 0x00],  # red
                # x = 2
                [0x00, 0x00, 0xFF],  # blue
                # x = 3
                [0x00, 0x00, 0xFF],  # blue
                # x = 4
                [0xFF, 0x00, 0x00],  # red
            ],
        ]
    ).all()
