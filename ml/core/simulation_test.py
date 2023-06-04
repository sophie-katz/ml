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

from typing import Optional, Tuple
from .simulation import *


def test_no_state_no_input() -> None:
    class MySimulation(SimulationBase[None, None, int]):
        def start(self) -> None:
            pass

        def step(self, state: None = None, input: None = None) -> Tuple[int, None]:
            assert state is None
            assert input is None

            return 5, None

    simulation = MySimulation()

    assert simulation.start() is None  # type: ignore

    output, state = simulation.step()
    assert output == 5
    assert state is None

    output, state = simulation.step()
    assert output == 5
    assert state is None


def test_no_state_with_input() -> None:
    class MySimulation(SimulationBase[None, int, int]):
        def start(self) -> None:
            pass

        def step(
            self, state: None = None, input: Optional[int] = None
        ) -> Tuple[int, None]:
            assert state is None
            assert input is not None

            return input, None

    simulation = MySimulation()

    assert simulation.start() is None  # type: ignore

    output, state = simulation.step(input=1)
    assert output == 1
    assert state is None

    output, state = simulation.step(input=2)
    assert output == 2
    assert state is None

    output, state = simulation.step(input=3)
    assert output == 3
    assert state is None


def test_with_state_no_input() -> None:
    class MySimulation(SimulationBase[int, int, int]):
        def start(self) -> int:
            return 0

        def step(
            self, state: Optional[int] = None, input: Optional[int] = None
        ) -> Tuple[int, Optional[int]]:
            assert state is not None
            assert input is None

            state_next = state + 1

            return state, state_next

    simulation = MySimulation()

    state: Optional[int] = simulation.start()
    assert state == 0

    output, state = simulation.step(state=state)
    assert output == 0
    assert state == 1

    output, state = simulation.step(state=state)
    assert output == 1
    assert state == 2

    output, state = simulation.step(state=state)
    assert output == 2
    assert state == 3


def test_with_state_with_input() -> None:
    class MySimulation(SimulationBase[int, int, int]):
        def start(self) -> int:
            return 0

        def step(
            self, state: Optional[int] = None, input: Optional[int] = None
        ) -> Tuple[int, Optional[int]]:
            assert state is not None
            assert input is not None

            state_next = state + input

            return state, state_next

    simulation = MySimulation()

    state: Optional[int] = simulation.start()
    assert state == 0

    output, state = simulation.step(state=state, input=1)
    assert output == 0
    assert state == 1

    output, state = simulation.step(state=state, input=2)
    assert output == 1
    assert state == 3

    output, state = simulation.step(state=state, input=3)
    assert output == 3
    assert state == 6
