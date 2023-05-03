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

import abc
from typing import Optional, Generic, TypeVar, Tuple

State = TypeVar("State")
Input = TypeVar("Input")
Output = TypeVar("Output")


class SimulationBase(abc.ABC, Generic[State, Input, Output]):
    """
    A base class for all simulations.

    Simulations are essentially data generators. Examples of them are:

     * Simulated environments for agents, such as OpenAI Gym

     * Image data generators for simplified classification data

    Simulations are characterized by:

     * Data generation upon request

     * Having a maximum number of times that data can be generated

     * Having configuration for the simulation as a whole

     * Having parameters to the generation on a per-generation basis (such as agent/user
       input)

    See https://www.notion.so/Simulations-26a385a241bf4e5199002ddc21c8dd56?pvs=4 for
    more information.

    Type parameters
    ---------------
    State
        A state that is passed from one step to the next. This state is initialized
        originally by the ``start`` method. Set to ``None`` to disable.
    Input
        An input to a given step. This can be agent or user input of some kind. Set to
        ``None`` to disable.
    Output
        The output of a given step. This cannot be set to ``None`` and is required.

    Example
    -------

    ```python
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

    state = simulation.start()
    assert state == 0

    output, state = simulation.step(state, 1)
    assert output == 0
    assert state == 1

    output, state = simulation.step(state, 2)
    assert output == 1
    assert state == 3

    output, state = simulation.step(state, 3)
    assert output == 3
    assert state == 6
    ```
    """

    @abc.abstractmethod
    def start(self) -> State:
        """
        Gets the initial state for the simulation. This is what gets passed to ``step``.
        """
        pass

    @abc.abstractmethod
    def step(
        self, state: Optional[State] = None, input: Optional[Input] = None
    ) -> Tuple[Output, Optional[State]]:
        """
        Generates data based on an optional current ``state`` and an optional ``input``.

         * If there is a ``state``, that means that it can change step to step. The lack
           of a ``state`` means that the output is entirely based off the ``input`` or
           is static.

         * If there is no ``input``, that means that the results are entirely based off
           ``state`` or are static.

         * Returns the results and an updated state. If the maximum number of steps have
           been reached, or if another halt condition has been reached, nothing will be
           returned.

        See https://www.notion.so/Simulations-26a385a241bf4e5199002ddc21c8dd56?pvs=4 for
        more information.

        Parameters
        ----------
        state: Optional[State]
            The current state. If set, this will get updated and then returned from the
            method. The update will not affect the value of this parameter.
        input: Optional[Input]
            The input to the current step. If set, this will impact the output of the
            step.

        Returns
        -------
        A tuple containing the output returned by the method and also an optional
        updated state.
        """
        pass
