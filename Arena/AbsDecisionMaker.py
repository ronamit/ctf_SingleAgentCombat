import abc
from typing import Tuple
import numpy as np
from constants import SIZE_Y, SIZE_X

from Arena.CState import State
from Arena.constants import AgentAction


class AbsDecisionMaker(metaclass=abc.ABCMeta):

    def update_context(self, new_state: State, reward, is_terminal):
        self.my_pos = new_state
        # TODO: correct _Q_matrix
        # blue_pos x red_pos x actions
        # self._Q_matrix
        pass

    def get_action(self, state: State)-> AgentAction:


        coin = np.random.randint(3)

        if coin == 2:
            return AgentAction.BottomLeft  # (up-left)
        elif coin == 1:
            return AgentAction.Left  # (up)
        elif coin == 0:
            return AgentAction.Bottom  # (left)
        else:
            raise ValueError

        pass

    def set_initial_state(self, state: State):
        # TODO: correct _Q_matrix
        self._Q_matrix = dict()
        for x1 in range(SIZE_X):
            for y1 in range(SIZE_Y):
                for x2 in range(SIZE_X):
                    for y2 in range(SIZE_Y):
                        self._Q_matrix[(x1, y1), (x2, y2)] = list(np.zeros(9)) # blue_pos x red_pos x actions
        self.my_pos = state
        pass