import abc
from typing import Tuple

from Arena.CState import State
from Arena.constants import AgentAction


class AbsDecisionMaker(metaclass=abc.ABCMeta):

    def update_context(self, new_state: State, reward, is_terminal):
        self.my_pos = new_state
        pass

    def get_action(self, state: State)-> AgentAction:
        return AgentAction.BottomLeft
        pass

    def set_initial_state(self, state: State):

        self.my_pos = state
        pass