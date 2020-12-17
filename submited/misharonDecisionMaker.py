
from AbsDecisionMaker import AbsDecisionMaker

import numpy as np
from constants import SIZE_Y, SIZE_X
from Arena.CState import State
from Arena.constants import AgentAction

from json import load
from ast import literal_eval

#------------------------------------------------------------------------------------------------------------~


with open('misharon_policy.json', 'r') as myfile:
    # (i) load json object
    obj = load(myfile)
    # (ii) convert loaded keys from string back to tuple
    my_policy = {literal_eval(k): v for k, v in obj.items()}
    pass

#------------------------------------------------------------------------------------------------------------~


class misharonDecisionMaker(AbsDecisionMaker):

    def update_context(self, new_state: State, reward, is_terminal):
        self.state = new_state
        # this is needed for main.py to run:
        self._Q_matrix[(new_state.my_pos._x, new_state.my_pos._y), \
                       (new_state.enemy_pos._x, new_state.enemy_pos._y)] \
            = list(np.zeros(9))
        # end def

    def get_action(self, state: State) -> AgentAction:
        s = (state.my_pos._x, state.my_pos._y,
             state.enemy_pos._x, state.enemy_pos._y)
        a = my_policy[s]
        action = a + 1  # change to 1-based index
        return action
        # end def

    def set_initial_state(self, state: State):

        self.state = state

        # this is needed for main.py to run
        self._Q_matrix = dict()
        for x1 in range(SIZE_X):
            for y1 in range(SIZE_Y):
                for x2 in range(SIZE_X):
                    for y2 in range(SIZE_Y):
                        self._Q_matrix[(x1, y1), (x2, y2)] = list(np.ones(9))  # blue_pos x red_pos x actions

    # end def