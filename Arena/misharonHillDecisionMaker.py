
from copy import copy
import numpy as np

from AbsDecisionMaker import AbsDecisionMaker
from constants import SIZE_Y, SIZE_X
from Arena.CState import State
from Arena.constants import AgentAction, HARD_AGENT
from Arena.Environment import Environment
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker

from misharon_utils import derive_greedy_policy, update_Q_matrix, init_Q_matrix, is_terminal_state
from misharon_learn_the_enemy import learn_decision_maker
from plan_anti_policy import plan_anti_policy

#------------------------------------------------------------------------------------------------------------~

# define dummy players, just so we can use the class functions
env = Environment()

#------------------------------------------------------------------------------------------------------------~


class misharonHillDecisionMaker(AbsDecisionMaker):

    def update_context(self, new_state: State, reward, is_terminal):

        prev_state = copy(self.state)
        self.state = new_state

        red_prev_state = prev_state

        update_Q_matrix(self._Q_matrix, new_state) # needed for main.py to run

        # TODO: learn the enemy - see the last move the enemy made, and update the enemy_policy - update enemy_policy
        # have two enemy_policy dicts - one is enemy_policy_init (Rafael's) and other is enemy_policy_learned


        # Update the estimate to Q*
        n_iter = 1
        converge_epsilon = 1e-3
        self.qFunc = plan_anti_policy(self.enemy_policy_cnts, n_iter, converge_epsilon, initQ=self.qFunc)

        # Policy improvement (use argmax):
        self.my_policy = derive_greedy_policy(self.qFunc, env)

        # end def

    # ------------------------------------------------------------------------------------------------------------~

    def get_action(self, state: State) -> AgentAction:
        s = (state.my_pos._x, state.my_pos._y,
             state.enemy_pos._x, state.enemy_pos._y)
        if is_terminal_state(s):
            action = 1
        else:
            a = self.my_policy[s]
            action = a + 1  # change to 1-based index
        return action
        # end def

    # ------------------------------------------------------------------------------------------------------------~

    def set_initial_state(self, state: State):

        self.state = state

        # learn the enemy - Rafael's hard agent
        red_decision_maker = RafaelDecisionMaker(HARD_AGENT)
        self.enemy_policy_cnts = learn_decision_maker(red_decision_maker, n_samples=10)

        n_iter = 1
        converge_epsilon = 1e-3

        # Update the estimate to Q*
        self.qFunc = plan_anti_policy(self.enemy_policy_cnts, n_iter, converge_epsilon, initQ=None)

        # Policy improvement (use argmax):
        self.my_policy = derive_greedy_policy(self.qFunc, env)

        self._Q_matrix = init_Q_matrix() # needed for main.py to run

    # end def

    # ------------------------------------------------------------------------------------------------------------~
