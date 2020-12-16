
import time
import timeit
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import HARD_AGENT, AgentAction, SIZE_X, SIZE_Y, DSM
from Arena.Entity import Entity
from Arena.CState import State
import numpy as np

n_actions = 9




def learn_agent(decision_maker):

    env = Environment()

    blue_decision_maker = decision_maker   # the agent we want to learn its behaviour policy
    red_decision_maker = decision_maker    # the imaginary  agent we want to learn its behaviour policy


    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)

    policy_counts = dict{}

    def valid_pos_generator():
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x, y] == 1:  # obstacle
                    continue
                yield x, y
            # end for
        # end for
    # end for

    learned_policy = {}
    # go over each state = (blue position, red position)
    for blue_pos in valid_pos_generator():
        for red_pos in valid_pos_generator():
            if blue_pos == red_pos:
                continue # terminal state
            state = blue_pos + red_pos
            policy_counts[state][action] += 1

        # end for
    # end for



    pass
    # TODO: save result in pickle
# end def

if __name__ == '__main__':
    start_time = timeit.default_timer()

    decision_maker = RafaelDecisionMaker(HARD_AGENT)

    aaa = learn_agent(decision_maker)

    time_str = time.strftime("%H hours, %M minutes and %S seconds",
                             time.gmtime(timeit.default_timer() - start_time))

