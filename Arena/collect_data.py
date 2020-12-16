
import time
import timeit
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import HARD_AGENT, AgentAction, SIZE_X, SIZE_Y
from Arena.Entity import Entity
from Arena.CState import State
import numpy as np


def update_count(d, k):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1

def collect_data(n_samples):


    env = Environment()

    blue_decision_maker = RafaelDecisionMaker(HARD_AGENT)
    red_decision_maker = RafaelDecisionMaker(HARD_AGENT)

    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)
    # initialize the decision_makers for the players


    # Init model estimator
    # P_est: [S x A x S] estimated transitions probabilities matrix  P_{s,a,s'}=P(s'|s,a)

    # nS = (SIZE_X * SIZE_Y)**2  # number of states
    # nA = 9
    # counts_sas = np.zeros((nS, nA, nS), dtype=np.uint)
    # counts_sa = np.zeros((nS, nA), dtype=np.uint)

    counts_sas = {}
    counts_sa = {}


    for i_samp in range(n_samples):
        # TODO: parallel?
        # set new start position for the players
        env.blue_player._choose_random_position()
        env.red_player._choose_random_position()

        # get observation
        initial_state_blue: State = env.get_observation_for_blue()
        initial_state_red: State = env.get_observation_for_red()

        blue_decision_maker.set_initial_state(initial_state_blue)
        red_decision_maker.set_initial_state(initial_state_red)

        # check of the start state is terminal
        if env.check_terminal():
            continue

        ##### Blue's turn! #####
        action_blue: AgentAction = blue_decision_maker.get_action(initial_state_blue)
        env.blue_player.action(action_blue)  # take the action!

        ##### Red's turn! #####
        action_red: AgentAction = red_decision_maker.get_action(initial_state_red)
        env.red_player.action(action_red)  # take the action!

        ##### Next state - get new observation
        new_observation_for_blue: State = env.get_observation_for_blue()
        new_observation_for_red: State = env.get_observation_for_red()

        # handle reward
        reward_step_blue, reward_step_red = env.handle_reward()


        blue_x = initial_state_blue.my_pos._x
        blue_y = initial_state_blue.my_pos._y
        red_x = initial_state_red.my_pos._x
        red_y = initial_state_red.my_pos._y
        blue_a = int(action_blue) - 1
        red_a = int(action_red) - 1

        blue_x_next = new_observation_for_blue.my_pos._x
        blue_y_next = new_observation_for_blue.my_pos._y

        # Blue's State-transition:
        update_count(counts_sas, (blue_x, blue_y, blue_a, blue_x_next, blue_y_next))


        # TODO: update our model estimation P(s,a,s') = N(s,a,s')/N(s,s'), do this both ways (switch red and blue)
        pass


if __name__ == '__main__':
    start_time = timeit.default_timer()

    n_samples = int(1e5)
    collect_data(n_samples)

    time_str = time.strftime("%H hours, %M minutes and %S seconds",
                             time.gmtime(timeit.default_timer() - start_time))
    print(f'Collected 2 X {n_samples} samples in  {time_str}')