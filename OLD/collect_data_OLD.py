
import time
import timeit
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import HARD_AGENT, AgentAction, SIZE_X, SIZE_Y
from Arena.Entity import Entity
from Arena.CState import State
import numpy as np

n_actions = 9

def update_count(d, k):
    if k in d:
        d[k] += 1
    else:
        d[k] = 1
    # end if
# end def

def update_model_est(initial_state, new_state, action, reward, counts_sas, counts_sa, r_s, pi_s):

    my_x = initial_state.my_pos._x
    my_y = initial_state.my_pos._y

    a = int(action) - 1

    my_pos = (my_x, my_y)
    # if (my_x, my_y) in pi_s.keys():
    #     assert action == pi_s[(my_x, my_y)] # check that enemy is deterministic

    if my_pos not in pi_s:
        pi_s[my_pos] = np.zeros(n_actions, dtype=np.uint)
    pi_s[my_pos][a] += 1





    my_x_next = new_state.my_pos._x
    my_y_next = new_state.my_pos._y
    enemy_x_next = new_state.enemy_pos._x
    enemy_y_next = new_state.enemy_pos._y

    # Count transition s,a,s'
    update_count(counts_sas, (my_x, my_y, enemy_x, enemy_y, a, my_x_next, my_y_next, enemy_x_next, enemy_y_next))

    # Count visitation s,a
    update_count(counts_sa, (my_x, my_y, enemy_x, enemy_y, a))

    # The reward depends on current state only  (deterministic - no need to average)
    r_s[(my_x, my_y, enemy_x, enemy_y)] = reward
# end def

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
    r_s = {}   # The reward depends on current state only  (deterministic - no need to average)

    pi_s = {}

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

        # Update from Blue's perspective:
        update_model_est(initial_state_blue, new_observation_for_blue, action_blue, reward_step_blue, counts_sas,
                         counts_sa, r_s, pi_s)

        # Update from Red's perspective:
        update_model_est(initial_state_red, new_observation_for_red, action_red, reward_step_red, counts_sas,
                         counts_sa, r_s, pi_s)
    # end for
    pass
    # TODO: save result in pickle
    return counts_sas, counts_sa, r_s
# end def

if __name__ == '__main__':
    start_time = timeit.default_timer()

    n_samples = int(1e3)
    counts_sas, counts_sa, r_s = collect_data(n_samples)

    time_str = time.strftime("%H hours, %M minutes and %S seconds",
                             time.gmtime(timeit.default_timer() - start_time))
    print(f'Collected 2 X {n_samples} samples in  {time_str}')

    total_states = (SIZE_X * SIZE_Y)**2
    total_state_actions = (SIZE_X * SIZE_Y) ** 2


    print(f'Visited  {len(r_s.keys())} states out of {total_states}')
    print(f'Visited  {len(counts_sa.keys())} (s,a) pairs out of {total_states * n_actions}')
    print(f'Visited  {len(counts_sas.keys())} (s,a,s\') triplets out of {total_states**2 * n_actions}')
