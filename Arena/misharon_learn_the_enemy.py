
import time
import timeit
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import EASY_AGENT, MEDIUM_AGENT, HARD_AGENT, AgentAction, SIZE_X, SIZE_Y, DSM
from Arena.Entity import Entity
from Arena.CState import State
import numpy as np
import pickle
from misharon_utils import n_actions, state_generator, is_terminal_state, set_env_state
#------------------------------------------------------------------------------------------------------------~


#------------------------------------------------------------------------------------------------------------~


def update_pol_cnts(state, a, policy_counts):
    if state not in policy_counts:
        policy_counts[state] = np.zeros(n_actions, dtype=np.uint)
    policy_counts[state][a] += 1
# end def
# ------------------------------------------------------------------------------------------------------------~


#------------------------------------------------------------------------------------------------------------~

def learn_decision_maker(decision_maker, n_samples = 20, save_to_file=False):


    env = Environment()
    blue_decision_maker = decision_maker   # the agent we want to learn its behaviour policy
    red_decision_maker = decision_maker    # the imaginary  agent we want to learn its behaviour policy
    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)
    policy_counts = {}

    # go over each states = (blue position, red position)
    for state in state_generator():

        if is_terminal_state(state):
            continue

        # set  position of the players in the environment
        set_env_state(env, state)

        # get observation
        observation_for_blue: State = env.get_observation_for_blue()

        # the agents are not deterministic, so we want to find the distribution p(a|s)
        for i_samp in range(n_samples):
            # get the action chosen by each player
            action_blue = blue_decision_maker.get_action(observation_for_blue)
            a = action_blue - 1   # change to 0-based index
            update_pol_cnts(state, a, policy_counts)
        # end for
    # end for

    print('Finished learning the enemy')
    if save_to_file:
        with open(f'learned_{agent_name}_enemy', 'wb') as myfile:
            pickle.dump([agent_name, policy_counts, n_samples], myfile)

    return policy_counts
# end def
#------------------------------------------------------------------------------------------------------------~

if __name__ == '__main__':
    start_time = timeit.default_timer()

    # agent_name = 'hard'  # 'easy' | 'medium' | 'hard'

    for agent_name in ['hard']:

        if agent_name == 'easy':
            decision_maker = RafaelDecisionMaker(EASY_AGENT)
        elif agent_name == 'medium':
            decision_maker = RafaelDecisionMaker(MEDIUM_AGENT)
        elif agent_name == 'hard':
            decision_maker = RafaelDecisionMaker(HARD_AGENT)
        else:
            raise ValueError

        n_samples = 500  # 500 is overkill, but it is OK if this is offline
        # n_samples = 10

        print('-'*20, '\n Learning the ', agent_name, ' agent ....')

        policy_counts = learn_decision_maker(decision_maker, n_samples, save_to_file=True)

        time_str = time.strftime("%H hours, %M minutes and %S seconds",
                                 time.gmtime(timeit.default_timer() - start_time))
        print('-'*20, '\nFinished learning the ', agent_name, ' agent in ', time_str)
