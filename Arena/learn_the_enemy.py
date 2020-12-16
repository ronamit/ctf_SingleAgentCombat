
import time
import timeit
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import EASY_AGENT, MEDIUM_AGENT, HARD_AGENT, AgentAction, SIZE_X, SIZE_Y, DSM
from Arena.Entity import Entity
from Arena.CState import State
import numpy as np

n_actions = 9
import pickle

#------------------------------------------------------------------------------------------------------------~



def valid_pos_generator(init_pos=(0, 0)):
    for x in range(init_pos[0], SIZE_X):
        for y in range(init_pos[1], SIZE_Y):
            if DSM[x, y] == 1:  # obstacle
                continue
            yield x, y
        # end for
    # end for
# end for
#------------------------------------------------------------------------------------------------------------~


def update_pol_cnts(state, action, policy_counts):
    if state not in policy_counts:
        policy_counts[state] = np.zeros(n_actions, dtype=np.uint)
    policy_counts[state][action] += 1
# end def
# ------------------------------------------------------------------------------------------------------------~


def learn_agent(agent_name, n_samples = 20):

    if agent_name == 'easy':
        decision_maker = RafaelDecisionMaker(EASY_AGENT)
    elif agent_name == 'medium':
        decision_maker = RafaelDecisionMaker(MEDIUM_AGENT)
    elif agent_name == 'hard':
        decision_maker = RafaelDecisionMaker(HARD_AGENT)
    else:
        raise ValueError

    env = Environment()

    blue_decision_maker = decision_maker   # the agent we want to learn its behaviour policy
    red_decision_maker = decision_maker    # the imaginary  agent we want to learn its behaviour policy


    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)

    policy_counts = {}

    # go over each states = (blue position, red position)
    for blue_pos in valid_pos_generator():
        for red_pos in valid_pos_generator(blue_pos):
            if blue_pos == red_pos:
                continue # terminal state
            state_blue = blue_pos + red_pos  # state_blue=(blue_pos, red_pos)
            state_red = red_pos + blue_pos # state_red=(red_pos, blue_pos) , red see the state the other way around

            # set  position for the players
            env.blue_player.x = state_blue[0]
            env.blue_player.y = state_blue[1]
            env.red_player.x = state_red[0]
            env.red_player.y = state_red[1]

            # get observation
            observation_for_blue: State = env.get_observation_for_blue()
            observation_for_red: State = env.get_observation_for_red()

            # the agents are not deterministic, so we want to find the distribution p(a|s)
            for i_samp in range(n_samples):
                # get the action chosen by each player
                action_blue = blue_decision_maker.get_action(observation_for_blue) - 1  #  change to 0-based index
                action_red = red_decision_maker.get_action(observation_for_red) - 1  # change to 0-based index
                update_pol_cnts(state_blue, action_blue, policy_counts)
                update_pol_cnts(state_red, action_red, policy_counts)
            # end for
        # end for
    # end for
    with open(f'learned_{agent_name}_enemy', 'wb') as myfile:
        pickle.dump([agent_name, policy_counts, n_samples], myfile)
    return policy_counts
# end def

if __name__ == '__main__':
    start_time = timeit.default_timer()

    agent_name = 'hard'
    n_samples = 100
    learn_agent(agent_name, n_samples)

    time_str = time.strftime("%H hours, %M minutes and %S seconds",
                             time.gmtime(timeit.default_timer() - start_time))

