

from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import HARD_AGENT, AgentAction
from Arena.Entity import Entity
from Arena.CState import State




def collect_data():

    n_samples = int(1e2)

    env = Environment()

    blue_decision_maker = RafaelDecisionMaker(HARD_AGENT)
    red_decision_maker = RafaelDecisionMaker(HARD_AGENT)

    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)
    # initialize the decision_makers for the players




    for i_samp in range(n_samples): # TODO: parallel?


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

        ##### Red's turn! #####
        action_red: AgentAction = red_decision_maker.get_action(initial_state_red)

        # update our model estimation P(s,a,s') = N(s,a,s')/N(s,s'), do this both ways (switch red and blue)



if __name__ == '__main__':
    collect_data()