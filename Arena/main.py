import time
import timeit
from matplotlib import style

from Arena.CState import State
from Arena.Entity import Entity
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.Environment import Environment, Episode
from Arena.constants import *
from misharonDecisionMaker import misharonDecisionMaker
from misharonHillDecisionMaker import misharonHillDecisionMaker

style.use("ggplot")


# MAIN:
if __name__ == '__main__':

    start_time = timeit.default_timer()

    env = Environment()

    # blue_decision_maker = misharonDecisionMaker()  # use our agent
    blue_decision_maker = misharonHillDecisionMaker()  # use our agent for the King of the Hill

    # red_decision_maker = RafaelDecisionMaker(EASY_AGENT)
    # red_decision_maker = RafaelDecisionMaker(MEDIUM_AGENT)
    red_decision_maker = RafaelDecisionMaker(HARD_AGENT)

    env.blue_player = Entity(blue_decision_maker)
    env.red_player = Entity(red_decision_maker)

    for episode in range(1, NUM_OF_EPISODES + 1):

        current_episode = Episode(episode)

        # set new start position for the players
        env.blue_player._choose_random_position()
        env.red_player._choose_random_position()

        # get observation
        initial_state_blue: State = env.get_observation_for_blue()
        initial_state_red: State = env.get_observation_for_red()

        # initialize the decision_makers for the players
        blue_decision_maker.set_initial_state(initial_state_blue)
        red_decision_maker.set_initial_state(initial_state_red)

        steps_current_game = 0
        for steps_current_game in range(1, MAX_STEPS_PER_EPISODE + 1):

            env.number_of_steps += 1

            # get observation
            observation_for_blue: State = env.get_observation_for_blue()
            observation_for_red: State = env.get_observation_for_red()

            # check of the start state is terminal
            current_episode.is_terminal = env.check_terminal()
            if current_episode.is_terminal:
                env.tie_count += 1
                env.starts_at_win += 1
                current_episode.episode_reward_blue = 0
                break

            ##### Blue's turn! #####
            action: AgentAction = blue_decision_maker.get_action(observation_for_blue)
            env.blue_player.action(action)  # take the action!

            ##### Red's turn! #####
            action: AgentAction = red_decision_maker.get_action(observation_for_red)
            env.red_player.action(action) # take the action!

            # get new observation
            new_observation_for_blue: State = env.get_observation_for_blue()
            new_observation_for_red: State = env.get_observation_for_red()

            # handle reward
            reward_step_blue, reward_step_red = env.handle_reward()
            current_episode.episode_reward_blue = reward_step_blue

            # update Q-table blue
            blue_decision_maker.update_context(new_observation_for_blue,
                                               reward_step_blue,
                                               current_episode.is_terminal)

            # update Q-table
            red_decision_maker.update_context(new_observation_for_red,
                                              reward_step_red,
                                              current_episode.is_terminal)

            # check terminal
            current_episode.is_terminal = env.check_terminal()

            current_episode.print_episode(env, steps_current_game)
            if current_episode.is_terminal:
                env.update_win_counters()
                break

        if steps_current_game == MAX_STEPS_PER_EPISODE:
            # if we exited the loop because we reached MAX_STEPS_PER_EPISODE
            env.update_win_counters()
            current_episode.is_terminal = True

        print(f'Episode {episode+1}, Blue Wins: {env.wins_for_blue}')

        if env.wins_for_blue > WINS_FOR_FLAG:
            print('Rafael@Technion:beware')


        # for statistics
        env.episodes_rewards.append(current_episode.episode_reward_blue)
        env.steps_per_episode.append(steps_current_game)

    env.end_run()
    time_str = time.strftime("%H hours, %M minutes and %S seconds",
                             time.gmtime(timeit.default_timer() - start_time))
    print('-' * 20, '\nFinished the entire run in ', time_str)
