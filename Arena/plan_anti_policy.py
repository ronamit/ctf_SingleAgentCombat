
import time
import timeit
import pickle
from learn_the_enemy import valid_pos_generator, n_actions, set_env_state, is_terminal_state, state_generator
import numpy as np
from Arena.Environment import Environment
from Arena.Entity import Entity
from RafaelPlayer.RafaelDecisionMaker import RafaelDecisionMaker
from Arena.constants import WIN_REWARD, MOVE_PENALTY, MAX_STEPS_PER_EPISODE, HARD_AGENT

# define dummy players, just so we can use the class functions
dummy_blue = Entity(RafaelDecisionMaker(HARD_AGENT))
dummy_red = Entity(RafaelDecisionMaker(HARD_AGENT))
env = Environment()
env.blue_player = dummy_blue
env.red_player = dummy_red

#------------------------------------------------------------------------------------------------------------~

def state_action_generator():
    for blue_pos in valid_pos_generator():
        for red_pos in valid_pos_generator():
            for a in range(n_actions):
                yield blue_pos + red_pos + (a,)   # concatenate

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------~

def get_reward(env, state):
    set_env_state(env, state)
    reward_blue, reward_red = env.handle_reward()
    return reward_blue

# ------------------------------------------------------------------------------------------------------------~
#
def get_next_pos(pos, a):
    dummy_blue.x = pos[0]
    dummy_blue.y = pos[1]
    action = a + 1 # remember to change to 1-based index
    dummy_blue.action(action)
    next_pos = (dummy_blue.x, dummy_blue.y)
    return next_pos


# ------------------------------------------------------------------------------------------------------------~
def max_over_a_of_Qsa(Q, s):
    out = -np.infty
    for a in range(n_actions):
        s_a = s + (a,)  # concatenate the state-action pair
        out = max(out, Q[s_a])
    # end for
    return out
#end def

#------------------------------------------------------------------------------------------------------------~


def plan_anti_policy(enemy_name, n_iter, converge_epsilon):

    with open(f'learned_{enemy_name}_enemy', 'rb') as myfile:
        _, enemy_policy_counts, n_samples  = pickle.load(myfile)

    # set discount factor in [0,1]
    #  Heuristic - at  t=(MAX_STEPS_PER_EPISODE-1) the discounted return accumulates (MOVE_PENALTY/WIN_REWARD) of the reward
    gamma = (MOVE_PENALTY / WIN_REWARD)**(1/(MAX_STEPS_PER_EPISODE-1))

    # Note: to get the optimal time-dependent policy we need to solve a finete-horizon problem and learn Q_t for each t in the horizon
    # BUT, our desicion maker seems to only depend on the state and not on the time ( get_action(self, state: State)-> AgentAction:)

    # init Q function:
    qFunc = {}
    for state_action in state_action_generator():
        # use a random number to create some diversity
        qFunc[state_action] = np.random.uniform()
    # end for

    # Q-value Iteration Algorithm (with async updates)
    for i_iter in range(n_iter):
        max_diff = 0
        for state_action in state_action_generator():
            # unpack
            blue_pos = state_action[0:2]
            red_pos = state_action[2:4]
            a_blue = state_action[4]  # blue's action
            state = state_action[:4]

            reward = get_reward(env, state)   # get immediate reward

            if is_terminal_state(env, state):
                new_Q = reward
            else:
                # Bellman update
                # we need to go over all possible next states and weight by their probability
                enemy_action_probs = enemy_policy_counts[state] / n_samples
                # the next state is composed of pos_blue_next = f(pos_blue,a_blue),
                # and  pos_red_next = f(pos_blue,a_red),  where a_red is the random enemy action
                # so we need to sum all the possibilities for a_red, weighted by enemy_action_prob
                val_next = 0
                next_pos_blue = get_next_pos(blue_pos, a_blue)
                for a_red in range(n_actions):
                    next_pos_red = get_next_pos(red_pos, a_red)
                    next_state = next_pos_blue + next_pos_red
                    val_next += enemy_action_probs[a_red] * max_over_a_of_Qsa(qFunc, next_state)
                # end for
                new_Q = reward + gamma * val_next
            # end if
            delta_Q = np.abs(qFunc[state_action] - new_Q)
            qFunc[state_action] = new_Q
            max_diff = max(max_diff, delta_Q)
        # end for
        print(f'Finished iter {i_iter}/{n_iter}, max_diff = {max_diff}')
        if max_diff <= converge_epsilon:
            break
    # end for

    my_policy = {}
    # derive optimal policy and save to file
    for state in state_generator():
        if not is_terminal_state(env, state):
            qVals = [qFunc[state + (a,)] for a in range(n_actions)]
            my_policy[state] = np.argmax(qVals)
        # end if
    # end for
    with open(f'anti_policy_vs_{enemy_name}_enemy', 'wb') as myfile:
        pickle.dump([enemy_name, my_policy, n_iter, converge_epsilon], myfile)
    return my_policy
# end def
#------------------------------------------------------------------------------------------------------------~
if __name__ == '__main__':
    start_time = timeit.default_timer()

    for enemy_name in ['easy', 'medium', 'hard']:

        print('-'*20, '\n Plan anti policy to the ', enemy_name, ' agent ....')

        n_iter = int(1e2)
        converge_epsilon = 2e-3
        anti_policy = plan_anti_policy(enemy_name, n_iter, converge_epsilon)

        time_str = time.strftime("%H hours, %M minutes and %S seconds",
                                 time.gmtime(timeit.default_timer() - start_time))
        print('-'*20, '\nFinished learning the ', enemy_name, ' agent in ', time_str)
