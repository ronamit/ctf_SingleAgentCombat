

import pickle
from learn_the_enemy import valid_pos_generator, n_actions, set_env_state
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
#------------------------------------------------------------------------------------------------------------~

def is_terminal_state(state):
    set_env_state(env, state)
    return env.check_terminal()

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------~

def get_reward(state):
    set_env_state(env, state)
    reward_blue, reward_red = env.handle_reward()
    return reward_blue

# ------------------------------------------------------------------------------------------------------------~
#
def get_next_pos(pos, action):
    dummy_blue.x = pos[0]
    dummy_blue.y = pos[1]
    dummy_blue.action(action)
    return (dummy_blue.x, dummy_blue.y)


# ------------------------------------------------------------------------------------------------------------~
def max_a_Qprev(Qprev, s):
    out = -np.infty
    for a in range(n_actions):
        s_a = s + (a,)  # concatenate the state-action pair
        out = max(out, Qprev[s_a])
    # end for
    return out
#end def

#------------------------------------------------------------------------------------------------------------~
enemy_name = 'hard'  # 'easy' | 'medium' | 'hard'

with open(f'learned_{enemy_name}_enemy', 'rb') as myfile:
    _, enemy_policy_counts, n_samples  = pickle.load(myfile)


n_iter = int(1e3)
converge_epsilon = 1e-5

# set discount factor in [0,1]
#  Heuristic - at  t=(MAX_STEPS_PER_EPISODE-1) the discounted return accumulates (MOVE_PENALTY/WIN_REWARD) of the reward
gamma = (MOVE_PENALTY / WIN_REWARD)**(1/(MAX_STEPS_PER_EPISODE-1))

# Note: to get the optimal time-dependent policy we need to solve a finete-horizon problem and learn Q_t for each t in the horizon
# BUT, our desicion maker seems to only depend on the state and not on the time ( get_action(self, state: State)-> AgentAction:)

# init Q:
Q_prev = {}
for state_action in state_action_generator():
    Q_prev[state_action] = 0

Q = {}

# Q-value Iteration Algorithm
for i_iter in range(n_iter):
    max_diff = 0
    for state_action in state_action_generator():
        # unpack
        blue_pos = state_action[0:2]
        red_pos = state_action[2:4]
        a_blue = state_action[4]  # blue's action
        state = state_action[:4]

        # get immediate reward
        reward = get_reward(state)

        if is_terminal_state(state):
            Q[state_action] = reward
            continue
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
                val_next += enemy_action_probs[a_red] * max_a_Qprev(Q_prev, next_state)
            # end for
            Q[state_action] = reward + gamma * val_next
        # end if
        max_diff = max(max_diff, np.abs(Q[state_action] - Q_prev[state_action]))
    # end for
    if max_diff <= converge_epsilon:
        break
# end for