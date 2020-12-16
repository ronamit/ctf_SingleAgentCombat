
from Arena.constants import TIE
import pickle
from learn_the_enemy import valid_pos_generator, n_actions, set_env_state
import numpy as np
from Arena.Environment import Environment
from Arena.Entity import Entity

# define dummy players, just so we can use the classs functions
dummy_blue = Entity(RafaelDecisionMaker(HARD_AGENT))
dummy_red = Entity(RafaelDecisionMaker(HARD_AGENT))
env = Environment()
env.blue_player = Entity(blue_decision_maker)
env.red_player = Entity(red_decision_maker)

#------------------------------------------------------------------------------------------------------------~

def state_action_generator():
    for blue_pos in valid_pos_generator():
        for red_pos in valid_pos_generator():
            for a in range(n_actions):
                yield blue_pos, red_pos, a
#------------------------------------------------------------------------------------------------------------~

def is_terminal_state(state_action):
    blue_pos, red_pos, a = state_action
    state = blue_pos + red_pos
    set_env_state(env, state)
    return env.check_terminal()

# ------------------------------------------------------------------------------------------------------------~

def get_reward(state_action):
    blue_pos, red_pos, a = state_action
    state = blue_pos + red_pos
    set_env_state(env, state)
    reward_blue, reward_red = env.handle_reward()



# ------------------------------------------------------------------------------------------------------------~


#------------------------------------------------------------------------------------------------------------~
enemy_name = 'hard'  # 'easy' | 'medium' | 'hard'

with open(f'learned_{enemy_name}_enemy', 'rb') as myfile:
    _, enemy_policy_counts, _ = pickle.load(myfile)


n_iter = int(1e3)
converge_epsilon = 1e-5

# init Q:
Q_prev = {}
for state_action in state_action_generator():
    Q_prev[state_action] = 0

Q = {}

# Q-value Iteration Algorithm
for i_iter in range(n_iter):
    max_diff = 0
    for state_action in state_action_generator():

        # get immediate reward
        reward = get_reward(state_action)

        if is_terminal_state(state_action):
            Q[state_action] = reward
            continue
        else:
            # Bellman update
            Q[state_action] = Q_prev[state_action]
            max_diff = max(max_diff, np.abs(Q[state_action] - Q_prev[state_action]))
    # end for
    if max_diff <= converge_epsilon:
        break
# end for