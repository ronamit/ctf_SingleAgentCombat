
from Arena.constants import TIE
import pickle
from learn_the_enemy import valid_pos_generator, n_actions
import numpy as np
from Arena.Environment import Environment
from Arena.Entity import Entity

# define dummy players, just so we can use the classs functions
dummy_blue = Entity(RafaelDecisionMaker(HARD_AGENT))
dummy_red = Entity(RafaelDecisionMaker(HARD_AGENT))
dummy_env = Environment()
#------------------------------------------------------------------------------------------------------------~

def state_action_generator():
    for blue_pos in valid_pos_generator():
        for red_pos in valid_pos_generator():
            for a in range(n_actions):
                yield blue_pos, red_pos, a
#------------------------------------------------------------------------------------------------------------~

def is_terminal_state(state_action):
    blue_pos, red_pos, a = state_action
    return blue_pos == red_pos

# ------------------------------------------------------------------------------------------------------------~

def get_reward(state_action):
    blue_pos, red_pos, a = state_action



# ------------------------------------------------------------------------------------------------------------~

    def check_terminal(self):
        flag_red_on_blue, _ = is_dominating(self.red_player, self.blue_player)
        flag_blue_on_red, _ = is_dominating(self.blue_player, self.red_player)
        if self.blue_player.x == self.red_player.x and self.blue_player.y == self.red_player.y:
            return True
        elif flag_red_on_blue:
            return True
        elif flag_blue_on_red:
            return True
        return False

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
        if is_terminal(state_action):
            Q[state_action] = TIE  # 0 reward
        # Bellman update
        Q[state_action] = Q_prev[state_action]
        max_diff = max(max_diff, np.abs(Q[state_action] - Q_prev[state_action]))
    # end for
    if max_diff <= converge_epsilon:
        break
# end for