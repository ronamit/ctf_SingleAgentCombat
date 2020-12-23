from Arena.constants import EASY_AGENT, MEDIUM_AGENT, HARD_AGENT, AgentAction, SIZE_X, SIZE_Y, DSM

n_actions = 9
import numpy as np
#------------------------------------------------------------------------------------------------------------~
from Arena.Environment import Environment
from Arena.Entity import Entity
from Arena.helper_funcs import is_dominating
#------------------------------------------------------------------------------------------------------------~

def valid_pos_generator():
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            if DSM[x, y] == 1:  # obstacle
                continue
            yield x, y
        # end for
    # end for
# end ded
#------------------------------------------------------------------------------------------------------------~

def state_generator():
    # note: to get all states  -go over also the state with swapped blue-red roles
    for blue_pos in valid_pos_generator():
        for red_pos in valid_pos_generator():
            yield blue_pos + red_pos   # concatenate  state_blue=(blue_pos, red_pos)
# end for
#------------------------------------------------------------------------------------------------------------~

def state_action_generator():
    for blue_pos in valid_pos_generator():
        for red_pos in valid_pos_generator():
            for a in range(n_actions):
                yield blue_pos + red_pos + (a,)   # concatenate
#------------------------------------------------------------------------------------------------------------~


def set_env_state(env, state):
    # set  position of the players in the environment
    # state = (blue_pos, red_pos)
    env.blue_player.x = state[0]
    env.blue_player.y = state[1]
    env.red_player.x = state[2]
    env.red_player.y = state[3]

#------------------------------------------------------------------------------------------------------------~
def get_players(state):
    blue_player = Entity()
    blue_player.x = state[0]
    blue_player.y = state[1]
    red_player = Entity()
    red_player.x = state[2]
    red_player.y = state[3]
    return blue_player, red_player

#------------------------------------------------------------------------------------------------------------~
def is_terminal_state(state):
    blue_player, red_player = get_players(state)
    flag_red_on_blue, _ = is_dominating(red_player, blue_player)
    flag_blue_on_red, _ = is_dominating(blue_player, red_player)
    if blue_player.x == red_player.x and blue_player.y == red_player.y:
        return True
    elif flag_red_on_blue:
        return True
    elif flag_blue_on_red:
        return True
    return False
    # set_env_state(env, state)
    # return env.check_terminal()

#------------------------------------------------------------------------------------------------------------~
def get_Q_vals(qFunc, s):
    # returns array of Q(s,a) values for  all a
    return [qFunc[s + (a,)] for a in range(n_actions)]
#end def
#------------------------------------------------------------------------------------------------------------~

def derive_greedy_policy(qFunc):
    my_policy = {}
    for state in state_generator():
        if not is_terminal_state(state):
            my_policy[state] = np.argmax(get_Q_vals(qFunc, state))
        # end if
    # end for
    return my_policy
    # end def

# ------------------------------------------------------------------------------------------------------------~

def update_Q_matrix(_Q_matrix, new_state):
        # this is needed for main.py to run:
        _Q_matrix[(new_state.my_pos._x, new_state.my_pos._y), \
                       (new_state.enemy_pos._x, new_state.enemy_pos._y)] \
            = list(np.zeros(9))
# end def

# ------------------------------------------------------------------------------------------------------------~
def init_Q_matrix():
    # this is needed for main.py to run
    _Q_matrix = dict()
    for x1 in range(SIZE_X):
        for y1 in range(SIZE_Y):
            for x2 in range(SIZE_X):
                for y2 in range(SIZE_Y):
                    _Q_matrix[(x1, y1), (x2, y2)] = list(np.ones(9))  # blue_pos x red_pos x actions
    return _Q_matrix
