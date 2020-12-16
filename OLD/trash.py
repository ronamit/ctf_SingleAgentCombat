# set discount factor
#  Heuristic - at  t=(MAX_STEPS_PER_EPISODE-1) the discounted return accumulates (MOVE_PENALTY/WIN_REWARD) of the reward
gamma = (MOVE_PENALTY / WIN_REWARD)**(1/(MAX_STEPS_PER_EPISODE-1))
# Note: to get the real solution for the finete-horizon problem we need to learn Q_t for each t in the horizon
