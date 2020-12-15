
from Arena.constants import *
from Arena.geometry import LOS, bresenham
import pandas as pd
from copy import deepcopy
import unittest
import os
import pickle
import time



def check_if_LOS(x1,y1,x2,y2):
    """returns True is there are no obstacles between (x1,y1) and (x2,y2)
    otherwise return False"""
    if USE_BRESENHAM_LINE:
        list_of_points = bresenham(x1, y1, x2, y2)
    else:
        list_of_points = LOS(x1,y1,x2,y2)

    points_in_los = []
    for [x,y] in list_of_points:
        if DSM[x,y]==1:
            #print("Hit in: (", {x}, " ,", {y}, ")")
            return False, points_in_los
        else:
            points_in_los.append([x,y])
    return True, points_in_los

def is_dominating(first_player, second_player):
    """dominating point is defined to be a point that:
     1. has LOS to the dominated_point
     2. there IS an action that executing it will end in a point that have no LOS to the second_player
     3. there is NO action for the second_player to take that will end in no LOS to the first_player

     The function will return True if the first player is dominating the second player
     False otherwise"""

    is_los, _ = check_if_LOS(first_player.x, first_player.y, second_player.x, second_player.y)
    if not is_los: #no LOS
        return False, [-1, -1]

    org_cor_first_player_x, org_cor_first_player_y = first_player.get_coordinates()
    org_cor_second_player_x, org_cor_second_player_y = second_player.get_coordinates()

    for a_first in range(1,NUMBER_OF_ACTIONS+1):
        first_player.set_coordinatess(org_cor_first_player_x, org_cor_first_player_y)
        first_player.action(a_first)
        first_player_after_action_x, first_player_after_action_y = first_player.get_coordinates()

        is_los_first_to_second, _ = check_if_LOS(first_player_after_action_x, first_player_after_action_y, org_cor_second_player_x, org_cor_second_player_y)
        if not is_los_first_to_second: #there is an action that executing it will end in a point that have no LOS to the second_player
            for a_second in range(1,NUMBER_OF_ACTIONS+1):

                second_player.set_coordinatess(org_cor_second_player_x, org_cor_second_player_y)
                second_player.action(a_second)
                second_player_after_action_x, second_player_after_action_y = second_player.get_coordinates()

                is_lost_second_to_first, _ = check_if_LOS(org_cor_first_player_x, org_cor_first_player_y, second_player_after_action_x, second_player_after_action_y)
                if not is_lost_second_to_first:
                    first_player.set_coordinatess(org_cor_first_player_x, org_cor_first_player_y)
                    second_player.set_coordinatess(org_cor_second_player_x, org_cor_second_player_y)
                    return False, [-1, -1] #there IS an action for the second player that taking it will end in lost of LOS to the first_player

            first_player.set_coordinatess(org_cor_first_player_x, org_cor_first_player_y)
            second_player.set_coordinatess(org_cor_second_player_x, org_cor_second_player_y)
            return True, [first_player_after_action_x, first_player_after_action_y]  #there is no action for the second_player to take that will end in lost of LOS

    first_player.set_coordinatess(org_cor_first_player_x, org_cor_first_player_y)
    second_player.set_coordinatess(org_cor_second_player_x, org_cor_second_player_y)
    return False, [-1, -1]


#if __name__ == '__main__':
    # DSM = np.array([
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ])

    #point_in_LOS = check_if_LOS(1,1,9,2)
    #point_in_LOS = check_if_LOS(0,0,9,9)
    #

    #
    #
    # blue_player = Blue_Entity()
    # blue_player.x = 1
    # blue_player.y = 3
    #
    # red_player = Red_Entity()
    # red_player.x = 4
    # red_player.y = 2
    #
    #
    # print(is_dominating(blue_player, red_player))
    # while True:
    #     blue_player.print_episode(blue_player, blue_player, red_player, 0, True)

class Test_is_dominating(unittest.TestCase):
    # DSM = np.array([
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 1., 1., 0., 0., 0., 1., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 1., 1., 1., 1., 0., 0.],
    #     [0., 1., 1., 0., 1., 0., 0., 1., 0., 0.],
    #     [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    # ])

    def test_controling1(self):
        blue_player = Blue_Entity()
        blue_player.x = 1
        blue_player.y = 3

        red_player = Red_Entity()
        red_player.x = 4
        red_player.y = 2

        self.assertFalse(is_dominating(blue_player, red_player))
        if not is_dominating(blue_player, red_player):
            print("Pass controlingTest1")
        else:
            print("WTF")

    def test_controling2(self):
        blue_player = Blue_Entity()
        blue_player.x = 2
        blue_player.y = 3

        red_player = Red_Entity()
        red_player.x = 4
        red_player.y = 2

        self.assertTrue(is_dominating(blue_player, red_player))
        if is_dominating(blue_player, red_player):
            print("Pass controlingTest2")