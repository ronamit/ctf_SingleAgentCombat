import matplotlib.pyplot as plt
import numpy as np
from Arena.constants import *
from Arena.helper_funcs import *
from Arena.geometry import LOS, bresenham
import cv2


def print_stats(array_of_results, save_folder_path, plot_every, save_figure=True, steps=False):
    moving_avg = np.convolve(array_of_results, np.ones((plot_every,)) / plot_every, mode='valid')
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.xlabel("episode #")
    if steps:
        plt.axis([0, len(array_of_results), 0, MAX_STEPS_PER_EPISODE])
        plt.suptitle(f"Avg number of steps per episode")
        plt.ylabel(f"steps per episode {SHOW_EVERY}ma")
        if save_figure:
            plt.savefig(save_folder_path + os.path.sep + '#steps')
    else:
        plt.axis([0, len(array_of_results), -WIN_REWARD-50, WIN_REWARD+50])
        plt.suptitle(f"Rewards per episode")
        plt.ylabel(f"Reward {SHOW_EVERY}ma")
        if save_figure:
            plt.savefig(save_folder_path + os.path.sep + 'rewards')

    plt.show()

def print_stats_humna_player(array_of_results, save_folder_path, number_of_episodes, save_figure=True, steps=False, red_player=False):
    moving_avg = np.convolve(array_of_results, np.ones((1,)) / 1, mode='valid')
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.xlabel("episode #")
    if steps:
        plt.axis([0, len(array_of_results), 0, MAX_STEPS_PER_EPISODE])
        if red_player: #number of steps figure for red player
            plt.suptitle(f"avg number of steps per episode red player")
            plt.ylabel(f"steps{number_of_episodes}")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + '#steps red player')
        else: #number of steps figure for blue player
            plt.suptitle(f"avg number of steps per episode blue player")
            plt.ylabel(f"steps{number_of_episodes}")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + '#steps blue player')
    else:
        plt.axis([0, len(array_of_results), -WIN_REWARD, WIN_REWARD])
        if red_player: #reward figure for red player
            plt.suptitle(f"Reward for red player")
            plt.ylabel(f"Rewards")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + 'rewards red player')
        else: #reward figure for blue player
            plt.suptitle(f"Reward for blue player")
            plt.ylabel(f"Rewards")
            if save_figure:
                plt.savefig(save_folder_path + os.path.sep + 'rewards blue player')

    plt.show()



def print_episode_graphics(env, episode, last_step_number):
    #     blue, red, game_number, is_terminal, number_of_steps, wins_for_blue, wins_for_red, tie_count):
    # env, last_step_number

    blue = env.blue_player
    red = env.red_player
    game_number = episode.episode_number
    is_terminal = episode.is_terminal
    number_of_steps = last_step_number
    wins_for_blue = env.wins_for_blue
    wins_for_red = env.wins_for_red
    tie_count = env.tie_count

    const = 50
    env = np.zeros((SIZE_X * const, SIZE_Y * const, 3), dtype=np.uint8)  # starts an RBG of our size

    # paint the tiles in line from blue to red in yellow
    if PRINT_TILES_IN_LOS:
        _, points_in_LOS = check_if_LOS(blue.x, blue.y, red.x, red.y)
        for point in points_in_LOS:
            env[point[0] * const:point[0] * const + const, point[1] * const:point[1] * const + const] = dict_of_colors[
                YELLOW_N]  # sets all the tiles in LOS green

    # paint the blue_player and the red_player
    radius = int(np.ceil(const / 2))
    thickness = -1

    center_cor_blue_x = blue.x * const + radius
    center_cor_blue_y = blue.y * const + radius
    blue_color = dict_of_colors[BLUE_N]
    cv2.circle(env, (center_cor_blue_y, center_cor_blue_x), radius, blue_color, thickness)

    center_cor_red_x = red.x * const + radius
    center_cor_red_y = red.y * const + radius
    red_color = dict_of_colors[RED_N]
    cv2.circle(env, (center_cor_red_y, center_cor_red_x), radius, red_color, thickness)

    # print domination points
    red_over_blue_flag, red_over_blue_point = is_dominating(red, blue)
    blue_over_red_flag, blue_over_red_point = is_dominating(blue, red)
    if is_terminal:
        if red_over_blue_flag:
            env[red_over_blue_point[0] * const:red_over_blue_point[0] * const + const,
            red_over_blue_point[1] * const:red_over_blue_point[1] * const + const] = dict_of_colors[
                DARK_RED_N]  # sets the green location tile to green color
        if blue_over_red_flag:
            env[blue_over_red_point[0] * const:blue_over_red_point[0] * const + const,
            blue_over_red_point[1] * const:blue_over_red_point[1] * const + const] = dict_of_colors[
                DARK_BLUE_N]  # sets the green location tile to green color

    # set the obstacle tile to grey
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            if DSM[x][y] == 1.:
                env[x * const:x * const + const, y * const:y * const + const] = dict_of_colors[GREY_N]

    cv2.waitKey(35)  # delay of refresh
    font = cv2.FONT_HERSHEY_SIMPLEX
    bootomLeftCornerOfText = (5, SIZE_Y * const - 15)
    fontScale = 1
    color = (100, 200, 120)  # greenish
    thickness = 2
    cv2.putText(env, f"episode #{game_number}", bootomLeftCornerOfText, font, fontScale, color, thickness,
                cv2.LINE_AA)

    # print who won
    if is_terminal or number_of_steps==MAX_STEPS_PER_EPISODE:
        thickness = 3
        bootomLeftCornerOfText_steps = (int(np.floor(SIZE_Y / 2)) * const - 79, 55)
        if red_over_blue_flag and blue_over_red_flag:
            bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const - 38, 30)
            cv2.putText(env, f"TIE!", bootomLeftCornerOfText, font, fontScale, dict_of_colors[
                PURPLE_N], thickness,
                        cv2.LINE_AA)
            cv2.putText(env, f"after {number_of_steps} steps", bootomLeftCornerOfText_steps, font, 0.7, dict_of_colors[
                PURPLE_N], 0,
                        cv2.LINE_AA)
        elif red_over_blue_flag:
            bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const - 80, 30)
            cv2.putText(env, f"RED WON!", bootomLeftCornerOfText, font, fontScale, dict_of_colors[
                RED_N], thickness,
                        cv2.LINE_AA)
            cv2.putText(env, f"after {number_of_steps} steps", bootomLeftCornerOfText_steps, font, 0.7, dict_of_colors[
                PURPLE_N], 0,
                        cv2.LINE_AA)
        elif blue_over_red_flag:
            bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const - 90, 30)
            cv2.putText(env, f"BLUE WON!", bootomLeftCornerOfText, font, fontScale, dict_of_colors[
                BLUE_N], thickness,
                        cv2.LINE_AA)
            cv2.putText(env, f"after {number_of_steps} steps", bootomLeftCornerOfText_steps, font, 0.7, dict_of_colors[
                PURPLE_N], 0,
                        cv2.LINE_AA)
        else: #both lost...
            bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const - 90, 30)
            cv2.putText(env, f"both lost...", bootomLeftCornerOfText, font, fontScale, dict_of_colors[
                PURPLE_N], thickness-1,
                        cv2.LINE_AA)
            cv2.putText(env, f"after {number_of_steps} steps", bootomLeftCornerOfText_steps, font, 0.7, dict_of_colors[
                PURPLE_N], 0,
                        cv2.LINE_AA)

    else: #not terminal state
        bootomLeftCornerOfText = (int(np.floor(SIZE_Y / 2)) * const -45, 20)
        cv2.putText(env, f"steps: {number_of_steps}", bootomLeftCornerOfText, font, 0.7, dict_of_colors[
            PURPLE_N], 0,
                    cv2.LINE_AA)

    #print number of wins
    bootomLeftCornerOfText = (5, 20)
    cv2.putText(env, f"Blue won: {wins_for_blue}", bootomLeftCornerOfText, font, 0.7, dict_of_colors[
        BLUE_N], 0,
                cv2.LINE_AA)
    bootomLeftCornerOfText = (5, 40)
    cv2.putText(env, f"Red won: {wins_for_red}", bootomLeftCornerOfText, font, 0.7, dict_of_colors[
        RED_N], 0,
                cv2.LINE_AA)
    bootomLeftCornerOfText = (5, 60)
    cv2.putText(env, f"Tie: {tie_count}", bootomLeftCornerOfText, font, 0.7, dict_of_colors[
        PURPLE_N], 0,
                cv2.LINE_AA)


    cv2.imshow(f"state as ((blue_cor), (red_cor))", env)
    if is_terminal:  # if we reached the end of the episode
        if cv2.waitKey(500) & 0xFF == ord('q'):
            cv2.destroyWindow(f"state as ((blue_cor), (red_cor))")
            time.sleep(2)
            pass
        time.sleep(2)
    else:
        if cv2.waitKey(3) & 0xFF == ord('q'):
            cv2.destroyWindow(f"state as ((blue_cor), (red_cor))")
            pass

