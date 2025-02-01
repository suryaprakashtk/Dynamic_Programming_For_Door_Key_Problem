from utils import *
from parta import *
from partb import *

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def doorkey_problem(env,info):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """

    mdp_obj = MDP_a(env,info)
    mdp_obj.init_fuction()
    last_time, total_cost = mdp_obj.dp()
    optim_act_seq = mdp_obj.get_sequence(last_time)
    return optim_act_seq


def partA():
    global_path = "/Users/harishprasathtk/Desktop/UCSD_MS/SP23/ECE276B_Planning_for_Robots/ECE276BPR1/starter_code"
    # Uncomment the appropriate variable to run the corresponding environment 
    # env_path = global_path + "/envs/known_envs/doorkey-5x5-normal.env"
    # env_path = global_path + "/envs/known_envs/doorkey-6x6-normal.env"
    # env_path = global_path + "/envs/known_envs/doorkey-8x8-normal.env"

    # env_path = global_path + "/envs/known_envs/doorkey-6x6-direct.env"
    # env_path = global_path + "/envs/known_envs/doorkey-8x8-direct.env"

    # env_path = global_path + "/envs/known_envs/doorkey-6x6-shortcut.env"
    env_path = global_path + "/envs/known_envs/doorkey-8x8-shortcut.env"
    env, info = load_env(env_path)  # load an environment
    seq = doorkey_problem(env,info)  # find the optimal action sequence

    draw_gif_from_seq(seq, load_env(env_path)[0],"/gif/doorkey-8x8-shortcut.gif")  # draw a GIF & save


def partB():
    
    # Calcutates Optimal policies without the environment input
    mdp_obj = MDP_b()
    mdp_obj.init_fuction()
    mdp_obj.dp()
    
    # Finding sequence for all maps at once
    for i in range(1,37):
        global_path = "/Users/harishprasathtk/Desktop/UCSD_MS/SP23/ECE276B_Planning_for_Robots/ECE276BPR1/starter_code"
        env_folder = global_path + "/envs/random_envs"
        env, info, env_path = load_random_env(env_folder,str(i))
        prior = mdp_obj.get_prior(env,info)
        optim_act_seq = mdp_obj.get_sequence(prior)

        # name = "doorkey-8x8-" + str(i) + "_start"
        # my_plot_env(env,name)
        # for act in optim_act_seq:
        #     step(env, act)
        # name2 = "doorkey-8x8-" + str(i) + "_end"
        # my_plot_env(env,name2)

        print(optim_act_seq)
        draw_gif_from_seq(optim_act_seq, load_env(env_path)[0],"/gif/random/doorkey-8x8-" + str(i) + ".gif") 


if __name__ == "__main__":
    partA()
    # partB()

