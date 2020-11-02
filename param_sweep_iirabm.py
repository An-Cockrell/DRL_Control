import os
import gym
import sys

import numpy as np
import copy
import random
from collections import namedtuple, deque
import time
import gc
import math

from mpi4py import MPI

from iirabm_env import Iirabm_Environment

total_time = time.time()

def ddpg(agent, episodes, step, pretrained, display_batch_size, save_cyto_data=True, NO_ACTION=False):
    if save_cyto_data:
        cyto_data = np.zeros((20, 10000, episodes))
        action_data = np.zeros((11,10000,episodes))
    total_time = time.time()
    reward_list = []
    TESTING = False
    problem_solved = False
    start = time.time()
    running_score = 0
    score = 0
    simulation_time = 0
    step_total = 0
    # print("\n\n\nTHIS VERSION LEARNS WHILE PLAYING\n\n\n")

    for current_episode in range(episodes):
        if pretrained:
            TESTING = True
        if ID == 1:
            print(current_episode, end=" ")
        # env.seed(current_episode)
        state = env.reset(OH=OH, IS=IS, NRI=NRI, NIR=NIR, injNum=injNum, seed = current_episode)
        current_step = 1
        output_range = None
        score = 0
        for _ in range(step):
            simulation_start = time.time()

            action = np.expand_dims(np.array([0,0,0,0,0,0,0,0,0,0,0]),0)

            next_state, reward, done, info = env.step(action[0])
            current_step += 1
            step_total += 1

            score += reward
            simulation_time += time.time() - simulation_start

            if current_step == step:
                done = True
            if done:
                if save_cyto_data:
                    cyto_output = env.full_history[:, env.cytokine_history[0,:] >0]
                    cyto_data[:,:cyto_output.shape[1],current_episode] = cyto_output
                    action_data[:,:cyto_output.shape[1],current_episode] = env.action_history[:,:cyto_output.shape[1]]
                    cyto_data[:,cyto_output.shape[1]:,current_episode] = float('nan')
                    action_data[:,cyto_output.shape[1]:,current_episode] = float('nan')

                running_score += score
                reward_list.append(score)
                gc.collect()
                break

    # print("Done Sweep")
    np.save("ParamSweep/cytokine_data_{}_{}.npy".format(ID, param_num), cyto_data[:,:,:current_episode +1])
    return reward_list

# TRAINING TIME

BUFFER_SIZE = 1000000      # replay buffer size
GAMMA = 0.99               # discount factor
TAU = 0.001                # for soft update of target parameters
LR_ACTOR = 0.0001          # learning rate of the actor
LR_CRITIC = 0.001          # learning rate of the critic
WEIGHT_DECAY = 0.001       # L2 weight decay
BATCH_SIZE = 64        # minibatch size
STARTING_NOISE_MAG = .2  #initial exploration noise magnitude
EPS_BETWEEN_EXP_UPDATE = 500 #episodes inbetween exploration update

WIN_THRESHOLD = 999999

EPS_BETWEEN_TEST = 200
NUM_TEST_EPS = 100
ENV_STEPS = 8000
AGENT_ACTION_REPEATS = 4
AGENT_MAX_STEPS = math.floor(ENV_STEPS/AGENT_ACTION_REPEATS)
env = Iirabm_Environment(rendering=None, action_repeats=AGENT_ACTION_REPEATS, ENV_MAX_STEPS=ENV_STEPS, action_L1=0.1, potential_difference_mult=200, phi_mult = 100)
# env = gym.make("LunarLanderContinuous-v2")  # Create the environment

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

OH=.08 #.05-.15
OH_arr = [.05, .06, .07, .08, .09, .10, .11, .12, .13, .14]
IS=4 # 1-10
IS_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NRI=2 #1-10
NRI_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NIR=2 #1-4
NIR_arr = [1, 2, 3, 4]
injNum=27 #25-35
injNum_arr = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# # EXPECT 400 cores
# comm = MPI.COMM_WORLD
# ID = comm.Get_rank()
#
# NIR_ind = math.floor(ID/100)
# temp_ID = ID%100 #0-99
# injNum_ind = math.floor(temp_ID/10)
# NRI_ind = temp_ID%10
#
# for param_num in range(100):
#     OH_ind = math.floor(param_num/10)
#     IS_ind = param_num%10
#
#     NIR = NIR_arr[NIR_ind]
#     injNum = injNum_arr[injNum_ind]
#     NRI = NRI_arr[NRI_ind]
#     OH = OH_arr[OH_ind]
#     IS = IS_arr[IS_ind]
#     # print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
#     scores = ddpg(None, episodes=100, step=AGENT_MAX_STEPS, pretrained=False, display_batch_size=NUM_TEST_EPS, NO_ACTION=False)

# # EXPECT 40 cores
# comm = MPI.COMM_WORLD
# ID = comm.Get_rank()
#
# NIR_ind = math.floor(ID/10)
# injNum_ind = ID%10 #0-99
#
# for param_num in range(1000):
#     OH_ind = math.floor(param_num/100)          #0-9
#     NRI_ind = math.floor((param_num%100)/10)    #0-99/10
#     IS_ind = param_num%10
#
#     NIR = NIR_arr[NIR_ind]
#     injNum = injNum_arr[injNum_ind]
#     NRI = NRI_arr[NRI_ind]
#     OH = OH_arr[OH_ind]
#     IS = IS_arr[IS_ind]
#     # print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
#     scores = ddpg(None, episodes=100, step=AGENT_MAX_STEPS, pretrained=False, display_batch_size=NUM_TEST_EPS, NO_ACTION=False)

# """ ---------- USE THIS ONE UNTIL RECURRENT INJURY BUG IS FIXED ---------- """
# # EXPECT 40 cores
# comm = MPI.COMM_WORLD
# ID = comm.Get_rank()
#
# NIR_ind = math.floor(ID/10)
# injNum_ind = ID%10 #0-99
#
# for param_num in range(100):
#     OH_ind = math.floor(param_num/10)          #0-9
#     IS_ind = param_num%10
#
#     NIR = NIR_arr[NIR_ind]
#     injNum = injNum_arr[injNum_ind]
#     NRI = 0
#     OH = OH_arr[OH_ind]
#     IS = IS_arr[IS_ind]
#     # print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
#     scores = ddpg(None, episodes=100, step=AGENT_MAX_STEPS, pretrained=False, display_batch_size=NUM_TEST_EPS, NO_ACTION=False)

""" ---------- USE THIS ONE UNTIL RECURRENT INJURY BUG IS FIXED ---------- """
# EXPECT 400 cores
comm = MPI.COMM_WORLD
ID = comm.Get_rank()

NIR_ind = math.floor(ID/100)
temp_ID = ID%100 #0-99
injNum_ind = math.floor(temp_ID/10)
OH_ind = temp_ID%10

for param_num in range(10):
    IS_ind = param_num%10

    NIR = NIR_arr[NIR_ind]
    injNum = injNum_arr[injNum_ind]
    NRI = 0
    OH = OH_arr[OH_ind]
    IS = IS_arr[IS_ind]
    # print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
    scores = ddpg(None, episodes=100, step=AGENT_MAX_STEPS, pretrained=False, display_batch_size=NUM_TEST_EPS, NO_ACTION=False)

print("Total Time Taken: {:4.2f} minutes".format((time.time()-total_time)/60))
