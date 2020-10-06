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

def ddpg(agent, episodes, step, pretrained, display_batch_size, save_cyto_data=True, NO_ACTION=False):
    if save_cyto_data:
        cyto_data = np.zeros((20, 10000, episodes))
        action_data = np.zeros((11,10000,episodes))
    total_time = time.time()
    reward_list = []
    random_explore = False
    TESTING = False
    noise = True
    problem_solved = False
    cytoMax = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0])
    start = time.time()
    running_score = 0
    score = 0
    Q_score = 0
    simulation_time = 0
    step_total = 0
    death_count = 0
    heal_count = 0
    transient_infection_count = 0
    timeout_count = 0
    oxydef_total = 0
    # print("\n\n\nTHIS VERSION LEARNS WHILE PLAYING\n\n\n")

    for current_episode in range(episodes):
        if pretrained:
            TESTING = True
        if ID == 1:
            print(current_episode, end="\r")
        # env.seed(current_episode)
        state = env.reset(OH=OH, IS=IS, NRI=NRI, NIR=NIR, injNum=injNum, seed = current_episode)
        current_step = 1
        output_range = None
        score = 0
        for _ in range(step):
            simulation_start = time.time()
            # print(env.current_step)
            # print(env.cytokine_history[:,env.current_step-1])
            # env.render()
            # state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            if random_explore:
                action = env.action_space.sample()
                action = tf.expand_dims(action, axis=0)
            else:
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
                oxydef_total += info["oxydef"]
                if info["dead"]:
                    death_count += 1
                if info["healed"]:
                    heal_count += 1
                if info["transient_infection"]:
                    transient_infection_count += 1
                if info["timeout"] or current_step == step:
                    timeout_count += 1
                # print(death_count, heal_count, timeout_count)
                if current_episode % display_batch_size==0:
                    display_divisor = display_batch_size
                    temp_score = 0
                    temp_step = 0
                    if TESTING:
                        # print("TESTING -- TESTING -- TESTING -- TESTING")
                        score = running_score
                        # temp_score = running_score
                        # temp_step = step_total
                        # step_total = current_step
                        # display_divisor = NUM_TEST_EPS
                        # display_divisor = 1
                    else:
                        score = running_score
                    # print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
                    # print('Episode: {:4.0f} | Steps: {:4.0f} | Avg Reward last {} episodes: {:5.2f} | Avg Time last {} episodes: {:.2f} Seconds'.format(current_episode, step_total/display_divisor, display_divisor, score/display_divisor, display_divisor, (time.time() - start)/display_divisor))
                    # print("Avg Times last {} - Selecting: {:3.2f}, Training: {:3.2f}, Updating: {:3.2f}, Simulating: {:3.2f}".format(display_divisor, agent.selecting_time/display_divisor, agent.training_time/display_divisor, agent.updating_time/display_divisor, simulation_time/display_divisor))
                    # print("LOWS:  {:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(*output_range[0,:]))
                    # print("HIGHS: {:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(*output_range[1,:]))
                    # print("Avg Rates last {} - Mortality Rate: {:2.1f}%, Healing Rate {:2.1f}%, Timeout Rate {:2.1f}%. Final Oxydef: {:4.1f}".format(display_divisor, (death_count/display_divisor)*100, 100*(heal_count/display_divisor), 100*(timeout_count/display_divisor), oxydef_total/display_divisor))
                    # if transient_infection_count > 0:
                    #     print("Num Reinfections: {:2.0f}".format(transient_infection_count))
                    # print("Real reward: {:4.0f}, Q_score: {:4.0f}, Difference: {:4.0f}".format(score, Q_score, score-Q_score))
                    # print()
                    running_score = temp_score
                    Q_score = 0
                    start = time.time()
                    simulation_time = 0
                    death_count = 0
                    heal_count = 0
                    transient_infection_count = 0
                    timeout_count = 0
                    oxydef_total = 0
                    step_total = temp_step

                gc.collect()
                break

    print("Done Sweep")
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
injNum_arr = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

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

# EXPECT 40 cores
comm = MPI.COMM_WORLD
ID = comm.Get_rank()

NIR_ind = math.floor(ID/10)
injNum_ind = ID%10 #0-99
# injNum_ind = math.floor(temp_ID/10)
# NRI_ind = temp_ID%10

for param_num in range(1000):
    OH_ind = math.floor(param_num/100)
    NRI_ind = math.floor(param_num/10)
    IS_ind = param_num%10

    NIR = NIR_arr[NIR_ind]
    injNum = injNum_arr[injNum_ind]
    NRI = NRI_arr[NRI_ind]
    OH = OH_arr[OH_ind]
    IS = IS_arr[IS_ind]
    # print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
    scores = ddpg(None, episodes=100, step=AGENT_MAX_STEPS, pretrained=False, display_batch_size=NUM_TEST_EPS, NO_ACTION=False)

print("Total Time Taken: {:4.2f} minutes".format((time.time()-total_time)/60))
