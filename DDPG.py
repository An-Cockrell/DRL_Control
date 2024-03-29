"""
This file contains the main function to use an agent and attempt to solve the iirabm_env
"""


# https://github.com/shivaverma/OpenAIGym/tree/master/bipedal-walker/ddpg-torch
import os
import gym

import numpy as np
import copy
import random
from collections import namedtuple, deque
import time
import gc
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from iirabm_env_V2 import Iirabm_Environment
from ddpg_agent import Agent
from ddpg_agent import shallow_tanh

def solve_problem(agent, episodes, step, pretrained, display_batch_size, save_cyto_data=False, NO_ACTION=False, changing_parameters=False):
    # agent - an agent object to train and solve the environment
    # episodes - the maximum number of episodes to train and attempt to solve the environment
    # step - the maximum number of steps the agent is allowed to take before timeout
    # pretrained - boolean of if a pretrained agent should be used or not. If True, will only test and not train
    # display_batch_size - the number of episodes to batch togther for displaying
    # save_cyto_data - boolean of whether to save the full output and action choices of this training run
    # NO_ACTION - If true, multipliers will be set to 1. This is used to test base environment settings for mortality/timeout/heal rates

    if save_cyto_data:                              # if saving data, creating placeholders for save data
        cyto_data = np.zeros((20, 10000, episodes))
        action_data = np.zeros((11,10000,episodes))
    total_time = time.time()                        # starting total time timer
    reward_list = []
    heal_list = []                      # list of reward total at end of each episode
    random_explore = True                           # bool for whether to use random actions or not
    TESTING = False                                 # bool for whether to test or not
    noise = True                                    # bool to add training noise or not
    problem_solved = False                          # bool for if the environment is solved or not
    start = time.time()                             # start timer for current display batch
    running_score = 0                               # initialize score for current display batch
    score = 0                                       # initialize score for current episode
    simulation_time = 0                             # initialize timer value for simulation time
    step_total = 0                                  # initialize step count for display batch
    random_explore_step = 0
    death_count = 0                                 # initialize count for deaths for display batch
    heal_count = 0                                  # initialize count for heal for display batch
    transient_infection_count = 0                   # initialize count for transient infections after healing
    timeout_count = 0                               # initialize count for timeouts for display batch
    oxydef_total = 0                                # initialize count for oxydef total for display batch
    choose_new_parameters = True
    new_param_force_change = 0

    print("\n\n\nTHIS VERSION LEARNS WHILE PLAYING\n\n\n")

    if pretrained:
        agent.actor_local = tf.keras.models.load_model('successful_actor_local.h5',custom_objects={'shallow_tanh': shallow_tanh})
        agent.critic_local = tf.keras.models.load_model('successful_critic_local.h5')
        agent.actor_target = tf.keras.models.load_model('successful_actor_target.h5',custom_objects={'shallow_tanh': shallow_tanh})
        agent.critic_target = tf.keras.models.load_model('successful_critic_target.h5')
        print("USING AGENT ACTIONS NOW")
    if changing_parameters:
        try:
            params = np.loadtxt("./edge_parameters_deadly.txt", skiprows=1, usecols=(0,1,2,3,4))
        except:
            print("WARNING: IIRABM Parameters not loaded, using single parameter")
    for current_episode in range(1, episodes+1):
        if choose_new_parameters:
            try:
                param = random.choice(params)
                OH, IS, NRI, NIR, injNum = param
                IS, NRI, NIR, injNum = int(IS), int(NRI), int(NIR), int(injNum)
                choose_new_parameters = False
                print("\nOH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}\n".format(OH, IS, NRI, NIR, injNum))

            except Exception as e:
                print(str(e))
                # OH: 0.08, IS: 3, NRI: 0, NIR: 1, injNum: 18
                # OH: 0.12, IS: 2, NRI: 0, NIR: 3, injNum: 17
                # OH: 0.12, IS: 5, NRI: 0, NIR: 1, injNum: 22
                print("Failed to pick random params, using default")
                OH=0.12
                IS=2
                NRI=0
                NIR=3
                injNum=17
                choose_new_parameters = False
                print("\nOH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}\n".format(OH, IS, NRI, NIR, injNum))

        if pretrained and not changing_parameters:
            TESTING = True
        if pretrained or NO_ACTION:
            random_explore = False
        # print(current_episode, end="\r")
        state = env.reset(OH=OH, IS=IS, NRI=NRI, NIR=NIR, injNum=injNum, seed = current_episode)
        current_step = 1                            # current agent step count
        score = 0                                   # initalize score for current episode
        for _ in range(step):
            simulation_start = time.time()
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)  # expand dimensions to correct dimensionality for agents
            # if random_explore and current_step > BATCH_SIZE:        # turn off random_explore after enough samples to learn from
            #     random_explore = False

            if random_explore:
                action = env.action_space.sample()
                action = tf.expand_dims(action, axis=0)
            else:
                action = agent.act(state, training = not TESTING)
                if np.isnan(action).any():
                    print(action)
                    print(state)
                    print(env.cytokine_history[:,env.current_step-5:env.current_step+2])
                    return
            if NO_ACTION:
                action = tf.expand_dims(np.array([0,0,0,0,0,0,0,0,0,0,0]),0)

            next_state, reward, done, info = env.step(action[0])
            score += reward                                     # increment score
            agent.step(state, action, reward, next_state, done)
            current_step += 1                                   # increment current agent step
            step_total += 1                                     # increment display batch step
            random_explore_step += 1
            state = next_state.squeeze()                        # set state equal to the next state and squeeze dimensions
            simulation_time += time.time() - simulation_start   # calculate simulation time

            if not TESTING:
                agent.train()

            if env.current_step >= ENV_STEPS:    # if the current agent step is equal to the max agent steps, done from agent timeout
                done = True
            if done:                    # save data from current episode to the final save data
                if save_cyto_data:
                    cyto_output = env.full_history[:, env.cytokine_history[0,:] >0]
                    cyto_data[:,:cyto_output.shape[1],current_episode-1] = cyto_output
                    action_data[:,:cyto_output.shape[1],current_episode-1] = env.action_history[:,:cyto_output.shape[1]]
                    cyto_data[:,cyto_output.shape[1]:,current_episode-1] = float('nan')
                    action_data[:,cyto_output.shape[1]:,current_episode-1] = float('nan')

                running_score += score
                reward_list.append(score)
                oxydef_total += info["oxydef"]
                if info["timeout"] or env.current_step >= ENV_STEPS:
                    timeout_count += 1
                    heal_list.append(0)
                elif info["dead"]:
                    death_count += 1
                    heal_list.append(0)
                elif info["healed"]:
                    heal_count += 1
                    heal_list.append(1)
                if info["transient_infection"]:
                    transient_infection_count += 1



                if np.sum(heal_list[-200:]) >= 190 and current_episode > 200: #95% healing and episode greater than 200
                    print('Task Solved')
                    if not pretrained:
                        agent.actor_local.save('successful_actor_local.h5')
                        agent.critic_local.save('successful_critic_local.h5')
                        agent.actor_target.save('successful_actor_target.h5')
                        agent.critic_target.save('successful_critic_target.h5')
                        print('Training saved')
                    problem_solved = True

                if current_episode % display_batch_size==0:
                    display_divisor = display_batch_size
                    temp_score = 0
                    temp_step = 0
                    new_param_force_change +=1
                    if 100*(heal_count/display_divisor) >= 95 or new_param_force_change > 5:
                        new_param_force_change = 0
                        choose_new_parameters = True

                    if TESTING:
                        print("TESTING -- TESTING -- TESTING -- TESTING")
                        score = running_score
                    else:
                        score = running_score
                    print('Episode: {:4.0f} | Actions:{:4.0f} | Avg Reward last {} episodes: {:5.2f} | Avg Time last {} episodes: {:.2f} Seconds'.format(current_episode, step_total/display_divisor, display_divisor, score/display_divisor, display_divisor, (time.time() - start)/display_divisor))
                    # print("Avg Times last {} - Selecting: {:3.2f}, Training: {:3.2f}, Updating: {:3.2f}, Simulating: {:3.2f}".format(display_divisor, agent.selecting_time/display_divisor, agent.training_time/display_divisor, agent.updating_time/display_divisor, simulation_time/display_divisor))
                    print("Avg Rates last {} - Mortality Rate: {:2.1f}%, Healing Rate {:2.1f}%, Timeout Rate {:2.1f}%. Final Oxydef: {:4.1f}".format(display_divisor, (death_count/display_divisor)*100, 100*(heal_count/display_divisor), 100*(timeout_count/display_divisor), oxydef_total/display_divisor))
                    if transient_infection_count > 0:
                        print("Num Reinfections: {:2.0f}".format(transient_infection_count))
                    print()
                    running_score = temp_score
                    agent.reset_timers()
                    start = time.time()
                    simulation_time = 0
                    death_count = 0
                    heal_count = 0
                    transient_infection_count = 0
                    timeout_count = 0
                    oxydef_total = 0
                    step_total = temp_step

                if random_explore and current_episode > -1 and random_explore_step > BATCH_SIZE:
                    random_explore = False
                    print ("\033[A                                                     \033[A")
                    print("USING AGENT ACTIONS NOW")
                if current_episode % EPS_BETWEEN_TEST - NUM_TEST_EPS == 0 and TESTING:
                    TESTING = False

                if current_episode%EPS_BETWEEN_TEST == 0 and current_episode > 1:
                    TESTING = True
                if current_episode % EPS_BETWEEN_EXP_UPDATE == 0 and current_episode > 0 and not changing_parameters:
                    agent.update_exploration()

                gc.collect()
                break
        # for _ in range(current_step):
        #     if not TESTING:
        #         agent.train()

        if problem_solved:
            break


    print("Done Training")
    print("Total Time Taken: {:4.2f} minutes".format((time.time()-total_time)/60))
    if save_cyto_data:
        np.save("cytokine_data.npy", cyto_data[:,:,:current_episode])
        np.save("action_data.npy", action_data[:,:,:current_episode])
    return reward_list

# TRAINING TIME

BUFFER_SIZE = 1000000           # replay buffer size
GAMMA = 0.99                    # discount factor
TAU = 0.001                     # for soft update of target parameters
LR_ACTOR = 0.0001               # learning rate of the actor
LR_CRITIC = 0.1               # learning rate of the critic
BATCH_SIZE = 128                 # minibatch size
STARTING_NOISE_MAG = .1         # initial exploration noise magnitude
EPS_BETWEEN_EXP_UPDATE = 100    # episodes inbetween exploration update

WIN_THRESHOLD = -10            # Reward threshold for win the environment

EPS_BETWEEN_TEST = 100          # Total episodes between the start of testing groups (test eps + non test eps)
NUM_TEST_EPS = 20               # Number of episodes in each test group and also size for display batch
ENV_STEPS = 9999                # Total number of environment frames allowed
action_L1= None
potential_difference_mult=200
phi_mult = 100

print("Training Parameters \n")
print("BUFFER_SIZE = {}".format(BUFFER_SIZE))
print("GAMMA = {}".format(GAMMA))
print("TAU = {}".format(TAU))
print("LR_ACTOR = {}".format(LR_ACTOR))
print("LR_CRITIC = {}".format(LR_CRITIC))
print("BATCH_SIZE = {}".format(BATCH_SIZE))
print("STARTING_NOISE_MAG = {}".format(STARTING_NOISE_MAG))
print("EPS_BETWEEN_EXP_UPDATE = {}".format(EPS_BETWEEN_EXP_UPDATE))
print("\nWIN_THRESHOLD = {}\n".format(WIN_THRESHOLD))
print("EPS_BETWEEN_TEST = {}".format(EPS_BETWEEN_TEST))
print("NUM_TEST_EPS = {}".format(NUM_TEST_EPS))
print("ENV_STEPS = {}".format(ENV_STEPS))
print("action_L1 = {}".format(action_L1))
print("potential_difference_mult = {}".format(potential_difference_mult))
print("phi_mult = {}".format(phi_mult))

# creating the environment
env = Iirabm_Environment(rendering="console", ENV_MAX_STEPS=ENV_STEPS, action_L1=action_L1, potential_difference_mult=potential_difference_mult, phi_mult = phi_mult)

env_steps_per_agent_step = env.action_repeats + env.steps_until_regression # Number of environment frames a single action is used for each agent step
AGENT_MAX_STEPS = math.floor(ENV_STEPS/env_steps_per_agent_step)    # calculating the max number of agent steps


print("Observation Space Shape: " + str(env.observation_space.shape))
print("Action Space Shape: " + str(env.action_space.shape))
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# creating the agent
ddpg_agent = Agent(state_size=state_dim, action_size=action_dim, LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, noise_magnitude=STARTING_NOISE_MAG, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TAU=TAU)

# start attempting to solve the environment
scores = solve_problem(ddpg_agent, episodes=10000, step=AGENT_MAX_STEPS, pretrained=False, display_batch_size=NUM_TEST_EPS, NO_ACTION=False, save_cyto_data=False, changing_parameters=False)
