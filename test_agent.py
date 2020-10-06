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

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from iirabm_env import Iirabm_Environment
from ddpg_agent import Agent
tf.config.set_visible_devices([], 'GPU')

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
    print("\n\n\nTHIS VERSION LEARNS WHILE PLAYING\n\n\n")

    if pretrained:
        agent.actor_local = tf.keras.models.load_model('successful_actor_local.h5')
        agent.critic_local = tf.keras.models.load_model('successful_critic_local.h5')
        agent.actor_target = tf.keras.models.load_model('successful_actor_target.h5')
        agent.critic_target = tf.keras.models.load_model('successful_critic_target.h5')
        TESTING = True

    for current_episode in range(1, episodes+1):
        if pretrained:
            TESTING = True
        print(current_episode, end="\r")
        # env.seed(current_episode)
        state = env.reset(OH=OH, IS=IS, NRI=NRI, NIR=NIR, injNum=injNum, seed = current_episode-1)
        current_step = 1
        output_range = None
        score = 0
        for _ in range(step):
            simulation_start = time.time()

            state = tf.expand_dims(tf.convert_to_tensor(state), 0)

            action = agent.act(state, training = not TESTING)

            next_state, reward, done, info = env.step(action[0])
            current_step += 1
            step_total += 1

            agent.step(state, action, reward, next_state, done)
            score += reward
            Q_reward = agent.critic_local([state, action])[0][0]
            Q_score += Q_reward
            state = next_state.squeeze()
            simulation_time += time.time() - simulation_start

            if current_step == step:
                done = True
            if done:
                if save_cyto_data:
                    cyto_output = env.full_history[:, env.cytokine_history[0,:] >0]
                    cyto_data[:,:cyto_output.shape[1],current_episode-1] = cyto_output
                    action_data[:,:cyto_output.shape[1],current_episode-1] = env.action_history[:,:cyto_output.shape[1]]
                    cyto_data[:,cyto_output.shape[1]:,current_episode-1] = float('nan')
                    action_data[:,cyto_output.shape[1]:,current_episode-1] = float('nan')

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
                        print("TESTING -- TESTING -- TESTING -- TESTING")
                        score = running_score
                        # temp_score = running_score
                        # temp_step = step_total
                        # step_total = current_step
                        # display_divisor = NUM_TEST_EPS
                        # display_divisor = 1
                    else:
                        score = running_score
                    print('Episode: {:4.0f} | Steps: {:4.0f} | Avg Reward last {} episodes: {:5.2f} | Avg Time last {} episodes: {:.2f} Seconds'.format(current_episode, step_total/display_divisor, display_divisor, score/display_divisor, display_divisor, (time.time() - start)/display_divisor))
                    # print("Avg Times last {} - Selecting: {:3.2f}, Training: {:3.2f}, Updating: {:3.2f}, Simulating: {:3.2f}".format(display_divisor, agent.selecting_time/display_divisor, agent.training_time/display_divisor, agent.updating_time/display_divisor, simulation_time/display_divisor))
                    # print("LOWS:  {:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(*output_range[0,:]))
                    # print("HIGHS: {:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(*output_range[1,:]))
                    print("Avg Rates last {} - Mortality Rate: {:2.1f}%, Healing Rate {:2.1f}%, Timeout Rate {:2.1f}%. Final Oxydef: {:4.1f}".format(display_divisor, (death_count/display_divisor)*100, 100*(heal_count/display_divisor), 100*(timeout_count/display_divisor), oxydef_total/display_divisor))
                    if transient_infection_count > 0:
                        print("Num Reinfections: {:2.0f}".format(transient_infection_count))
                    # print("Real reward: {:4.0f}, Q_score: {:4.0f}, Difference: {:4.0f}".format(score, Q_score, score-Q_score))
                    print()
                    running_score = temp_score
                    Q_score = 0
                    agent.reset_timers()
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



    print("Done Training")
    print("Total Time Taken: {:4.2f} minutes".format((time.time()-total_time)/60))

    return reward_list, cyto_data, action_data

BUFFER_SIZE = 100      # replay buffer size
GAMMA = 0.99               # discount factor
TAU = 0.001                # for soft update of target parameters
LR_ACTOR = 0.0001          # learning rate of the actor
LR_CRITIC = 0.001          # learning rate of the critic
WEIGHT_DECAY = 0.001       # L2 weight decay
BATCH_SIZE = 64        # minibatch size
STARTING_NOISE_MAG = .2  #initial exploration noise magnitude
EPS_BETWEEN_EXP_UPDATE = 500 #episodes inbetween exploration update

EPS_BETWEEN_TEST = 100
NUM_TEST_EPS = 20
ENV_STEPS = 8000
AGENT_ACTION_REPEATS = 4
AGENT_MAX_STEPS = math.floor(ENV_STEPS/AGENT_ACTION_REPEATS)
env = Iirabm_Environment(rendering="console", action_repeats=AGENT_ACTION_REPEATS, ENV_MAX_STEPS=ENV_STEPS, action_L1=0.1, potential_difference_mult=200, phi_mult = 100)
# env = gym.make("LunarLanderContinuous-v2")  # Create the environment
print(env.observation_space.shape)
print(env.action_space.shape)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


ddpg_agent = Agent(state_size=state_dim, action_size=action_dim, LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, noise_magnitude=STARTING_NOISE_MAG, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TAU=TAU)



params = np.loadtxt("./edge_parameters.txt", skiprows=1)
save_data_cyto = np.zeros((20, 10000, EPS_BETWEEN_TEST, params.shape[0]))
save_data_action = np.zeros((11,10000, EPS_BETWEEN_TEST, params.shape[0]))
param_count = 0
for param in params:
    OH, IS, NRI, NIR, injNum = param
    IS, NRI, NIR, injNum = int(IS), int(NRI), int(NIR), int(injNum)
    print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
    scores, save_data_cyto[:,:,:, param_count], save_data_action[:,:,:, param_count] = ddpg(ddpg_agent, episodes=EPS_BETWEEN_TEST, step=AGENT_MAX_STEPS, pretrained=True, display_batch_size=NUM_TEST_EPS, NO_ACTION=False)
    param_count += 1
    np.save("final_test_cytokine_data.npy", save_data_cyto)
    np.save("final_test_action_data.npy", save_data_action)
