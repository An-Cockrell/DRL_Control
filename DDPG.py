# https://github.com/shivaverma/OpenAIGym/tree/master/bipedal-walker/ddpg-torch

import gym

import numpy as np
import copy
import random
from collections import namedtuple, deque
import time
import gc

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

from iirabm_env import Iirabm_Environment
from ddpg_agent import Agent

def ddpg(agent, episodes, step, pretrained, display_batch_size):
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

    print("\n\n\nTHIS VERSION LEARNS WHILE PLAYING\n\n\n")

    if pretrained:
        agent.actor_local = tf.keras.models.load_model('successful_actor_local.h5')
        agent.critic_local = tf.keras.models.load_model('successful_critic_local.h5')
        agent.actor_target = tf.keras.models.load_model('successful_actor_target.h5')
        agent.critic_target = tf.keras.models.load_model('successful_critic_target.h5')
        TESTING = True

    for current_episode in range(1, episodes):
        env.seed(0)
        state = env.reset()
        current_step = 0
        output_range = None
        score = 0

        for _ in range(step):
            simulation_start = time.time()
            # env.render()
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            if random_explore:
                action = env.action_space.sample()
                action = tf.expand_dims(action, axis=0)
            else:
                action = agent.act(state, training = not TESTING)

            next_state, reward, done, info = env.step(action[0])
            current_step += 1
            # print(reward)
            # if current_step > 100: # after the burn in period, then start learning. Also so we dont add step < 100 to memory
            agent.step(state, action, reward, next_state, done)
            score += reward
            Q_reward = agent.critic_local([state, action])[0][0]
            Q_score += Q_reward
            state = next_state.squeeze()
            simulation_time += time.time() - simulation_start

            if not TESTING:
                agent.train()

            if done:
                running_score += score
                reward_list.append(score)

                if current_episode % display_batch_size==0 or TESTING:
                    display_divisor = display_batch_size
                    temp_score = 0
                    if TESTING:
                        print("TESTING -- TESTING -- TESTING -- TESTING")
                        temp_score = running_score
                        display_divisor = 1
                    else:
                        score = running_score
                    print('Episode: {:4.0f} | Avg Reward last {} episodes: {:5.2f} | Avg Time last {} episodes: {:.2f} Seconds'.format(current_episode, display_divisor, score/display_divisor, display_batch_size, (time.time() - start)/display_divisor))
                    print("Avg last {} - Selecting: {:3.2f}, Training: {:3.2f}, Updating: {:3.2f}, Simulating: {:3.2f}".format(display_divisor, agent.selecting_time/display_divisor, agent.training_time/display_divisor, agent.updating_time/display_divisor, simulation_time/display_divisor))
                    # print("LOWS:  {:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(*output_range[0,:]))
                    # print("HIGHS: {:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(*output_range[1,:]))

                    # print("Real reward: {:4.0f}, Q_score: {:4.0f}, Difference: {:4.0f}".format(score, Q_score, score-Q_score))
                    running_score = temp_score
                    Q_score = 0
                    agent.reset_timers()
                    start = time.time()
                    simulation_time = 0

                if random_explore and current_episode > -1:
                    random_explore = False
                    print("USING AGENT ACTIONS NOW")
                if current_episode % (NUM_TEST_EPS+1) == 0 and TESTING:
                    TESTING = False
                    if np.mean(reward_list[-50:]) >= 200:
                        print('Task Solved')
                        agent.actor_local.save('successful_actor_local.h5')
                        agent.critic_local.save('successful_critic_local.h5')
                        agent.actor_target.save('successful_actor_target.h5')
                        agent.critic_target.save('successful_critic_target.h5')
                        print('Training saved')
                        problem_solved = True
                if current_episode%100 == 0:
                    TESTING = True
                if current_episode % EPS_BETWEEN_EXP_UPDATE == 0 and current_episode > 0:
                    agent.update_exploration()

                gc.collect()
                break
        # for _ in range(current_step):
        #     if not TESTING:
        #         agent.train()

        if problem_solved:
            break


    print("Done Training")
    print("Total Time Taken: {:f4.2} minutes".format((time.time()-total_time)/60))
    return reward_list

# TRAINING TIME

BUFFER_SIZE = 1000000      # replay buffer size
BATCH_SIZE = 32        # minibatch size
GAMMA = 0.99               # discount factor
TAU = 0.001                # for soft update of target parameters
LR_ACTOR = 0.0001          # learning rate of the actor
LR_CRITIC = 0.001          # learning rate of the critic
WEIGHT_DECAY = 0.001       # L2 weight decay
STARTING_NOISE_MAG = .5    #initial exploration noise magnitude
EPS_BETWEEN_EXP_UPDATE = 200 #episodes inbetween exploration update

NUM_TEST_EPS = 2

env = Iirabm_Environment(rendering="human", action_repeats=20)
# env = gym.make("LunarLanderContinuous-v2")  # Create the environment
print(env.observation_space.shape)
print(env.action_space.shape)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


ddpg_agent = Agent(state_size=state_dim, action_size=action_dim, LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, noise_magnitude=STARTING_NOISE_MAG, BUFFER_SIZE=BUFFER_SIZE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, TAU=TAU)

scores = ddpg(ddpg_agent, episodes=10000, step=5000, pretrained=False, display_batch_size=20)

fig = plt.figure()
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
