# Script to create DRL controller for IIRABM
# Created by Dale Larie 7/29/2020
# Questions to daleblarie@gmail.com

# following basic outline from:
# https://keras.io/examples/rl/actor_critic_cartpole/
# https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
# https://gist.github.com/nevkontakte/2db02b57345ca521d541f8cdbf4081c5

import ctypes
from numpy.ctypeslib import ndpointer
from iirabm_env import Iirabm_Environment
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

import tensorflow.compat.v1 as tfc
import tensorflow_probability as tfp
tfd = tfp.distributions


env = Iirabm_Environment()
env.render = True

################################################################
#sample from state space for state normalization
import sklearn
import sklearn.preprocessing
state_space_samples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)
#function to normalize states
def scale_state(state):                 #requires input shape=(2,)
    scaled = scaler.transform([state])
    return scaled                       #returns shape =(1,2)
###################################################################

def value_function():
    num_hidden1 = 121
    num_hidden2 = 121
    num_outputs = np.squeeze(env.action_space.shape)

    with tfc.variable_scope("value_network"):
        init_xavier = tf.initializers.glorot_uniform()
        input = tf.keras.layers.Input(shape=env.observation_space.shape)
        hidden1 = tf.keras.layers.Dense(num_hidden1, tf.nn.elu, kernel_initializer=init_xavier)(input)
        hidden2 = tf.keras.layers.Dense(num_hidden2, tf.nn.elu, kernel_initializer=init_xavier)(hidden1)
        V = tf.keras.layers.Dense(num_outputs, None, kernel_initializer=init_xavier)(hidden2)

    model = tf.keras.Model(name='value_model', inputs=input, outputs=[V])
    model.compile(optimizer='adam', loss='mse')
    return model

def policy_network():
    num_hidden1 = 121
    num_hidden2 = 1331
    num_outputs = np.squeeze(env.action_space.shape)

    init_xavier = tf.initializers.glorot_uniform()

    input = tf.keras.layers.Input(shape=env.observation_space.shape)

    meta_learn0 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn1 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn2 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn3 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn4 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn5 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn6 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn7 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn8 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn9 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learn10 = tf.keras.layers.Dense(num_hidden1, kernel_initializer=init_xavier)(input)
    meta_learner_layer = tf.keras.layers.Concatenate(axis=1)([meta_learn0, meta_learn1, meta_learn3, meta_learn4, meta_learn5,
                                meta_learn6, meta_learn7, meta_learn8, meta_learn9, meta_learn10])

    common = tf.keras.layers.Dense(num_hidden2, kernel_initializer=init_xavier)(meta_learner_layer)
    norm = tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(num_outputs),tf.nn.relu,
                              kernel_initializer=init_xavier)(common)
    output = tfp.layers.IndependentNormal(num_outputs)(norm)

    def loss(y_true, y_pred):
        action_true = y_true[:, :num_outputs]
        advantage = y_true[:, num_outputs:]
        return -tfc.log(y_pred.prob(action_true) + 1e-5) * advantage

    model = tf.keras.Model(name='policy_model', inputs=input, outputs=output)
    model.compile(optimizer='adam', loss=loss)
    return model

lr_actor = 0.00002  #set learning rates
lr_critic = 0.001

# ----------------------------------------------------------------------------------------------------------------------
# Instantiate models and do the training.

actor_model = policy_network()
value_model = value_function()

gamma = 0.99  # Reward discount factor
num_episodes = 300

episode_history = []
for episode in range(num_episodes):
    # receive initial state from E
    state = env.reset()
    reward_total = 0
    steps = 0
    done = False

    while not done:

        # Sample action according to current policy
        # action.shape = (1,1)
        action_dist = actor_model(scale_state(state))
        action = tf.convert_to_tensor(action_dist)
        action = tf.clip_by_value(action, env.action_space.low, env.action_space.high)

        # Execute action and observe reward & next state from E
        # next_state shape=(2,)
        # env.step() requires input shape = (1,)
        # print(action)
        if tf.math.is_nan(action[0,0]):
            print(state)
            print("\n\n\n")
            print(action_dist)
            break
        next_state, reward, done, _ = env.step(np.squeeze(action, axis=0))
        steps += 1
        reward_total += reward

        # V_of_next_state.shape=(1,1)
        V_of_next_state = value_model(scale_state(next_state))

        # Set TD Target
        # target = r + gamma * V(next_state)
        target = reward + gamma * np.squeeze(V_of_next_state)

        V_state = value_model(scale_state(state))
        # td_error = target - V(s)
        # needed to feed delta_placeholder in actor training
        td_error = target - V_state


        # A trick to pass TD error *and* actual action to the loss function: join them into a tensor and split apart
        # Inside the loss function.
        annotated_action = tf.concat([action, td_error], axis=1)

        # Update actor by minimizing loss (Actor training)
        actor_model.train_on_batch([scale_state(state)], [annotated_action])
        # Update critic by minimizing loss (Critic training)
        value_model.train_on_batch([scale_state(state)], [target])

        state = next_state
    # end while
    if tf.math.is_nan(action[0,0]):
        print(action)
        break
    episode_history.append(reward_total)
    print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.2f}                                                            ".format(
        episode, steps, reward_total))

    if np.mean(episode_history[-100:]) > 10000 and len(episode_history) >= 101:
        print("****************Solved***************")
        print("Mean cumulative reward over 100 episodes:{:0.2f}".format(
            np.mean(episode_history[-100:])))
