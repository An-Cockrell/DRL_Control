# https://github.com/shivaverma/OpenAIGym/tree/master/bipedal-walker/ddpg-torch

import numpy as np
import copy
import random
from collections import namedtuple, deque
import math
import time
import resource
import gc
import psutil

from iirabm_env import Iirabm_Environment

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt


def output_activation(x):
    # function to scale tanh activation to be 1-10 if x>0 or 0-1 if x < 0
    # return tf.cond(x >= 0, lambda: tf.math.tanh(x+0.1)*10, lambda: tf.math.tanh(x) + 1)
    out = K.switch(x >= 0, tf.math.tanh(x+0.1)*10, tf.math.tanh(x) + 1)
    return tf.clip_by_value(out, .001, 10)


def actor_network(obs_size, action_size):
    num_hidden1 = 484
    num_hidden2 = 242
    input = tf.keras.layers.Input(shape=obs_size)

    hidden = layers.Dense(num_hidden1, activation="linear")(input)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dense(num_hidden2, activation="linear")(hidden)
    hidden = layers.BatchNormalization()(hidden)

    output = layers.Dense(action_size, activation='tanh',kernel_initializer='random_normal')(hidden)

    model = tf.keras.Model(input, output)
    return model


def critic_network(obs_size, action_size):
    state_hidden = 300
    action_hidden = 300
    output_hidden = 600

    # State as input
    state_input = layers.Input(shape=(obs_size))
    state_out = layers.Dense(state_hidden, activation="relu")(state_input)
    state_out = layers.BatchNormalization()(state_out)
    state_out = layers.Dense(state_hidden*2, activation="relu")(state_out)
    state_out = layers.BatchNormalization()(state_out)

    # Action as input
    action_input = layers.Input(shape=(action_size))
    action_out = layers.Dense(action_hidden, activation="relu")(action_input)
    action_out = layers.BatchNormalization()(action_out)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(output_hidden, activation="linear")(concat)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(output_hidden, activation="relu")(out)
    out = layers.BatchNormalization()(out)
    output = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], output)

    return model

BUFFER_SIZE = 1000000      # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99               # discount factor
TAU = 0.001                # for soft update of target parameters
LR_ACTOR = 0.0001          # learning rate of the actor
LR_CRITIC = 0.001          # learning rate of the critic
WEIGHT_DECAY = 0.001       # L2 weight decay


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        self.episode = 0
        self.training_time = 0
        self.updating_time = 0
        self.selecting_time = 0
        self.noise_magnitude = .5
        # Actor Network (w/ Target Network)

        self.actor_local = actor_network(state_size, action_size)
        self.actor_target = actor_network(state_size, action_size)
        # setting weights to be the same
        self.actor_target.set_weights(self.actor_local.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = critic_network(state_size, action_size)
        self.critic_target = critic_network(state_size, action_size)
        self.critic_target.set_weights(self.critic_local.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=LR_CRITIC)

        # Noise process
        self.noise = GaussianNoiseProcess(self.noise_magnitude, self.action_size)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

    def reset_timers(self):
        self.training_time = 0
        self.updating_time = 0
        self.selecting_time = 0

    def step(self, state, action, reward, next_state, done, testing=False):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory

        if len(self.memory) > BATCH_SIZE and not testing:
            selecting_time_start = time.time()
            experiences = self.memory.sample()
            self.selecting_time += time.time() - selecting_time_start
            self.learn(experiences, tf.constant(GAMMA))

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        action = self.actor_local(state)
        if add_noise:
            action += self.noise.sample()

        # print(action)
        # print(state)
        return action

    def reset(self):
        self.noise.reset()
        self.episode += 1

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        train_time_start = time.time()
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with tf.GradientTape() as tape:
            actions_next = self.predict_actor_local(next_states)
            Q_targets_next = self.predict_critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = compute_Q_targets(rewards, gamma, Q_targets_next, dones)
            # Compute critic loss
            Q_expected = self.predict_critic_local(states, actions)
            critic_loss = compute_critic_loss(Q_expected, Q_targets)

        # Minimize the loss
        critic_grad = tape.gradient(critic_loss, self.critic_local.trainable_variables)
        self.apply_critic_grads(critic_grad, self.critic_local.trainable_variables)
        # self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_local.trainable_variables))

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        with tf.GradientTape() as tape:
            actions_pred = self.predict_actor_local(states)
            actor_loss = compute_actor_loss(self.predict_critic_local(states, actions_pred))
        # Minimize the loss
        actor_grad = tape.gradient(actor_loss,self.actor_local.trainable_variables)
        self.apply_actor_grads(actor_grad, self.actor_local.trainable_variables)
        # self.actor_optimizer.apply_gradients(
        #     zip(actor_grad, self.actor_local.trainable_variables))
        self.training_time += time.time() - train_time_start
        # self.tf_learn(states, actions, rewards, next_states, dones, gamma)
        # ----------------------- update target networks ----------------------- #
        update_time_start = time.time()
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.updating_time += time.time() - update_time_start
    @tf.function
    def predict_actor_local(self, state):
        return self.actor_local(state)
    @tf.function
    def predict_actor_target(self,state):
        return self.actor_target(state)
    @tf.function
    def predict_critic_local(self, states, actions_pred):
        return self.critic_local([states, actions_pred])
    @tf.function
    def predict_critic_target(self, states, actions_next):
        return self.critic_target([states, actions_next])
    @tf.function
    def apply_critic_grads(self, crit_grad, trainable_vars):
        self.critic_optimizer.apply_gradients(zip(crit_grad, trainable_vars))
    @tf.function
    def apply_actor_grads(self, act_grad, trainable_vars):
        self.actor_optimizer.apply_gradients(zip(act_grad, trainable_vars))

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        new_weights = []
        target_variables = target_model.weights
        for i, variable in enumerate(local_model.weights):
            new_weights.append(variable * TAU + target_variables[i] * (1 - TAU))

        target_model.set_weights(new_weights)

    def set_noise_process(self, np):
        self.noise_process = np

    def update_exploration(self):
        self.noise_magnitude /= 5
        self.set_noise_process(GaussianNoiseProcess(self.noise_magnitude, self.action_size))
        print("Reducing noise to {}".format(self.noise_magnitude))

    def suspend_exploration(self):
        self.set_noise_process(GaussianNoiseProcess(0))

    def restore_exploration(self):
        self.set_noise_process(GaussianNoiseProcess(self.noise_magnitude))

    # @tf.function
    # def tf_learn(self, states, actions, rewards, next_states, dones, gamma):
    #
    #     # ---------------------------- update critic ---------------------------- #
    #     # Get predicted next-state actions and Q values from target models
    #     with tf.GradientTape() as tape:
    #         actions_next = self.actor_target(next_states)
    #         Q_targets_next = self.critic_target([next_states, actions_next])
    #         # Compute Q targets for current states (y_i)
    #         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    #         # Compute critic loss
    #         Q_expected = self.critic_local([states, actions])
    #         critic_loss = tf.math.reduce_mean(tf.math.square(Q_expected-Q_targets))
    #
    #     # Minimize the loss
    #     critic_grad = tape.gradient(critic_loss, self.critic_local.trainable_variables)
    #     self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_local.trainable_variables))
    #
    #     # ---------------------------- update actor ---------------------------- #
    #     # Compute actor loss
    #     with tf.GradientTape() as tape:
    #         actions_pred = self.actor_local(states)
    #         actor_loss = -1 * tf.math.reduce_mean(self.critic_local([states, actions_pred]))
    #     # Minimize the loss
    #     actor_grad = tape.gradient(actor_loss, self.actor_local.trainable_variables)
    #     self.actor_optimizer.apply_gradients(
    #         zip(actor_grad, self.actor_local.trainable_variables))
@ tf.function
def compute_Q_targets(rewards, gamma, Q_targets_next, dones):
    return (rewards + (gamma * Q_targets_next * (1 - dones)))
@tf.function
def compute_critic_loss(Q_expected, Q_targets):
    return tf.math.reduce_mean(tf.math.square(Q_expected - Q_targets))
@tf.function
def compute_actor_loss(critic_val):
     return -1 * tf.math.reduce_mean(critic_val)
"""
Noises processes

    These define how noise is added to the training policy to encourage exploration
"""
class GaussianNoiseProcess:
    """
    Simply adds noise of N(0, std^2)
    """
    def __init__(self, std, shape):
        self.std = std
        self.shape = shape
    def sample(self):
        return np.random.normal(np.zeros(self.shape), self.std)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.4, sigma=.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.uniform(-1,1) for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = tf.convert_to_tensor(np.vstack([e.state for e in experiences if e is not None]).astype(np.float32))
        actions = tf.convert_to_tensor(np.vstack([e.action for e in experiences if e is not None]).astype(np.float32))
        rewards = tf.convert_to_tensor(np.vstack([e.reward for e in experiences if e is not None]).astype(np.float32))
        next_states = tf.convert_to_tensor(np.vstack([e.next_state for e in experiences if e is not None]).astype(np.float32))
        dones = tf.convert_to_tensor(np.vstack([e.done for e in experiences if e is not None]).astype(np.float32))

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def ddpg(agent, episodes, step, pretrained, display_batch_size):

    if pretrained:
        agent.actor_local = tf.keras.models.load_model('successful_actor_local.h5')
        agent.critic_local = tf.keras.models.load_model('successful_critic_local.h5')
        agent.actor_target = tf.keras.models.load_model('successful_actor_target.h5')
        agent.critic_targe = tf.keras.models.load_model('successful_critic_target.h5')

    reward_list = []
    random_explore = False
    noise = True
    cytoMax = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0])
    start = time.time()
    score = 0

    for current_episode in range(1, episodes):
        env.set_seed(current_episode)
        state = env.reset()
        output_range = None
        while True:
            t = env.current_step
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            if random_explore:
                action = env.action_space.sample()
                action = tf.expand_dims(action, axis=0)
            else:
                action = agent.act(state, noise = -1* TESTING)
            if output_range is not None:
                action_np = env.cytokine_mults
                for j in range(action_np.shape[0]):
                    if output_range[0,j] > action_np[j]:
                        output_range[0,j] = action_np[j]
                    elif output_range[1,j] < action_np[j]:
                        output_range[1,j] = action_np[j]
            else:
                output_range = np.stack([env.cytokine_mults, env.cytokine_mults])
                output_range = np.squeeze(output_range)
            next_state, reward, done, info = env.step(action[0])
            # print(reward)
            if env.current_step > 100: # after the burn in period, then start learning. Also so we dont add step < 100 to memory
                agent.step(state, action, reward, next_state, done, testing=TESTING)
                score += reward
            state = next_state.squeeze()

            if done:
                if current_episode % display_batch_size==0:
                    if TESTING:
                        print("TESTING -- TESTING")
                    print('Episode: {:4.0f} | Avg Reward last {} episodes: {:5.2f} | Avg Time last {} episodes: {:.2f} Seconds'.format(current_episode, display_batch_size, score/display_batch_size, display_batch_size, (time.time() - start)/display_batch_size))
                    print("Avgs of last {} - Selecting Time: {:3.2f}, Training Time: {:3.2f}, Updating Time: {:3.2f}".format(display_batch_size, agent.selecting_time/display_batch_size, agent.training_time/display_batch_size, agent.updating_time/display_batch_size))
                    # print("LOWS:  {:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(*output_range[0,:]))
                    # print("HIGHS: {:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(*output_range[1,:]))
                    print("% Total Memory available: {:3.5f}".format(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total))

                    print()
                    score = 0
                    agent.reset_timers()
                    start = time.time()


                if current_episode % 10 == 0 and TESTING:
                    TESTING = False
                    if np.mean(reward_list[-10:]) >= 2000:
                        print('Task Solved')
                        agent.actor_local.save('successful_actor_local.h5')
                        agent.critic_local.save('successful_critic_local.h5')
                        agent.actor_target.save('successful_actor_target.h5')
                        agent.critic_target.save('successful_critic_target.h5')
                        print('Training saved')
                        break
                if current_episode%100 == 0:
                    TESTING = True
                if current_episode % 200 == 0 and current_episode > 0:
                    agent.update_exploration()
                # if current_episode % 5 == 0 and current_episode > 0:
                #     print("saving model checkpoints and clearing memory")
                #     agent.actor_local.save('checkpoint_actor_local.h5')
                #     agent.actor_target.save('checkpoint_actor_target.h5')
                #     agent.critic_local.save('checkpoint_critic_local.h5')
                #     agent.critic_target.save('checkpoint_critic_target.h5')
                #     K.clear_session()
                #     print("reloading models")
                #     agent.actor_local = tf.keras.models.load_model('checkpoint_actor_local.h5')
                #     agent.actor_target = tf.keras.models.load_model('checkpoint_actor_target.h5')
                #     agent.critic_local = tf.keras.models.load_model('checkpoint_critic_local.h5')
                #     agent.critic_target = tf.keras.models.load_model('checkpoint_critic_target.h5')
                gc.collect()
                break

        reward_list.append(score)


    print("Done Training")
    return reward_list

# TRAINING TIME



env = Iirabm_Environment(rendering=None)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


ddpg_agent = Agent(state_size=state_dim, action_size=action_dim)

scores = ddpg(ddpg_agent, episodes=10000, step=2000, pretrained=False, display_batch_size=10)

fig = plt.figure()
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
