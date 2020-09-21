import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.keras import backend as K

from noise_processes import *
from replay_buffer import ReplayBuffer

kernel_init = tf.keras.initializers.RandomNormal(stddev=0.001)

def output_activation(x):
    # function to scale tanh activation to be 1-10 if x>0 or 0-1 if x < 0
    out = K.switch(x >= 0, tf.math.tanh(x+0.1)*10, tf.math.tanh(x) + 1)
    return tf.clip_by_value(out, .001, 10)


def actor_network(obs_size, action_size):
    num_hidden1 = 400
    num_hidden2 = 300
    num_hidden3 = 200
    input = tf.keras.layers.Input(shape=obs_size)

    hidden = layers.Dense(num_hidden1, activation="relu", kernel_initializer=kernel_init)(input)
    hidden = layers.Dense(num_hidden2, activation="relu",kernel_initializer=kernel_init)(hidden)
    # hidden = layers.BatchNormalization()(hidden)
    # hidden = layers.Dense(num_hidden3, activation="relu")(hidden)

    output = layers.Dense(action_size, activation='tanh',kernel_initializer=kernel_init)(hidden)

    model = tf.keras.Model(input, output)
    return model


def critic_network(obs_size, action_size):
    state_hidden = 400
    action_hidden = 400
    output_hidden = 300

    # State as input
    state_input = layers.Input(shape=(obs_size))
    state_out = layers.Dense(state_hidden, activation="relu",kernel_initializer=kernel_init)(state_input)

    # Action as input
    action_input = layers.Input(shape=(action_size))
    # action_out = action_input
    # action_out = layers.Dense(action_hidden, activation="relu",kernel_initializer=kernel_init)(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_input])

    out = layers.Dense(output_hidden, activation="relu",kernel_initializer=kernel_init)(concat)
    output = layers.Dense(1, kernel_initializer=kernel_init)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], output)

    return model

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, LR_ACTOR=.0001, LR_CRITIC=.001, noise_magnitude=.1, BUFFER_SIZE=1000000, BATCH_SIZE=32, GAMMA=.99, TAU=.001):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = LR_ACTOR
        self.lr_critic = LR_CRITIC
        self.noise_magnitude = noise_magnitude
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.episode = 0
        self.training_time = 0
        self.updating_time = 0
        self.selecting_time = 0
        # Actor Network (w/ Target Network)

        self.actor_local = actor_network(state_size, action_size)
        self.actor_target = actor_network(state_size, action_size)
        # setting weights to be the same
        self.actor_target.set_weights(self.actor_local.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = critic_network(state_size, action_size)
        self.critic_target = critic_network(state_size, action_size)
        self.critic_target.set_weights(self.critic_local.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_critic)

        # Noise process
        self.noise = GaussianNoiseProcess(self.noise_magnitude, self.action_size)
        # self.noise = OUNoise(self.action_size)
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)

    def reset_timers(self):
        self.training_time = 0
        self.updating_time = 0
        self.selecting_time = 0

    def act(self, state, training):
        """Returns actions for given state as per current policy."""

        action = self.actor_local(state)
        if training:
            action += self.noise.sample()
        else:
            action = self.actor_target(state)
        # print(action)
        # print(state)
        return action

    def reset(self):
        self.noise.reset()
        self.episode += 1

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory"""
        self.memory.add(state, action, reward, next_state, done)

    def train(self):
        """Learn using random samples, if enough samples are available in memory"""
        if len(self.memory) > self.batch_size:
            selecting_time_start = time.time()
            experiences = self.memory.sample()
            self.selecting_time += time.time() - selecting_time_start
            self.learn(experiences)

    def learn(self, experiences):
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
        self.tf_learn(states, actions, rewards, next_states, dones)
        self.training_time += time.time() - train_time_start

        # ----------------------- update target networks ----------------------- #
        update_time_start = time.time()
        self.update_target_variables(self.critic_target.weights, self.critic_local.weights, tau=self.tau)
        self.update_target_variables(self.actor_target.weights, self.actor_local.weights, tau=self.tau)
        self.updating_time += time.time() - update_time_start

    @tf.function
    def tf_learn(self, states, actions, rewards, next_states, dones):

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        with tf.GradientTape() as tape:
            actions_pred = self.actor_local(states)
            actor_loss = -tf.reduce_mean(self.critic_local([states, actions_pred]))
            # Minimize the loss
            actor_grad = tape.gradient(actor_loss, self.actor_local.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_local.trainable_variables))

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with tf.GradientTape() as tape:
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target([next_states, actions_next])
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local([states, actions])
            critic_loss = tf.reduce_mean((Q_expected-Q_targets)**2)
        # print("{:1.4f}".format(critic_loss), end="\r")
        # Minimize the loss
            critic_grad = tape.gradient(critic_loss, self.critic_local.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_local.trainable_variables))


    # Function taken from
    # https://github.com/keiohta/tf2rl/blob/84a3f4f19dd42e5e0fb5e5714529c51fe2b7da11/tf2rl/misc/target_update_ops.py
    def update_target_variables(self, target_variables,
                                source_variables,
                                tau=1.0,
                                use_locking=False,
                                name="update_target_variables"):
        """
        Returns an op to update a list of target variables from source variables.
        The update rule is:
        `target_variable = (1 - tau) * target_variable + tau * source_variable`.
        :param target_variables: a list of the variables to be updated.
        :param source_variables: a list of the variables used for the update.
        :param tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
            with small tau representing an incremental update, and tau == 1
            representing a full update (that is, a straight copy).
        :param use_locking: use `tf.Variable.assign`'s locking option when assigning
            source variable values to target variables.
        :param name: sets the `name_scope` for this op.
        :raise TypeError: when tau is not a Python float
        :raise ValueError: when tau is out of range, or the source and target variables
            have different numbers or shapes.
        :return: An op that executes all the variable updates.
        """
        if not isinstance(tau, float):
            raise TypeError("Tau has wrong type (should be float) {}".format(tau))
        if not 0.0 < tau <= 1.0:
            raise ValueError("Invalid parameter tau {}".format(tau))
        if len(target_variables) != len(source_variables):
            raise ValueError("Number of target variables {} is not the same as "
                             "number of source variables {}".format(
                                 len(target_variables), len(source_variables)))

        same_shape = all(trg.get_shape() == src.get_shape()
                         for trg, src in zip(target_variables, source_variables))
        if not same_shape:
            raise ValueError("Target variables don't have the same shape as source "
                             "variables.")

        def update_op(target_variable, source_variable, tau):
            if tau == 1.0:
                return target_variable.assign(source_variable, use_locking)
            else:
                return target_variable.assign(
                    tau * source_variable + (1.0 - tau) * target_variable, use_locking)

        # with tf.name_scope(name, values=target_variables + source_variables):
        update_ops = [update_op(target_var, source_var, tau)
                      for target_var, source_var
                      in zip(target_variables, source_variables)]
        return tf.group(name="update_all_variables", *update_ops)


    def set_noise_process(self, np):
        self.noise_process = np

    def update_exploration(self):
        self.noise_magnitude /= 2
        self.set_noise_process(GaussianNoiseProcess(self.noise_magnitude, self.action_size))
        print("Reducing noise to {}".format(self.noise_magnitude))

    def suspend_exploration(self):
        self.set_noise_process(GaussianNoiseProcess(0))

    def restore_exploration(self):
        self.set_noise_process(GaussianNoiseProcess(self.noise_magnitude))
