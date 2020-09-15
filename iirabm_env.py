# IIRABM DRL Environment
# Created by Dale Larie 7/29/2020
# Questions to daleblarie@gmail.com

# following this:
# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e


import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import wrapper_setup

import gym

import matplotlib.pyplot as plt
from scipy import stats

# PLOTTING VARS
globalLine = None
globalAx = None
globalFig = None
globalBG = None



MAX_OXYDEF = 8160
MAX_STEPS = 9999
NUM_CYTOKINES_CONTROLLED = 11
NUM_OBSERVTAIONS = 4
# OBS_VEC_SHAPE = NUM_CYTOKINES*((NUM_OBSERVTAIONS*2)-1)
all_signals_max = np.array([8164,  250,  118, 1675,  880,  108, 4027,  730, 1232, 2204,   87,   83])


SIM = wrapper_setup.setUpWrapper()

def createIIRABM(OH, IS, NRI, NIR, injNum, seed, numCytokines):
    oxyHeal = ctypes.c_float(OH)
    IS = ctypes.c_int(IS)
    NRI = ctypes.c_int(NRI)
    NIR = ctypes.c_int(NIR)
    injNum = ctypes.c_int(injNum)
    seed = ctypes.c_int(seed)
    numCytokines = ctypes.c_int(numCytokines)
    internalParameterization = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    instance = SIM.CreateInstance(oxyHeal,IS,NRI,NIR,injNum,seed,numCytokines, internalParameterization.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return instance

class Iirabm_Environment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, rendering=None, action_repeats=4, ENV_MAX_STEPS=MAX_STEPS, action_L1=None, potential_difference_mult=None, phi_mult=1):
        super(Iirabm_Environment, self).__init__()

        self.reward_range = (-250,250)
        self.cytokine_history = np.zeros((12,10000))
        self.cytokine_mults = np.zeros((11,1))
        self.oxydef_history = np.zeros((1,10000))
        self.action_history = np.zeros((11,10000))
        self.current_step = 0
        self.RL_step = 0
        self.action_L1 = action_L1
        self.phi_mult = phi_mult
        self.potential_difference_mult = potential_difference_mult
        self.rendering = rendering
        self.action_repeats = action_repeats
        self.max_steps = ENV_MAX_STEPS
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(NUM_CYTOKINES_CONTROLLED,),
            dtype=np.float32)

        obs_space_high = np.array(all_signals_max)
        # for i in range(NUM_OBSERVTAIONS-1):
        #     obs_space_high = np.hstack((obs_space_high, np.array([10,10,10,10,10,10,10,10,10,10,10])))
        for i in range(NUM_OBSERVTAIONS-1):
            obs_space_high = np.hstack((obs_space_high,all_signals_max))
        print(obs_space_high.shape)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=obs_space_high,
            shape=obs_space_high.shape,
            dtype=np.float32)

        self.reset()
        if self.rendering == "human":
            print("initializing")
            self.initialize_render()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=
        #                 (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    def seed(self, new_seed):
        SIM.setSeed(self.ptrToEnv, new_seed)

    def step(self, action):
    # Execute one time step within the environment, repeat for number of simulation steps and return average
        self.RL_step += 1
        dead = False
        action = self.take_action(action)
        self.cytokine_history = SIM.getAllSignalsReturn(self.ptrToEnv)[[0,2,3,4,5,12,13,14,15,16,17,18],:]
        self.oxydef_history = SIM.getAllSignalsReturn(self.ptrToEnv)[0,:]
        self.current_step = SIM.getSimulationStep(self.ptrToEnv)
        done = self.calculate_done()
        reward = self.calculate_reward(action)
        obs = self.next_observation()
        self.render(action)
        for num_repeats in range(self.action_repeats - 1):
            if done:
                if self.oxydef_history[self.current_step] > MAX_OXYDEF:
                    dead = True
                break
            action = self.take_action(action)
            self.cytokine_history = SIM.getAllSignalsReturn(self.ptrToEnv)[[0,2,3,4,5,12,13,14,15,16,17,18],:]
            self.oxydef_history = SIM.getAllSignalsReturn(self.ptrToEnv)[0,:]
            self.current_step = SIM.getSimulationStep(self.ptrToEnv)

            done = self.calculate_done()
            reward += self.calculate_reward(action)
            obs = np.add(obs,self.next_observation())
            self.render(action)

        # obs /= num_repeats+1
        # reward /= num_repeats+1
        return obs, reward, done, {"Dead":dead}

    def take_action(self,action_vector):
        action_vector = np.squeeze(action_vector)
        action = np.zeros(action_vector.shape)
        for i in range(action_vector.shape[0]):
            act = action_vector[i]
            if act >= 0:
                action[i] = (act*9)+1
            else:
                action[i] = act + 1.001
        action = np.clip(action, .001, 10)
        # action = tf.keras.backend.switch(action_vector >= 0, (action_vector*9)+1, action_vector + 1.0001)
        # action = tf.clip_by_value(action, .001, 10)

        self.action_history[:,self.current_step] = action_vector
        self.cytokine_mults = action

        SIM.setTNFmult(self.ptrToEnv, action[0])
        SIM.setsTNFrmult(self.ptrToEnv, action[1])
        SIM.setIL10mult(self.ptrToEnv, action[2])
        SIM.setGCSFmult(self.ptrToEnv, action[3])
        SIM.setIFNgmult(self.ptrToEnv, action[4])
        SIM.setPAFmult(self.ptrToEnv, action[5])
        SIM.setIL1mult(self.ptrToEnv, action[6])
        SIM.setIL4mult(self.ptrToEnv, action[7])
        SIM.setIL8mult(self.ptrToEnv, action[8])
        SIM.setIL12mult(self.ptrToEnv, action[9])
        SIM.setsIL1rmult(self.ptrToEnv, action[10])

        SIM.singleStep(self.ptrToEnv)

        return action_vector

    def next_observation(self):
        cytokines = self.cytokine_history[:,self.current_step-NUM_OBSERVTAIONS:self.current_step]
        observation = cytokines
        for i in range(observation.shape[0]):
            observation[i,:] = observation[i,:] / self.observation_space.high[i]
        observation = np.squeeze(observation).flatten()

        return observation

    def calculate_done(self):
        DONE = 0
        if self.oxydef_history[self.current_step] < 10:
            DONE = 1
        if self.oxydef_history[self.current_step] > MAX_OXYDEF:
            DONE = 1
        if self.current_step == MAX_STEPS:
            DONE = 1
        return bool(DONE)

    def calculate_reward(self, action):
        return_reward = 0
        reward_mult = 0.999 ** self.RL_step
        # return_reward += 1 #bonus for staying alive per step
        if self.oxydef_history[self.current_step] < 10:
            if self.calculate_done():
                return 250 * reward_mult
        if self.oxydef_history[self.current_step] > 8100:
            if self.calculate_done():
                return -250 * reward_mult


        phi = self.phi_mult * -self.oxydef_history[self.current_step]/(101*101)

        if self.phi_prev is not None and self.potential_difference_mult is not None:
            potential_difference = self.potential_difference_mult*(phi - self.phi_prev)
            # print("potential difference: " + str(potential_difference))
        else:
            potential_difference = 0
        self.phi_prev = phi
        return_reward += potential_difference

        if self.action_L1 is not None:
            # print("L1 penalty: " + str(self.action_L1*np.linalg.norm(action, ord=1)))
            return_reward -= self.action_L1*np.linalg.norm(action, ord=1) # L1 penalty

        return float(return_reward)

    def reset(self, OH=.08, IS=4, NRI=2, NIR=2, injNum=27, seed=0, numCytokines=9):
    # Reset the state of the environment to an initial state
        # del self.ptrToEnv
        self.ptrToEnv = createIIRABM(OH, IS, NRI, NIR, injNum, seed, numCytokines)
        for i in range(NUM_OBSERVTAIONS+100):
            SIM.singleStep(self.ptrToEnv)
        self.cytokine_history = SIM.getAllSignalsReturn(self.ptrToEnv)[[0,2,3,4,5,12,13,14,15,16,17,18],:]
        self.oxydef_history = SIM.getAllSignalsReturn(self.ptrToEnv)[0,:]
        self.current_step = SIM.getSimulationStep(self.ptrToEnv)
        self.RL_step = 0
        self.phi_prev = None # Starts at None so each episode's first reward doesn't include a potential function.

        return self.next_observation()

    def render(self, action=None, close=False):
        mode = self.rendering
        if action is None:
            action = self.action_history[:,self.current_step-1]
        np.set_printoptions(precision=3, suppress=True)
        output = "step: {:4.0f}, Oxygen Deficit: {:6.0f}, Mults:{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(self.current_step, SIM.getOxydef(self.ptrToEnv),*self.cytokine_mults)
        if mode == 'human' or mode == 'console':
            print(output, end="\r")
    # Render the environment to the screen
        if mode == 'human':
            self.fig.canvas.restore_region(self.bg)
            self.line.set_data(range(self.current_step), self.oxydef_history[:self.current_step])
            self.ax.draw_artist(self.line)
            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()
            plt.pause(.00000001)

    def initialize_render(self):
        plotx = np.array(range(self.max_steps))
        print(plotx.shape)
        ploty = np.zeros((self.max_steps-1))
        ploty = np.append(ploty,MAX_OXYDEF)
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Agent Oxygen Deficit Path")
        self.ax.set_xlabel("Timestep (7min)")
        self.ax.set_ylabel("Oxygen Deficit (arb. units)")
        (self.line,) = self.ax.plot(plotx, ploty, animated=True)
        self.ax.axhspan(0, 2750, facecolor='green', alpha=0.4)
        self.ax.axhspan(2750, 6000, facecolor='yellow', alpha=0.4)
        self.ax.axhspan(6000, MAX_OXYDEF, facecolor='red', alpha=0.4)
        self.ax.set_xlim(-10,self.max_steps)
        self.ax.set_ylim(0,MAX_OXYDEF)
        plt.show(block=False)
        plt.pause(0.0001)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.line)
        self.fig.canvas.blit(self.fig.bbox)
        plt.pause(.0001)
