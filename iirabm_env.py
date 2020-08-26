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
MAX_STEPS = 3500
NUM_CYTOKINES_CONTROLLED = 11
NUM_OBSERVTAIONS = 1
# OBS_VEC_SHAPE = NUM_CYTOKINES*((NUM_OBSERVTAIONS*2)-1)
all_signals_max = np.array([8164,  250,  118, 1675,  880,  108, 4027,  730, 1232, 2204,   87,   83])


SIM = wrapper_setup.setUpWrapper()

def createIIRABM():
    oxyHeal = ctypes.c_float(.05)
    IS = ctypes.c_int(4)
    NRI = ctypes.c_int(2)
    NIR = ctypes.c_int(2)
    injNum = ctypes.c_int(27)
    seed = ctypes.c_int(1)
    numCytokines = ctypes.c_int(9)
    internalParameterization = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    instance = SIM.CreateInstance(oxyHeal,IS,NRI,NIR,injNum,seed,numCytokines, internalParameterization.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return instance

class Iirabm_Environment(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, rendering=None):
        super(Iirabm_Environment, self).__init__()

        self.ptrToEnv = None
        self.reward_range = (-100,100)
        self.cytokine_history = np.zeros((12,10000))
        self.cytokine_mults = np.zeros((11,1))
        self.oxydef_history = np.zeros((1,10000))
        self.action_history = np.zeros((11,10000))
        self.current_step = 0
        self.rendering = rendering
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(NUM_CYTOKINES_CONTROLLED,),
            dtype=np.float32)

        obs_space_high = np.array(all_signals_max)
        # for i in range(NUM_OBSERVTAIONS-1):
        #     obs_space_high = np.hstack((obs_space_high, np.array([10,10,10,10,10,10,10,10,10,10,10])))
        # for i in range(NUM_OBSERVTAIONS):
        #     obs_space_high = np.hstack((obs_space_high,all_signals_max[[0,2,3,4,5,12,13,14,15,16,17,18]]))
        print(obs_space_high.shape)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=obs_space_high,
            shape=obs_space_high.shape,
            dtype=np.float32)

        if self.rendering == "human":
            print("initializing")
            self.initialize_render()
            # self.fig, self.ax, self. line, self.bg = initialize_render()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=
        #                 (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def step(self, action):
    # Execute one time step within the environment

        action = self.take_action(action)
        self.cytokine_history = SIM.getAllSignalsReturn(self.ptrToEnv)[[0,2,3,4,5,12,13,14,15,16,17,18],:]
        self.oxydef_history = SIM.getAllSignalsReturn(self.ptrToEnv)[0,:]
        self.current_step = SIM.getSimulationStep(self.ptrToEnv)
        # print("step: " + str(self.current_step) + ", Oxygen Deficit: " + str(np.round(SIM.getOxydef(self.ptrToEnv),0)) + ", Mults: " + str(np.round(action,2)),end="             \r")
        done = self.calculate_done()
        reward = self.calculate_reward()
        obs = self.next_observation()
        self.render(action)
        return obs, reward, done, {}

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
        observation = np.squeeze(observation)

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

    def calculate_reward(self):
        return_reward = 0
        if self.current_step > 100:
            # negative change from last step ie oxydef goes down reward goes up
            return_reward = self.oxydef_history[self.current_step-1] - self.oxydef_history[self.current_step]

        # return_reward += 1 #bonus for staying alive per step
        # if self.oxydef_history[self.current_step] < 2750:
        #     return_reward += 2
        #     if self.calculate_done():
        #         return_reward += 98
        #         # if it lives then total +100 reward
        #
        # if self.oxydef_history[self.current_step] > 6000:
        #     return_reward -= 0.5
        #     if self.calculate_done():
        #         return_reward -= 99.5
                # if it dies then total -100 reward

        return float(return_reward)

    def reset(self):
    # Reset the state of the environment to an initial state
        self.ptrToEnv = createIIRABM()
        for i in range(NUM_OBSERVTAIONS+5):
            SIM.singleStep(self.ptrToEnv)
        self.cytokine_history = SIM.getAllSignalsReturn(self.ptrToEnv)[[0,2,3,4,5,12,13,14,15,16,17,18],:]
        self.oxydef_history = SIM.getAllSignalsReturn(self.ptrToEnv)[0,:]
        self.current_step = SIM.getSimulationStep(self.ptrToEnv)

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
        plotx = np.array(range(MAX_STEPS))
        print(plotx.shape)
        ploty = np.zeros((MAX_STEPS-1))
        ploty = np.append(ploty,MAX_OXYDEF)
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Agent Oxygen Deficit Path")
        self.ax.set_xlabel("Timestep (7min)")
        self.ax.set_ylabel("Oxygen Deficit (arb. units)")
        (self.line,) = self.ax.plot(plotx, ploty, animated=True)
        self.ax.axhspan(0, 2750, facecolor='green', alpha=0.4)
        self.ax.axhspan(2750, 6000, facecolor='yellow', alpha=0.4)
        self.ax.axhspan(6000, MAX_OXYDEF, facecolor='red', alpha=0.4)
        self.ax.set_xlim(-10,MAX_STEPS)
        self.ax.set_ylim(0,MAX_OXYDEF)
        plt.show(block=False)
        plt.pause(0.0001)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.line)
        self.fig.canvas.blit(self.fig.bbox)
        plt.pause(.0001)
