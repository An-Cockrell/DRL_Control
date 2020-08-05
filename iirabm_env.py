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


MAX_OXYDEF = 8160
NUM_CYTOKINES = 11
NUM_OBSERVTAIONS = 5
OBS_VEC_SHAPE = NUM_CYTOKINES*(NUM_OBSERVTAIONS+1)

all_signals_max = np.array([9048.283, 8969.218, 56.453243, 24.432032, 203.75848, 194.54462, 59.198627, 93.91482, 986.0, 465., 133., 227., 176., 432.44122, 425.84256, 79.20918, 220.06897, 217.0821, 11.526534, 43.950306])


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

    def __init__(self):
        super(Iirabm_Environment, self).__init__()

        self.ptrToEnv = None
        self.reward_range = (-5,10000)
        self.cytokine_history = np.zeros((11,10000))
        self.cytokine_mults = np.zeros((11,1))
        self.oxydef_history = np.zeros((1,10000))
        self.current_step = 0
        self.render = False
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = gym.spaces.Box(
            low=0.01,
            high=10,
            shape=(NUM_CYTOKINES,),
            dtype=np.float32)

        obs_space_high = np.array([10,10,10,10,10,10,10,10,10,10,10])
        for i in range(NUM_OBSERVTAIONS):
            obs_space_high = np.hstack((obs_space_high,all_signals_max[[2,3,4,5,12,13,14,15,16,17,18]]))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=obs_space_high,
            shape=(OBS_VEC_SHAPE,),
            dtype=np.float32)
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=
        #                 (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def step(self, action):
    # Execute one time step within the environment
        self.take_action(action)
        self.cytokine_history = SIM.getAllSignalsReturn(self.ptrToEnv)[[2,3,4,5,12,13,14,15,16,17,18],:]
        self.oxydef_history = SIM.getAllSignalsReturn(self.ptrToEnv)[0,:]
        self.current_step = SIM.getSimulationStep(self.ptrToEnv)
        np.set_printoptions(precision=2, suppress=True)

        output = "step: {:4.0f}, Oxygen Deficit: {:5.0f}, Mults: {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}, {:5.2f}".format(self.current_step, SIM.getOxydef(self.ptrToEnv), action[0],action[1],action[2],action[3],action[4],action[5],action[6],action[7],action[8],action[9],action[10])
        if self.render:
            print(output, end="\r")
        # print("step: " + str(self.current_step) + ", Oxygen Deficit: " + str(np.round(SIM.getOxydef(self.ptrToEnv),0)) + ", Mults: " + str(np.round(action,2)),end="             \r")
        reward = self.calculate_reward()
        done = self.calculate_done()
        obs = self._next_observation()

        return obs, reward, done, {}

    def take_action(self,action_vector):
        SIM.setTNFmult(self.ptrToEnv, action_vector[0])
        SIM.setsTNFrmult(self.ptrToEnv, action_vector[1])
        SIM.setIL10mult(self.ptrToEnv, action_vector[2])
        SIM.setGCSFmult(self.ptrToEnv, action_vector[3])
        SIM.setIFNgmult(self.ptrToEnv, action_vector[4])
        SIM.setPAFmult(self.ptrToEnv, action_vector[5])
        SIM.setIL1mult(self.ptrToEnv, action_vector[6])
        SIM.setIL4mult(self.ptrToEnv, action_vector[7])
        SIM.setIL8mult(self.ptrToEnv, action_vector[8])
        SIM.setIL12mult(self.ptrToEnv, action_vector[9])
        SIM.setsIL1rmult(self.ptrToEnv, action_vector[10])

        SIM.singleStep(self.ptrToEnv)

    def _next_observation(self):
        frame = self.cytokine_history[:,self.current_step-NUM_OBSERVTAIONS:self.current_step]
        frame = frame.flatten()
        current_mults = self.cytokine_mults
        observation = np.append(current_mults,frame)
        return observation

    def calculate_done(self):
        DONE = False
        if self.oxydef_history[self.current_step] < 10:
            DONE = True
        if self.oxydef_history[self.current_step] > MAX_OXYDEF:
            DONE = True
        if self.current_step == 9999:
            DONE = True
        return bool(DONE)

    def calculate_reward(self):
        return_reward = 1
        # return_reward = return_reward - self.oxydef_history[self.current_step]
        # return_reward = return_reward / self.current_step
        # # return_reward += self.current_step
        if self.oxydef_history[self.current_step] > 6000:
            return_reward = -5
        # if self.oxydef_history[self.current_step] > 4500:
        #     return_reward = -5
        # if self.oxydef_history[self.current_step] > 6000:
        #     return_reward = -10
        # if self.oxydef_history[self.current_step] > 7500:
        #     return_reward = -100
        if self.oxydef_history[self.current_step] < 2750:
            return_reward = 5
        if self.oxydef_history[self.current_step] < 2500:
            return_reward = 7
        if self.oxydef_history[self.current_step] < 2250:
            return_reward = 10
        if self.oxydef_history[self.current_step] < 2000:
            return_reward = 100
        if self.oxydef_history[self.current_step] < 50:
            return_reward = 10000
        return float(return_reward)

    def reset(self):
    # Reset the state of the environment to an initial state
        self.ptrToEnv = createIIRABM()
        for i in range(NUM_OBSERVTAIONS+5):
            SIM.singleStep(self.ptrToEnv)
        self.cytokine_history = SIM.getAllSignalsReturn(self.ptrToEnv)[[2,3,4,5,12,13,14,15,16,17,18],:]
        self.oxydef_history = SIM.getAllSignalsReturn(self.ptrToEnv)[0,:]
        self.current_step = SIM.getSimulationStep(self.ptrToEnv)

        return self._next_observation()

    def render(self, mode='human', close=False):
    # Render the environment to the screen
        pass
