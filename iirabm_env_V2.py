# IIRABM DRL Environment-V2
# Created by Dale Larie 7/29/2020
# Questions to daleblarie@gmail.com

# following this:
# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
"""
This file contains all the functions to set up and use the iirabm gym environment. The following functions are the ones that should be used to interact with the environment:
    seed
    step
    reset
"""

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import wrapper_setup

import gym
import math

import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf

# PLOTTING VARS PLACEHOLDERS
globalLine = None
globalAx = None
globalFig = None
globalBG = None

time_for_action = 12 * 60 #12 hours
steps_for_action = math.floor(time_for_action/6)
steps_until_regression = 15


HEAL_OXYDEF = 100               #Oxydef level where therapy stops
FINAL_HEAL_OXYDEF = 50          #Oxydef level where subject is considered totally healed
MAX_OXYDEF = 8160               #Oxydef level where subject is considered dead
MAX_STEPS = 9999                #Max number of steps the simulation will play
NUM_CYTOKINES_CONTROLLED = 11   #Number of cytokines controlled by the agent
NUM_OBSERVTAIONS = 1            #Total number of timesteps observed by the agent

# Estimated maximum values for Oxydef, and 11 cytokines
all_signals_max = np.array([7851, 2350, 83, 1338, 8684, 210, 4123, 721, 1230, 1802,139, 103])
# all_signals_max = np.array([4230, 60, 15, 150, 120, 15, 200, 60, 70, 100, 20, 10])

SIM = wrapper_setup.setUpWrapper()

def createIIRABM(OH, IS, NRI, NIR, injNum, seed, numCytokines=9):
    # Function to create a simulation
        # OH - oxyheal
        # IS - infection spread
        # NRI - Number of recurrent injuries
        # NIR - Number of infection num_repeats
        # injNum - Injury size / number of inital injuries
        # seed - Random seed for the simulation
        # numCytokines - Not certain what this variable does, but it was an input for the IIRABM so set it equal to 9
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

    def __init__(self, rendering=None, action_repeats=steps_for_action, ENV_MAX_STEPS=MAX_STEPS, action_L1=None, potential_difference_mult=None, phi_mult=1, regression_cooldown=steps_until_regression):
        # Initialization function for the iirabm_env
            # Rendering - modes are "human", "console", and None
                # "human" - uses matplotlib to plot the oxydef of the simulation live. Also prints to the console in the same way as "console"
                # "console" - prints the current step, the current oxydef, and the multipliers being applied to the 11 control cytokines
                # None - does not print live updates
            # action_repeats - number of environment steps the agent action is repeated for each agent step
            # ENV_MAX_STEPS - the max number of environment steps (frames) before timeout
            # action_L1 - multiplier to discourage large actions
            # potential_difference_mult - multiplier for reward for difference in phi values
            # phi_mult - multiplier for oxydef/(101*101)
        super(Iirabm_Environment, self).__init__()

        self.oxydef_regressor = tf.keras.models.load_model("oxydef_regressor.h5")
        self.oxydef_regressor_hist = np.zeros((1,10000))
        self.obs_hist = np.zeros((11,10000))
        self.neutral_action = [0,0,0,0,0,0,0,0,0,0,0]    # Action that will set all multipliers to 1, essentially stopping therapy
        self.steps_until_regression = regression_cooldown
        self.reward_range = (-250,250)                              # Range of rewads at each step
        self.cytokine_history = np.zeros((12,10000))                # Placeholder for history of oxydef and 11 cytokines for this simulation run
        self.cytokine_mults = np.zeros((11,1))                      # Placeholder for current multipliers for cytokines (current action)
        self.oxydef_history = np.zeros((1,10000))                   # Placeholder for oxygen deficit history for this simulation run
        self.action_history = np.zeros((11,10000))                  # Placeholder for history of actions for this simulation run
        self.full_history = np.zeros((20,10000))                    # Placeholder for full outout of iirabm for this simulation run
        self.current_step = 0                                       # The current step of the simulation environment
        self.RL_step = 0                                            # The current step of the agent
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

        # offset and scaling factors for observation regularization
        obs_max = all_signals_max
        self.input_offset = (obs_max)/2
        self.input_scale = (obs_max)/2

        # observation space is oxydef and 11 cytokine values
        obs_space_high = np.zeros(12)
        obs_space_high[0] = 101*101
        obs_space_high[1:] = np.inf
        obs_max = obs_space_high

        # stack the observation space with the number of frames to be observed
        for i in range(NUM_OBSERVTAIONS-1):
            obs_space_high = np.vstack((obs_space_high,obs_max))
        obs_space_high = obs_space_high.T.flatten()

        self.observation_space = gym.spaces.Box(
            low=0,
            high=obs_space_high,
            shape=obs_space_high.shape,
            dtype=np.float32)

        # call reset to initialze all the variables so it is ready to simulatie
        self.reset(display=False)

        if self.rendering == "human":
            print("initializing")
            self.initialize_render()

    def seed(self, new_seed):
        # function to properly seed the environment
        SIM.setSeed(self.ptrToEnv, new_seed)

    def get_current_cyto(self):
        # function to get all the data for the current frame
        # returns oxydef and 11 cytokine values
        return SIM.getAllSignalsReturn(self.ptrToEnv)[[0,2,3,4,5,12,13,14,15,16,17,18],self.current_step]

    def step(self, action):
        # function to execute the chosen action for the number of action_repeats
        # action - an (11,1) array of actions in range of (-1,1)
        # returns:
            # the total of the observations over the course of the action_repeats
            # the total reward over the course of the action_repeats
            # the done as a boolean if the patient dies or heals
        self.RL_step += 1                   # increment the agent step
        dead = False                        # initialize dead as False for info
        healed = False                      # intialize healed as False for info
        timeout = False                     # intialize timeout as False for info
        self.take_action(action)             # apply the multipliers given
        # set the current step and add data to the proper history placeholders
        self.current_step = SIM.getSimulationStep(self.ptrToEnv)
        starting_step = self.current_step

        self.cytokine_history[:,self.current_step] = self.get_current_cyto()
        self.full_history[:,self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[:,self.current_step]
        self.oxydef_history[self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[0,self.current_step]
        done = self.calculate_done()            # calculate done - 1=healed, 2=dead, 3=timeout
        reward = self.calculate_reward(action)  # calculate the reward for this frame
        obs = self.next_observation()           # get the oxydef and cytokine values for this frame


        # repeat these actions for the number of action_repeats-1 because the action was already taken once
        for num_repeats in range(self.action_repeats - 1):
            if done > 0:
                break
            self.take_action(action)
            self.current_step = SIM.getSimulationStep(self.ptrToEnv)
            self.cytokine_history[:,self.current_step] = self.get_current_cyto()
            self.full_history[:,self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[:,self.current_step]
            self.oxydef_history[self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[0,self.current_step]

            reward += self.calculate_reward(action)
            obs = np.add(obs,self.next_observation())
            done = self.calculate_done()


        for regression_cooldown in range(self.steps_until_regression):
            if done > 0:
                break
            self.take_action(self.neutral_action)
            self.current_step = SIM.getSimulationStep(self.ptrToEnv)
            self.cytokine_history[:,self.current_step] = self.get_current_cyto()
            self.full_history[:,self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[:,self.current_step]
            self.oxydef_history[self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[0,self.current_step]

            reward += self.calculate_reward(self.neutral_action)
            obs = np.add(obs,self.next_observation())
            done = self.calculate_done()


        step_count = self.current_step - starting_step
        if step_count == 0:
            step_count = 1
        reward /= step_count
        obs /= step_count

        self.obs_hist[:,self.RL_step] = np.squeeze(obs)
        if NUM_OBSERVTAIONS > 1:
            if self.RL_step >= NUM_OBSERVTAIONS:
                prev_obs = self.obs_hist[:,self.RL_step-(NUM_OBSERVTAIONS-1):self.RL_step]
                prev_obs = np.flip(prev_obs, axis=1)
                obs = np.hstack((obs,prev_obs))
            else:
                obs = np.hstack((obs,np.zeros((obs.shape[0],NUM_OBSERVTAIONS-1))))

        oxydef = self.oxydef_regressor.predict(np.expand_dims(self.get_current_cyto()[1:], axis=0))
        self.oxydef_regressor_hist[0,self.RL_step] = oxydef

        if NUM_OBSERVTAIONS > 1:
            if self.RL_step >= NUM_OBSERVTAIONS:
                prev_steps = self.oxydef_regressor_hist[0,self.RL_step-(NUM_OBSERVTAIONS-1):self.RL_step]
                prev_steps = np.flip(prev_steps, axis=0)
                oxydef = np.hstack((oxydef[0],prev_steps))
            else:
                oxydef = np.hstack((oxydef[0],np.squeeze(np.zeros(NUM_OBSERVTAIONS-1))))

        obs = np.vstack((oxydef, obs))
        # attempt to make the observation 0 centered and normalized to (-1,1), then flatten to 1D
        for i in range(obs.shape[1]):
            obs[:,i] = (obs[:,i] - self.input_offset)/self.input_scale
        obs = obs.T.flatten()
        # change healed/dead/timeout based on done value and cast done to a boolean
        if done == 1:
            healed = True
        if done == 2:
            dead = True
        if done == 3:
            timeout = True
        done = bool(done)
        # set info vars
        info = {"dead":dead,
        "healed":healed,
        "timeout":timeout,
        "transient_infection":self.transient_infection,
        "step":self.current_step,
        "oxydef":self.oxydef_history[self.current_step]}

        return obs, reward, done, info

    def take_action(self,action_vector, testing_transient=False):
        # function to apply the chosen action to the simulation and step environment forward one frame
            # action_vector - a vector of shape (11,) with values in range(-1,1) that represent the chosen actions
            # testing_transient - a boolean used when the damage has been healed to confirm that no transient infection will return and kill the patient
        action_vector = np.squeeze(action_vector)       # reduce dimensions of chosen action to be 1D
        action = np.zeros(action_vector.shape)          # placeholder for the actions that will be taken

        for i in range(action_vector.shape[0]):         # for each multiplier in the chosen action convert the action to a multiplier for the iirabm simulation
            act = action_vector[i]
            if act >= 0:
                action[i] = (act*9) +1                 # if the action is > 0, convert to range (1,100)
            else:
                action[i] = act + 1.001                 # if the action is < 0, convert to range (0.001, 1
        action = np.clip(action, .001, 10)

        if not testing_transient:                       # if testing the normal case, add the iirabm multiplier action to the action_history
            self.action_history[:,self.current_step] = action
        self.cytokine_mults = action                    # set cytokine_mults to the current iirabme multiplier action
        # Apply the iirabm multiplier actions
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
        # step the environment forward with the applied action
        SIM.singleStep(self.ptrToEnv)
        if self.rendering == "console" or self.rendering == "human":
            self.render(action_vector)


    def next_observation(self):
        # a function to return the last NUM_OBSERVTAIONS frames as a 1D vector
        cytokines = np.array(self.cytokine_history[1:,self.current_step-1:self.current_step])
        observation = cytokines

        # attempt to make the observation 0 centered and normalized to (-1,1), then flatten to 1D
        # for i in range(observation.shape[1]):
        #     observation[:,i] = (observation[:,i] - self.input_offset[1:])/self.input_scale[1:]
        # observation = observation.flatten()

        return observation

    def calculate_done(self):
        # a function that will return done as an int
            # 0 = not done
            # 1 = Healed
            # 2 = Dead
            # 3 = Env Timeout
        DONE = 0
        if self.oxydef_history[self.current_step] < HEAL_OXYDEF:
            DONE = 1
        if self.oxydef_history[self.current_step] > MAX_OXYDEF:
            DONE = 2
        if self.current_step == MAX_STEPS:
            DONE = 3
        return DONE

    def test_transient_infection(self):
        # a function to test transient infection after damage threshold goes below heal threshold
        # returns boolean of if patient fully heals or timesout/dies
            # True - patient fully healed damage
            # False - patient died or env timeout

        # 100 frame test period to let the simulation try to finish healing on its own
        for _ in range(100):
            if self.calculate_done() == 3:
                break
            self.take_action(self.neutral_action, testing_transient=True)
            self.current_step = SIM.getSimulationStep(self.ptrToEnv)
            self.cytokine_history[:,self.current_step] = self.get_current_cyto()
            self.full_history[:,self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[:,self.current_step]
            self.oxydef_history[self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[0,self.current_step]

        # test if damage is healed every frame until death, final heal, or env timeout
        while self.current_step < MAX_STEPS:
            if SIM.getAllSignalsReturn(self.ptrToEnv)[0,self.current_step] < FINAL_HEAL_OXYDEF:
                self.transient_infection = False
                return True
            if SIM.getAllSignalsReturn(self.ptrToEnv)[0,self.current_step] > MAX_OXYDEF:
                self.transient_infection = True
                # print("WARNING PATIENT DIED AFTER THERAPY")
                return False
            self.take_action(self.neutral_action, testing_transient=True)
            self.current_step = SIM.getSimulationStep(self.ptrToEnv)
            self.cytokine_history[:,self.current_step] = self.get_current_cyto()
            self.full_history[:,self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[:,self.current_step]
            self.oxydef_history[self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[0,self.current_step]
        # print("WARNING PATIENT TIMEOUT WHILE TESTING TRANSIENT INFECTION")
        self.transient_infection = True
        return False


    def calculate_reward(self, action):
        # function to calculate reward for current step
            # action - agent action taken of shape (11,) in range(-1,1)
        # returns heal/death reward or reward based on change in damage, action size, and agent step

        return_reward = 0                               # initialize reward to 0
        reward_mult = 0.998 ** self.RL_step             # reward multiplier gets smaller as the agent takes more steps

        if self.calculate_done():
            if self.oxydef_history[self.current_step] < HEAL_OXYDEF:
                if self.test_transient_infection():     # if healed, test transient_infection
                    return 1000                         # if fully healed, return heal reward
                else:
                    return HEAL_OXYDEF - self.oxydef_history[self.current_step] # if timeout or dies after healing, return a negative reward
            if self.oxydef_history[self.current_step] > MAX_OXYDEF:
                return 1000                            # if dead return death reward

        # if not done, continue calculating reward
        # phi and potential difference are based on the change in oxydef/system damage
        phi = self.phi_mult * -self.oxydef_history[self.current_step]/(101*101)

        if self.phi_prev is not None and self.potential_difference_mult is not None:
            potential_difference = self.potential_difference_mult*(phi - self.phi_prev)
        else:
            potential_difference = 0
        self.phi_prev = phi
        return_reward += potential_difference

        # regularization to discourage large actions
        if self.action_L1 is not None:
            action_penalty = self.action_L1*np.sum(tf.math.abs(action)) # L1 penalty
            return_reward -= action_penalty
            # np.set_printoptions(precision=3)
            # print_act = (np.round(np.array(action),3)).tolist()
            # temp = list([return_reward])
            # temp.extend(print_act)
            # print("{:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}".format(*temp), end="\r")
        # print("Oxydef difference: " + str(potential_difference))
        # print("Action penalty: " + str(action_penalty))
        return float(return_reward * reward_mult)   # return reward discounted by return multiplier
# OH: 0.1, IS: 4, NRI: 0, NIR: 2, injNum: 14 <--- ~80% with pmn oxy turned off
# OH: 0.08, IS: 4, NRI: 0, NIR: 2, injNum: 13
# OH: 0.15, IS: 4, NRI: 0, NIR: 2, injNum: 19
# OH: 0.15, IS: 2, NRI: 0, NIR: 2, injNum: 25
# Mortality Percent: 77.000%
# OH: 0.08, IS: 3, NRI: 0, NIR: 1, injNum: 18
    def reset(self, OH=0.1, IS=8, NRI=0, NIR=1, injNum=15, seed=0, numCytokines=9, display=False):
        if display:
            print("\n\nOH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}\n\n".format(OH, IS, NRI, NIR, injNum))
        # function to reset the state of the environment to an initial state
            # OH - oxyheal
            # IS - infection spread
            # NRI - Number of recurrent injuries
            # NIR - Number of infection repeats
            # injNum - Injury size / number of inital injuries
            # seed - Random seed for the simulation
            # numCytokines - Not certain what this variable does, but it was an input for the IIRABM so set it equal to 9
        # returns the intial observation of the new environment
        self.ptrToEnv = createIIRABM(OH, IS, NRI, NIR, injNum, seed, numCytokines)  # create a new iirabm environment
        for i in range(NUM_OBSERVTAIONS+100):                                       # 100 frame burn in period
            SIM.singleStep(self.ptrToEnv)
        # Set histories for the new simulation
        self.cytokine_history = SIM.getAllSignalsReturn(self.ptrToEnv)[[0,2,3,4,5,12,13,14,15,16,17,18],:]
        self.full_history[:,self.current_step] = SIM.getAllSignalsReturn(self.ptrToEnv)[:,self.current_step]
        self.oxydef_history = SIM.getAllSignalsReturn(self.ptrToEnv)[0,:]
        self.current_step = SIM.getSimulationStep(self.ptrToEnv)
        # Initialize histories in the future of the current step = 0
        self.cytokine_history[:,self.current_step:] = 0
        self.full_history[:, self.current_step:] = 0
        self.oxydef_history[self.current_step:] = 0
        self.RL_step = 0                    # Reset RL step
        self.phi_prev = None                # Starts at None so each episode's first reward doesn't include a potential function.
        self.transient_infection = False    # reset transient infection

        obs = self.next_observation()
        if NUM_OBSERVTAIONS > 1:
            obs = np.hstack((obs,np.zeros((obs.shape[0],NUM_OBSERVTAIONS-1))))
        oxydef = self.oxydef_regressor.predict(np.expand_dims(self.get_current_cyto()[1:], axis=0))
        self.oxydef_regressor_hist[0,self.RL_step] = oxydef
        if NUM_OBSERVTAIONS > 1:
            oxydef = np.hstack((oxydef[0],np.squeeze(np.zeros(NUM_OBSERVTAIONS-1))))

        obs = np.vstack((oxydef, obs))
        # oxydef = np.vstack((oxydef,np.zeros(NUM_OBSERVTAIONS-1)))
        # obs = np.hstack((np.squeeze(oxydef),obs))
        # # attempt to make the observation 0 centered and normalized to (-1,1), then flatten to 1D
        for i in range(obs.shape[1]):
            obs[:,i] = (obs[:,i] - self.input_offset)/self.input_scale
        obs = obs.flatten()
        # print(obs)
        return obs

    def render(self, action=None):
        # function that will render the environment to the screen based on rendering choice
        mode = self.rendering
        if action is None:
            action = self.action_history[:,self.current_step-1]
        np.set_printoptions(precision=3, suppress=True)
        output = "step: {:4.0f}, Oxygen Deficit: {:6.0f}, Mults:{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f},{:6.3f}".format(self.current_step, SIM.getOxydef(self.ptrToEnv),*self.cytokine_mults)
        # Render the environment to the screen
        if mode == 'human' or mode == 'console':
            print(output, end="\r")
        if mode == 'human':
            self.fig.canvas.restore_region(self.bg)
            self.line.set_data(range(self.current_step), self.oxydef_history[:self.current_step])
            self.ax.draw_artist(self.line)
            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()
            plt.pause(.00000001)

    def initialize_render(self):
        # function to set up rendering for human mode
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
