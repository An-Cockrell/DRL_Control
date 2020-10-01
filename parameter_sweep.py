from iirabm_env import Iirabm_Environment
import numpy as np
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank%(20*8)
file = open("/users/d/l/dlarie/drl_control/good_params.txt", "a")
f.write(str(rank))
f.close()
is_rank = math.floor(rank/20)
oh_rank = rank%20

action = [1,1,1,1,1,1,1,1,1,1]

oxyHeal = np.linspace(0.05, 1, 20)
injNum = np.array([5, 10, 15, 20, 25, 30, 35, 40])
NIR = np.array([1, 2, 3, 4])
IS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
NRI = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# def reset(self, OH=.05, IS=4, NRI=2, NIR=2, injNum=2, seed=0, numCytokines=9):
middle_probability = []
params_tested = 0
env = Iirabm_Environment(rendering=None, action_repeats=1, ENV_MAX_STEPS=5000, action_L1=0.01)
default_action = [0,0,0,0,0,0,0,0,0,0,0]

OH = oxyHeal[oh_rank]
infect_spread = IS[is_rank]
for nri in NRI:
    for nir in NIR:
        for inj in injNum:
            counter = 0
            params_tested += 1
            file = open("/users/d/l/dlarie/drl_control/good_params.txt", "a")
            f.write(params_tested)
            f.close()
            for seed in range(100):
                env.reset(OH, infect_spread, nri, nir, inj, seed)
                done = False
                while not done:
                    _, _, done, info = env.step(default_action)
                if not info["Dead"]:
                    counter += 1
            if counter > 40 and counter < 60:
                good_params = {
                    "OH": OH,
                    "IS": infect_spread,
                    "NRI": nri,
                    "NIR": nir,
                    "INJ": inj
                }
                middle_probability.append(good_params)
                file = open("/users/d/l/dlarie/drl_control/good_params.txt", "a")
                f.write(good_params)
                f.close()
print(middle_probability)
