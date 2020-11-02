import numpy as np
import os
import math


OH=.08 #.05-.15
OH_arr = [.05, .06, .07, .08, .09, .10, .11, .12, .13, .14]
IS=4 # 1-10
IS_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NRI=2 #1-10
NRI_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NIR=2 #1-4
NIR_arr = [1, 2, 3, 4]
injNum=27 #25-35
injNum_arr = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

files = os.listdir("./ParamSweep")

try:
    moderate = open("edge_parameters_moderate.txt", "x")
    moderate.close()
except:
    pass
moderate = open("edge_parameters_moderate.txt", "w")
moderate.write("OH   IS   NRI   NIR   injNum   Mortality Percent\n")

try:
    deadly = open("edge_parameters_deadly.txt", "x")
    deadly.close()
except:
    pass
deadly = open("edge_parameters_deadly.txt", "w")
deadly.write("OH   IS   NRI   NIR   injNum   Mortality Percent\n")

try:
    fatal = open("edge_parameters_fatal.txt", "x")
    fatal.close()
except:
    pass
fatal = open("edge_parameters_fatal.txt", "w")
fatal.write("OH   IS   NRI   NIR   injNum   Timesteps\n")


for file in files:
    if file.startswith("action"):
        continue
    else:

        ID, param_num = file[:-4].split('_')[2:]
        ID, param_num = int(ID),int(param_num)

        # # EXPECT 400 cores
        # NIR_ind = math.floor(ID/100)
        # temp_ID = ID%100 #0-99
        # injNum_ind = math.floor(temp_ID/10)
        # NRI_ind = temp_ID%10
        # OH_ind = math.floor(param_num/10)
        # IS_ind = param_num%10
        #
        # NIR = NIR_arr[NIR_ind]
        # injNum = injNum_arr[injNum_ind]
        # NRI = NRI_arr[NRI_ind]
        # OH = OH_arr[OH_ind]
        # IS = IS_arr[IS_ind]

        # #EXPECT 40 cores
        # NIR_ind = math.floor(ID/10)
        # injNum_ind = ID%10 #0-99
        # OH_ind = math.floor(param_num/100)
        # NRI_ind = math.floor(param_num/10)
        # IS_ind = param_num%10
        #
        # NIR = NIR_arr[NIR_ind]
        # injNum = injNum_arr[injNum_ind]
        # NRI = NRI_arr[NRI_ind]
        # OH = OH_arr[OH_ind]
        # IS = IS_arr[IS_ind]

        # """ ---------- USE THIS ONE UNTIL RECURRENT INJURY BUG IS FIXED ---------- """
        # # EXPECT 40 cores
        # NIR_ind = math.floor(ID/10)
        # injNum_ind = ID%10 #0-99
        #
        # OH_ind = math.floor(param_num/10)          #0-9
        # IS_ind = param_num%10
        #
        # NIR = NIR_arr[NIR_ind]
        # injNum = injNum_arr[injNum_ind]
        # NRI = 0
        # OH = OH_arr[OH_ind]
        # IS = IS_arr[IS_ind]

        """ ---------- USE THIS ONE UNTIL RECURRENT INJURY BUG IS FIXED ---------- """
        # EXPECT 400 cores
        NIR_ind = math.floor(ID/100)
        temp_ID = ID%100 #0-99
        injNum_ind = math.floor(temp_ID/10)
        OH_ind = temp_ID%10
        IS_ind = param_num%10

        NIR = NIR_arr[NIR_ind]
        injNum = injNum_arr[injNum_ind]
        NRI = 0
        OH = OH_arr[OH_ind]
        IS = IS_arr[IS_ind]




        data = np.load("./ParamSweep/" + file)
        heal_count = 0
        death_count = 0
        for i in range(data.shape[2]):
            if np.any(data[0,100:,i] < 100):
                heal_count += 1
            elif np.any(data[0,100:,i] > 8100):
                death_count += 1
            else:
                death_count += 1

        # if mortality rate is between 75% and 85%
        if death_count/data.shape[2] <= 0.85 and death_count/data.shape[2] > 0.75:
            # print("Mortality Percent: {:.3f}%".format(100 * death_count/data.shape[2]))
            # print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
            moderate.write("{:<5.2f}{:<5d}{:<6d}{:<6d}{:<9d}{:<.2f}\n".format(OH, IS, NRI, NIR, injNum, (100 * death_count/data.shape[2])))
        # if mortality rate is between 75% and 85%
        if death_count/data.shape[2] <= 0.99 and death_count/data.shape[2] > 0.85:
            # print("Mortality Percent: {:.3f}%".format(100 * death_count/data.shape[2]))
            # print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
            deadly.write("{:<5.2f}{:<5d}{:<6d}{:<6d}{:<9d}{:<.2f}\n".format(OH, IS, NRI, NIR, injNum, (100 * death_count/data.shape[2])))
        # 100% mortality rate, but takes more than 2500 steps to get there (on average)
        if death_count/data.shape[2] > 0.99 and np.count_nonzero(~np.isnan(data[0,:,:]))/data.shape[2] > 3500:
            timesteps = np.count_nonzero(~np.isnan(data[0,:,:]))/data.shape[2]
            print("Mortality Percent: {:.3f}% --- Timesteps: {:4.0f}".format(100 * death_count/data.shape[2], timesteps))
            print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
            fatal.write("{:<5.2f}{:<5d}{:<6d}{:<6d}{:<9d}{:<4f}\n".format(OH, IS, NRI, NIR, injNum, timesteps))
fatal.close()
deadly.close()
moderate.close()
