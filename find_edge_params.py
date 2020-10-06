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
injNum_arr = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]

files = os.listdir("./ParamSweep")

try:
    f = open("edge_parameters.txt", "x")
    f.close()
except:
    pass
f = open("edge_parameters.txt", "w")
f.write("OH   IS   NRI   NIR   injNum\n")


for file in files:
    if file.startswith("action"):
        continue
    else:

        ID, param_num = file[:-4].split('_')[2:]
        ID, param_num = int(ID),int(param_num)

        # EXPECT 400 cores
        NIR_ind = math.floor(ID/100)
        temp_ID = ID%100 #0-99
        injNum_ind = math.floor(temp_ID/10)
        NRI_ind = temp_ID%10
        OH_ind = math.floor(param_num/10)
        IS_ind = param_num%10

        NIR = NIR_arr[NIR_ind]
        injNum = injNum_arr[injNum_ind]
        NRI = NRI_arr[NRI_ind]
        OH = OH_arr[OH_ind]
        IS = IS_arr[IS_ind]

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
        # if percent of healing is between 15% and 25%
        if heal_count/data.shape[2] < 0.25 and heal_count/data.shape[2] > 0.15:
            print("Heal Percent: {:.3f}%".format(100 * heal_count/data.shape[2]))
            print("OH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}".format(OH, IS, NRI, NIR, injNum))
            f.write("{:<5.2f}{:<5d}{:<6d}{:<6d}{:<d}\n".format(OH, IS, NRI, NIR, injNum))
f.close()
