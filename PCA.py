import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import decomposition

heal_color = "green"
timeout_color = "magenta"
death_color = "red"
infection_color = "black"

CUTOFF_TIME = 3000

action_data = np.load("./action_data.npy")
action_data = action_data[:,:,:]
sIL1r_action = action_data[-1,:,:]
full_data = np.load("./cytokine_data.npy")
cytokine_data = full_data[[0,2,3,4,5,12,13,14,15,16,17,18],:,:]

heal_index = np.zeros(cytokine_data.shape[2])
for i in range(len(heal_index)):
    if np.any(cytokine_data[0,100:,i] < 100):
        heal_index[i] = 1
    if np.any(cytokine_data[0,100:,i] > 8100):
        heal_index[i] = 2
        print(i)
high_low_index = np.zeros(action_data.shape[2])

index = 0
X = np.zeros((cytokine_data.shape[0], cytokine_data.shape[2]))
for i in range(cytokine_data.shape[2]):
    if not np.isnan(cytokine_data[0,CUTOFF_TIME,i]):
        X[:,index] = cytokine_data[:,CUTOFF_TIME,i]
        heal_index[index] = heal_index[i]
        if sIL1r_action[CUTOFF_TIME,i] > 7.5:
            high_low_index[index] = 1
        elif sIL1r_action[CUTOFF_TIME,i] < .1:
            high_low_index[index] = -1
        index += 1
heal_index = heal_index[:index]
high_low_index = high_low_index[:index]
y = X[-1,:index]
X = X[1:,:index]
# X = X / np.max(np.absolute(X), axis=0)
print(X.shape)

pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(X.T)
print(pca)
print(X.shape)

timeout_PC1 = np.zeros((X.shape[0],2))
timeout_PC2 = np.zeros((X.shape[0],2))
heal_PC1 = np.zeros((X.shape[0],2))
heal_PC2 = np.zeros((X.shape[0],2))
death_PC1 = np.zeros((X.shape[0],2))
death_PC2 = np.zeros((X.shape[0],2))

PCA_timeout_index = 0
PCA_heal_index = 0
PCA_death_index = 0
for i in range(heal_index.shape[0]):
    if heal_index[i] == 0:
        timeout_PC1[PCA_timeout_index,0] = X[i,0]
        timeout_PC2[PCA_timeout_index,0] = X[i,1]
        timeout_PC1[PCA_timeout_index,1] = high_low_index[i]
        timeout_PC2[PCA_timeout_index,1] = high_low_index[i]
        PCA_timeout_index += 1

    if heal_index[i] == 1:
        heal_PC1[PCA_heal_index,0] = X[i,0]
        heal_PC2[PCA_heal_index,0] = X[i,1]
        heal_PC1[PCA_heal_index,1] = high_low_index[i]
        heal_PC2[PCA_heal_index,1] = high_low_index[i]
        PCA_heal_index += 1

    if heal_index[i] == 2:
        death_PC1[PCA_death_index] = X[i,0]
        death_PC2[PCA_death_index] = X[i,1]
        death_PC1[PCA_death_index] = high_low_index[i]
        death_PC2[PCA_death_index] = high_low_index[i]
        PCA_death_index += 1

timeout_PC1 = timeout_PC1[:PCA_timeout_index]
timeout_PC2 = timeout_PC2[:PCA_timeout_index]
heal_PC1 = heal_PC1[:PCA_heal_index]
heal_PC2 = heal_PC2[:PCA_heal_index]
death_PC1 = death_PC1[:PCA_death_index]
death_PC2 = death_PC2[:PCA_death_index]
print(death_PC1.shape)
print(timeout_PC1.shape)
print(heal_PC1.shape)
timeout_high_PC1 = timeout_PC1[timeout_PC1[:,1] == 1]
timeout_low_PC1 = timeout_PC1[timeout_PC1[:,1] == -1]
timeout_med_PC1 = timeout_PC1[timeout_PC1[:,1] == 0]
timeout_high_PC2 = timeout_PC2[timeout_PC2[:,1] == 1]
timeout_low_PC2 = timeout_PC2[timeout_PC2[:,1] == -1]
timeout_med_PC2 = timeout_PC2[timeout_PC2[:,1] == 0]

heal_high_PC1 = heal_PC1[heal_PC1[:,1] == 1]
heal_low_PC1 = heal_PC1[heal_PC1[:,1] == -1]
heal_med_PC1 = heal_PC1[heal_PC1[:,1] == 0]
heal_high_PC2 = heal_PC2[heal_PC2[:,1] == 1]
heal_low_PC2 = heal_PC2[heal_PC2[:,1] == -1]
heal_med_PC2 = heal_PC2[heal_PC2[:,1] == 0]

death_high_PC1 = death_PC1[death_PC1[:,1] == 1]
death_low_PC1 = death_PC1[death_PC1[:,1] == -1]
death_med_PC1 = death_PC1[death_PC1[:,1] == 0]
death_high_PC2 = death_PC2[death_PC2[:,1] == 1]
death_low_PC2 = death_PC2[death_PC2[:,1] == -1]
death_med_PC2 = death_PC2[death_PC2[:,1] == 0]

plt.figure(figsize=(10,6))
plt.scatter(timeout_high_PC1, timeout_high_PC2, c=timeout_color, label="timeout_action_high", marker='^')
plt.scatter(timeout_low_PC1, timeout_low_PC2, c=timeout_color, label="timeout_action_low", marker='v')
plt.scatter(timeout_med_PC1, timeout_med_PC2, c=timeout_color, label="timeout_action_med", marker='*')
plt.scatter(heal_high_PC1, heal_high_PC2, c=heal_color, label="heal_action_high", marker='^')
plt.scatter(heal_low_PC1, heal_low_PC2, c=heal_color, label="heal_action_low", marker='v')
plt.scatter(heal_med_PC1, heal_med_PC2, c=heal_color, label="heal_action_med", marker='*')
plt.scatter(death_high_PC1, death_high_PC2, c=death_color, label="dead_action_high", marker='^')
plt.scatter(death_low_PC1, death_low_PC2, c=death_color, label="dead_action_low", marker='v')
plt.scatter(death_med_PC1, death_med_PC2, c=death_color, label="dead_action_med", marker='*')
plt.title("Principal component alalysis for sIL1r actions")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.show()
