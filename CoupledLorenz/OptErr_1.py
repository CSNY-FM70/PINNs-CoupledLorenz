#!/usr/bin/env python3

import numpy as np
import deepxde as dde
import Coupled_Lorenz_Inverse as CLI

########## Constants ##########
# t0, T, DriverPar, RespPar, CoupPar, InitV0
Physics = [0.0, 3.0, [10., 28., (8./3.)], [10., 12., (8./3.)],
           [0.1, 0.1, 0.1], [13.561, 10.162, 36.951, 3.948, 4.187, 11.531]]
# nDom, nBound, nObs, nTest
GenErr = [400, 2, 100, 400]
# TrainingTime(epochs), LearnRate, Residual Loss, Initial Value Loss, Data Loss
OptErr = [None, None, [1, 1, 1, 1, 1, 1], [
    1, 1, 1, 1, 1, 1], [1e2, 1e2, 1e2, 1e2, 1e2, 1e2]]
# NNarcht, ActFunc, InitWeights
AppErr = [[1] + [100]*4 + [6], 'swish', '']
###############################

initializer = ["Glorot uniform", "Glorot normal",
               "He uniform", "He normal", "LeCun uniform", "LeCun normal"]

epochs = []
for i in range(1, 7):
    epochs.append(pow(10, i))

learning_rate = 5e-4  # [1e-4, 1e-5]

savePath_Data = "OptError_Data_Plots/OptError_N" + \
    str(GenErr[0])+"_"+str(GenErr[2])+"_"+str(learning_rate)+".txt"

# Simulation loop
L_i = np.zeros(len(initializer))
means = np.zeros(len(epochs))
standard_var = np.zeros(len(epochs))
with open(savePath_Data, "w") as file:
    for e in range(len(epochs)):
        L_i.fill(0)
        for k in range(len(initializer)):
            OptErr[0] = epochs[e]
            OptErr[1] = learning_rate
            AppErr[2] = initializer[k]
            model, sample_y, sample_t = CLI.main(
                Physics, AppErr, GenErr, OptErr)
            y_predict = model.predict(sample_t)
            print("L2 Rel Error: ", dde.metrics.l2_relative_error(
                sample_y, y_predict))
            L_i[k] = dde.metrics.l2_relative_error(sample_y, y_predict)
            # print(
            #    f"learning rate = {learn_rate}, epochs = {epochs[e]}, initializer = {initializer[k]}")
        means[e] = np.mean(L_i)
        standard_var[e] = np.std(L_i)
        # Output data to a .txt file
        content = [str(epochs[e]), " ", str(means[e]),
                   " ", str(standard_var[e]), "\n"]
        file.writelines(content)

file.close()
