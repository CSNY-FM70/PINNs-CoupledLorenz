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
OptErr = [120000, 5e-4, [None]*6, [1, 1, 1, 1, 1, 1], [None]*6]
# NNarcht, ActFunc, InitWeights
AppErr = [[1] + [100]*4 + [6], 'swish', '']
###############################

# AppError variables 1 8 50 110
initializer = ["Glorot uniform", "Glorot normal",
               "He uniform", "He normal", "LeCun uniform", "LeCun normal"]

residual = []
data = []

for i in range(-2, 0):
    residual.append(pow(10, i))

for i in range(0, 2):
    data.append(pow(10, i))

savePath_Data = "OptError_Data_Plots/OptError_ResidualXData" + \
    "_"+str(residual[0])+"X"+str(residual[-1])+"_mixed.txt"
L_i = np.zeros(len(initializer))
means = np.zeros((len(residual), len(data)))
standard_var = np.zeros((len(residual), len(data)))

# Simulation loop
with open(savePath_Data, "w") as file:
    for i in range(len(residual)):
        for e in range(len(data)):
            L_i.fill(0)
            for k in range(len(initializer)):
                OptErr[2] = [residual[i]]*6
                OptErr[4] = [data[e]]*6
                AppErr[2] = initializer[k]
                model, sample_y, sample_t = CLI.main(
                    Physics, AppErr, GenErr, OptErr)
                y_predict = model.predict(sample_t)
                print("L2 Rel Error: ", dde.metrics.l2_relative_error(
                    sample_y, y_predict))
                L_i[k] = dde.metrics.l2_relative_error(sample_y, y_predict)
                # print(
                #    f"residual = {residual[i]}, data = {data[e]}, initializer = {initializer[k]}")
            means[i, e] = np.mean(L_i)
            standard_var[i, e] = np.std(L_i)
            # Output data to a .txt file
            content = [str(residual[i]), ",", str(data[e]), ",", str(
                means[i, e]), ",", str(standard_var[i, e]), "\n"]
            file.writelines(content)

file.close()
