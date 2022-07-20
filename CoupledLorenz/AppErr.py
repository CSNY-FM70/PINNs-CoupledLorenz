#!/usr/bin/env python3

import numpy as np
import Coupled_Lorenz_Inverse as CLI
import matplotlib
import matplotlib.pyplot as plt
import deepxde as dde

########## Constants ##########
# t0, T, DriverPar, RespPar, CoupPar, InitV0
Physics = [0.0, 3.0, [10., 28., (8./3.)], [10., 12., (8./3.)],
           [0.1, 0.1, 0.1], [13.561, 10.162, 36.951, 3.948, 4.187, 11.531]]
# nDom, nBound, nObs, nTest
GenErr = [400, 2, 100, 400]
# TrainingTime(epochs), LearnRate, Residual Loss, Initial Value Loss, Data Loss
OptErr = [1e5, 5e-4, [1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1], [1e2, 1e2, 1e2, 1e2, 1e2, 1e2]]

###############################

# AppError variables 1 8 50 110
layers = (np.arange(1, 2)).tolist()
# ab 8te (50, 110, 10) und (110, 160, 10)
neurons = (np.arange(50, 160, 10)).tolist()
initializer = ["Glorot uniform", "Glorot normal",
               "He uniform", "He normal", "LeCun uniform", "LeCun normal"]

savePath_Data = "AppErr_Data_Plots/AppError_" + \
    str(layers[-1])+'x'+str(neurons[-1])+'.txt'
#savePath_Plot = "ErrorPlot/AppError_"+str(layers[-1])+'x'+str(neurons[-1])+'.png'

L_i = np.zeros(len(initializer))
means = np.zeros((len(layers), len(neurons)))
standard_var = np.zeros((len(layers), len(neurons)))

# Simulation loop
with open(savePath_Data, "w") as file:
    for i in range(len(layers)):
        for j in range(len(neurons)):
            L_i.fill(0)
            for k in range(len(initializer)):
                ApproxTest = [[1] + [neurons[j]] *
                              layers[i] + [6], 'swish', initializer[k]]
                model, sample_y, sample_t = CLI.main(
                    Physics, ApproxTest, GenErr, OptErr)
                y_predict = model.predict(sample_t)
                print("L2 Rel Error: ", dde.metrics.l2_relative_error(
                    sample_y, y_predict))
                L_i[k] = dde.metrics.l2_relative_error(sample_y, y_predict)
                print(
                    f"layers =  {layers[i]},  neurons = {neurons[j]}, initializer = {initializer[k]}")

            means[i, j] = np.mean(L_i)
            standard_var[i, j] = np.std(L_i)
            content = [str(layers[i]), ",", str(neurons[j]), ",", str(
                means[i, j]), ",", str(standard_var[i, j]), "\n"]
            file.writelines(content)
file.close()
