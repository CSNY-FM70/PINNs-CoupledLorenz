import numpy as np
import deepxde as dde
import Coupled_Lorenz_Inverse as CLI


initializer = ["Glorot uniform", "Glorot normal",
               "He uniform", "He normal", "LeCun uniform", "LeCun normal"]

epochs = 5e4
learning_rate = 1e-3

init = initializer[0]
SaveInit = '_'+init[0]+'_'+init[-1]+'_'
Data_Iteration = [10,20,50,100,200,500]
#Data_Iteration = [1000]
#Data_Iteration = [5000]
savePath_Data = 'GenError_Data_Plots/5e4_GlorotU_'+str(Data_Iteration[-1])+'.txt'

########## Constants ##########
# t0, T, DriverPar, RespPar, CoupPar, InitV0
Physics = [0.0, 3.0, [10., 28., (8./3.)], [10., 12., (8./3.)],
           [0.1, 0.1, 0.1], [13.561, 10.162, 36.951, 3.948, 4.187, 11.531]]
# nDom, nBound, nObs, nTest
GenErr = [None, 2, None, 400]
# TrainingTime(epochs), LearnRate, Residual Loss, Initial Value Loss, Data Loss
OptErr = [int(epochs), learning_rate, (1e0*np.full(6, 1)).tolist(), 
          (1e0*np.full(6, 1)).tolist(), (1e2*np.full(6, 1)).tolist()]
# NNarcht, ActFunc, InitWeights
AppErr = [[1] + [100]*4 + [6], 'swish', init]
###############################
GenErr[0] = 100



TrainData_Path = 'data_T10_1e5.npz' 
data = np.load(TrainData_Path)
time, states   \
    = data['t'], data['w']
time = time[:,np.newaxis]


data_span = 0.3
t_max = time.shape[0]*data_span
padFactor = int(t_max/3e4)
sample_t = time[1:int(t_max)-1:padFactor].copy()
sample_t[-1] = time[int(t_max)].copy()
sample_y = states[1:int(t_max)-1:padFactor].copy()
sample_y[-1] = states[int(t_max)].copy()

# Simulation loop
with open(savePath_Data, "w") as file:
    for n in Data_Iteration:
        GenErr[2] = n
        model, _, _ = CLI.main(
            Physics, AppErr, GenErr, OptErr)
        y_predict = model.predict(sample_t)
        print("L2 Rel Error: ", dde.metrics.l2_relative_error(
            sample_y, y_predict))
        L = dde.metrics.l2_relative_error(sample_y, y_predict)
        content = [str(GenErr[0]), " ", str(GenErr[2])," ", str(L), "\n"]
        file.writelines(content)

file.close()
