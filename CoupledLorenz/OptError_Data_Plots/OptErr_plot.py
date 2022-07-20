#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

nObs = '25'
nDom = '100'
learn_rate1 = '0.0001'
learn_rate2 = '1e-05'

openPath_Data1 = "OptError_N"+nDom+"_"+nObs+"_"+learn_rate1+".txt"
openPath_Data2 = "OptError_N"+nDom+"_"+nObs+"_"+learn_rate2+".txt"
savePath_Plot = "OptError_N"+nDom+"_"+nObs+".png"

file1 = np.loadtxt(openPath_Data1)
file2 = np.loadtxt(openPath_Data2)

fig = plt.figure()
ax = fig.add_subplot(111)
#fig1.suptitle('Mean', fontsize=12)
ax.set_title('Mean', fontsize=12)
ax.set_xscale('log')
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('L2 norm of error', fontsize=12)
ax.plot(file1[:, 0], file1[:, 1], label=(f"learning rate: {learn_rate1}"))
ax.plot(file2[:, 0], file2[:, 1], label=(f"learning rate: {learn_rate2}"))
ax.fill_between(file1[:, 0], file1[:, 1] -
                file1[:, 2], file1[:, 1] + file1[:, 2], alpha=0.2)
ax.fill_between(file2[:, 0], file2[:, 1] -
                file2[:, 2], file2[:, 1] + file2[:, 2], alpha=0.2)
fig.legend(loc="upper left")
plt.show()

fig.savefig(savePath_Plot)
