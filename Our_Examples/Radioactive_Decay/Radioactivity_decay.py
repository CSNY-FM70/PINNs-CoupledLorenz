#!/usr/bin/env python3

import deepxde as dde
import numpy as np

#Definition of problem constants - STRONTIUM 90 
N_0 = 1             #6.70e17	    #Initial number of atoms
dpy = 0.693/28.8    #7.61e-10		#decay rate [1/year] 
t_end = 28.8     	#28.8 years     #half life [year]

#only time dependent, no space domain
geom = dde.geometry.TimeDomain(0,t_end)

#ODE => dN/dt = -lambda * N(t)
def ode(t, N):
    dN_t = dde.grad.jacobian(N, t)
    return (dN_t + (dpy * N))

#Analytical solution => N(t) = N_0 * exp(-lambda*t)
def func(t):
    return N_0 * np.exp(-dpy*t)
    
#Initial condition N(0)=N_0
ic = dde.icbc.IC(geom, lambda x:N_0, lambda _, on_initial: on_initial)

#Problem Setup
data = dde.data.PDE(geom, ode, ic, num_domain=30, num_boundary=2, solution=func, num_test=40)

#Neuronal Network Setup
net = dde.nn.FNN([1] + [50]*1 + [1], "swish", "Glorot uniform")

#Union of Problem and Neuronal Network
model = dde.Model(data, net)

#Defifnition of training parameters/algorithm
model.compile("adam", lr=0.001)
#model.compile("L-BFGS")

losshistory, train_state = model.train(epochs=16000)
#losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

