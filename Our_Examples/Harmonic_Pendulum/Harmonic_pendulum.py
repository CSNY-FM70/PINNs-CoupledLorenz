#!/usr/bin/env python3

import deepxde as dde
import numpy as np

#Definition of problem constants - pendulum
phi_0 = 2	# This number has to be less than 2 degree [deg]
g = 9.81	# Gravity [m/s^2]
L = 2		# Length of cord attached to mass [m]
t_end = 3	# Time [s]

#only time dependent, no space domain
geom = dde.geometry.TimeDomain(0,t_end)

#ODE => d2_phi/dt^2 = - (g/L) * phi
def ode(t, phi):
    d_phi_t = dde.grad.hessian(phi, t)
    return (d_phi_t + (g/L) * phi)

#Analytical solution => phi(t) = phi_0 * cos(sqrt(g/L)*t)
def func(t):
    return phi_0 * np.cos(np.sqrt(g/L)*t)
    
#Initial condition phi(0)=phi_0
ic = dde.icbc.IC(geom, lambda x:phi_0, lambda _, on_initial: on_initial)

#Problem Setup
data = dde.data.PDE(geom, ode, ic, num_domain=100, num_boundary=1, solution=func, num_test=20)

#Neuronal Network Setup
net = dde.nn.FNN([1] + [50]*3 + [1], "swish", "Glorot uniform")

#Union of Problem and Neuronal Network
model = dde.Model(data, net)

#Defifnition of training parameters/algorithm
model.compile("adam", lr=0.001)

losshistory, train_state = model.train(epochs=20000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)

