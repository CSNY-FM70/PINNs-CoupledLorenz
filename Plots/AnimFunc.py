import deepxde as dde
import numpy as np
from deepxde.backend import tf

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from Our_Examples import Heat

model = Heat.model
model.restore('Models/Heat_2D/model_heat_2D.ckpt-10000.ckpt', verbose=1)
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot()
nelx = 100
nely = 100
timesteps = 101
x = np.linspace(0,1,nelx+1)
y = np.linspace(0,1,nely+1)
t = np.linspace(0,1,timesteps)
delta_t = t[1] - t[0]
xx,yy = np.meshgrid(x,y)


x_ = np.zeros(shape = ((nelx+1) * (nely+1),))
y_ = np.zeros(shape = ((nelx+1) * (nely+1),))
for c1,ycor in enumerate(y):
    for c2,xcor in enumerate(x):
        x_[c1*(nelx+1) + c2] = xcor
        y_[c1*(nelx+1) + c2] = ycor
Ts_Predicted = []
Ts_Solution = []
        
for time in t:
    t_ = np.ones((nelx+1) * (nely+1),) * (time)
    X = np.column_stack((x_,y_))
    X = np.column_stack((X,t_))
    T = model.predict(X)
    #T = T*30
    T = T.reshape(T.shape[0],)
    T = T.reshape(nelx+1,nely+1)
    Ts_Predicted.append(T)

    X = np.column_stack((x_,y_))
    X = np.column_stack((X,t_))
    T_Sol = Heat.func(X)
    T_Sol = T_Sol.reshape(T_Sol.shape[0],)
    T_Sol = T_Sol.reshape(nelx+1,nely+1)
    Ts_Solution.append(T_Sol)

def plotheatmap(T,time):
  # Clear the current plot figure
    plt.clf()
    plt.title(f"Temperature at t = {time*delta_t} unit time")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolor(xx, yy, T,cmap = 'RdBu_r', vmin=0.01, vmax=6)
    plt.colorbar()
    return plt

def animate_Predicted(k):
    plotheatmap(Ts_Predicted[k], k)
    
anim_Predicted = animation.FuncAnimation(fig, animate_Predicted, interval=1, frames=len(t), repeat=False)
    
anim_Predicted.save('Plots/Heat_2D/Predicted.gif')

def animate_Solution(k):
    plotheatmap(Ts_Solution[k],k)

anim_Solution = animation.FuncAnimation(fig, animate_Solution, interval=1, frames=len(t), repeat = False)
#anim_Timed = animation.TimedAnimation(fig,,interval=1)
anim_Solution.save('Plots/Heat_2D/Sol.gif')
