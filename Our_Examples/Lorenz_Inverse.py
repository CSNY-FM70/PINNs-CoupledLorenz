"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

t0 = 0.0
T = 3.0

nDom=400
nBound=2
nTest=400
nObs = 25

NNarcht = [1] + [75]*3 + [3]
ActFunc = "swish"
InitWeights = "Glorot normal"
TrainingTime = 60000

#[RD1, RD2, RD3, IC1, IC2, IC3, OB1, OB2, OB3]
loss_weights = [1,1,1,1,1,1,1e2,1e2,1e2]
loss_weights_nonU = [1,1,1,1,1,1,1,1,1] 

zeroState = [-8.0, 7., 27.]
Parameters = [10., 28., 8./3.]

def gen_traindata(num_points, zeroState, Parameters, T, t_start=0.0, dt=1e-4):
    def lorenz(x,Y, s, r, b):
        """
        Given:
           x, y, z: a point of interest in three dimensional space
           s, r, b: parameters defining the lorenz attractor
        Returns:
           x_dot, y_dot, z_dot: values of the lorenz attractor's partial
               derivatives at the point x, y, z
        """
        y1,y2,y3 = Y
        dy1_x = s*(y2 - y1)
        dy2_x = r*y1 - y2 - y1*y3
        dy3_x = y1*y2 - b*y3
        return dy1_x, dy2_x, dy3_x

    sigma, rho, beta = Parameters

    if num_points>=1e4:
        dt = (T-t_start)/num_points
    
    try:
        observed_t = np.arange(t_start,T+dt,dt)[:,np.newaxis]
    
        observed_y = odeint(lorenz, zeroState, observed_t[:,0], args=(sigma,rho,beta), tfirst=True)

        assert observed_t.shape[0] == observed_y.shape[0] , \
                "Padding Needed"

    except AssertionError as AE:
        print(AE)
        print("Padding Done")
        observed_t =np.arange(t0,T+dt,dt)[:-1][:,np.newaxis]
    
    padFactor = int(observed_t.shape[0]/num_points)
    sample_t = observed_t[1:-1:padFactor].copy()
    sample_t[-1] = observed_t[-1].copy()
    sample_y = observed_y[1:-1:padFactor].copy()
    sample_y[-1] = observed_y[-1].copy()
    
    return  observed_t, observed_y, sample_t, sample_y


C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)


def Lorenz_system(x, y):
    """Lorenz system.
    dy1/dx = 10 * (y2 - y1)
    dy2/dx = y1 * (15 - y3) - y2
    dy3/dx = y1 * y2 - 8/3 * y3
    """
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    return [
        dy1_x - C1 * (y2 - y1),
        dy2_x - y1 * (C2 - y3) + y2,
        dy3_x - y1 * y2 + C3 * y3,
    ]


def boundary(_, on_initial):
    return on_initial


geom = dde.geometry.TimeDomain(t0, T)

def data_cluster(num_points,ref_t, ref_y):
    
    n = len(ref_t)
    idx = np.append(
        np.random.choice(np.arange(1, n - 1), size=num_points -2, replace=False), [0, n - 1]
    )
    sampl_t = ref_t[idx].copy()
    sampl_y = ref_y[idx].copy()    
    return sampl_t, sampl_y

    
# Initial conditions
ic1 = dde.icbc.IC(geom, lambda X: zeroState[0], boundary, component=0)
ic2 = dde.icbc.IC(geom, lambda X: zeroState[1], boundary, component=1)
ic3 = dde.icbc.IC(geom, lambda X: zeroState[2], boundary, component=2)

# Get the train data
sampled_t, sampled_y, observe_t, ob_y = gen_traindata(nObs, zeroState, Parameters, T)
observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

data = dde.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=nDom,
    num_boundary=nBound,
    anchors=observe_t,
    num_test=nTest
)

net = dde.nn.FNN(NNarcht, ActFunc, InitWeights)
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue(
    [C1, C2, C3], period=1000, filename="variables.dat"
)

#model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2, C3],loss_weights = loss_weights)
#losshistory, train_state = model.train(epochs=TrainingTime, callbacks=[variable])

model.compile("L-BFGS", dde.optimizers.set_LBFGS_options(maxcor=50), external_trainable_variables=[C1, C2, C3],loss_weights = loss_weights)
losshistory, train_state = model.train(epochs = TrainingTime*0.5, callbacks=[variable])    #,model_save_path='Models/Lorenz/Inverese_Simple_BFGS')

step = 1e-4
highresTIME =  sampled_t.copy() #np.arange(t0,2.*T+step,step)[:,np.newaxis]
uniform_y = model.predict(highresTIME)
 

#Clustered DATA
C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
C3 = dde.Variable(1.0)

random_t,y=  data_cluster(nObs, sampled_t, sampled_y)

observe_y0 = dde.icbc.PointSetBC(random_t, y[:, 0:1], component=0)
observe_y1 = dde.icbc.PointSetBC(random_t, y[:, 1:2], component=1)
observe_y2 = dde.icbc.PointSetBC(random_t, y[:, 2:3], component=2)

data = dde.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=nDom,
    num_boundary=nBound,
    anchors=random_t,
    num_test=nTest
)

net = dde.nn.FNN(NNarcht, ActFunc, InitWeights)
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue(
    [C1, C2, C3], period=1000, filename="variables.dat"
)

model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2, C3],loss_weights = loss_weights_nonU)
losshistory, train_state = model.train(epochs=TrainingTime, callbacks=[variable])

model.compile("L-BFGS", dde.optimizers.set_LBFGS_options(maxcor=50), external_trainable_variables=[C1, C2, C3],loss_weights = loss_weights_nonU)
losshistory, train_state = model.train(epochs = TrainingTime*0.5, callbacks=[variable])    #,model_save_path='Models/Lorenz/Inverese_Simple_BFGS')

clustered_y = model.predict(sampled_t)


# Plot
fig1 = plt.figure()
fig1.suptitle(str(round(Parameters,2))+'Lorenz Attractor - Sol.', fontsize=18)

ax1 = fig1.add_subplot(1,2,1,projection='3d')
ax1.plot(sampled_y[:,0],sampled_y[:,1], sampled_y[:,2], 'k-', label = "Data Trace", lw=0.5)
ax1.plot(ob_y[:,0], ob_y[:,1], ob_y[:,2], 'k.', label = "Obs. Data #"+str(observe_t.shape[0]), lw=0.1)
ax1.plot(y[:,0], y[:,1], y[:,2], 'b.', label = "Obs. Data #"+str(random_t.shape[0]), lw=0.1)
#ax1.plot(y[:,0], y[:,1] , y[:,2], 'b.', label=' #'+str(t.shape[0]))
ax1.set_xlabel("X Axis",fontsize=5)
ax1.set_ylabel("Y Axis",fontsize=5)
ax1.set_zlabel("Z Axis",fontsize=5)
ax1.set_title("Data Quality", fontsize=10)

ax2 = fig1.add_subplot(1,2,2,projection='3d')
ax2.plot(uniform_y[:,0], uniform_y[:,1], uniform_y[:,2], 'r', label = 'Model 1', lw=0.5)
ax2.plot(clustered_y[:,0], clustered_y[:,1], clustered_y[:,2], 'b--', label = 'Model 2', lw=0.5)
ax2.set_xlabel("X Axis",fontsize=5)
ax2.set_ylabel("Y Axis",fontsize=5)
ax2.set_zlabel("Z Axis",fontsize=5)
ax2.set_title("ANN-Solution", fontsize=10)

fig2=plt.figure()
ax3 = fig2.add_subplot(3,1,1)
ax3.axis([sampled_t[0],sampled_t[-1],_,_])
ax3.plot(sampled_t, sampled_y[:,0],'k',label = "Data Trace")
ax3.plot(sampled_t, uniform_y[:,0],'r--', label = 'Model 1')
ax3.plot(sampled_t, clustered_y[:,0],'b--', label = 'Model 2')
ax3.set_title("X - Convective Intensity", fontsize=10)

ax4 = fig2.add_subplot(3,1,2)
ax4.axis([sampled_t[0],sampled_t[-1],_,_])
ax4.plot(sampled_t, sampled_y[:,1],'k')
ax4.plot(sampled_t, uniform_y[:,1],'r--')
ax4.plot(sampled_t, clustered_y[:,1],'b--')
ax4.set_title("Y - Horizontal Temperature Variation", fontsize=10)

ax5 = fig2.add_subplot(3,1,3)
ax5.axis([sampled_t[0],sampled_t[-1],_,_])
ax5.plot(sampled_t, sampled_y[:,2],'k')
ax5.plot(sampled_t, uniform_y[:,2],'r--')
ax5.plot(sampled_t, clustered_y[:,2],'b--')
ax5.set_title("Z - Vertical Temperature Variation", fontsize=10)


fig1.tight_layout()
fig1.legend(loc="lower left", borderaxespad=0.1)
fig2.legend(loc="lower left", borderaxespad=0.1)

plt.show()