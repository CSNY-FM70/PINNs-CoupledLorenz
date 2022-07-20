import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt


t0 = 0.0
T = 2.

nDom=50
nBound=16
nInit = 25
nTest=1000
nObs = 100

NNarcht = [3] + [20]*2 + [1]
ActFunc = "swish"
InitWeights = "Glorot normal"
TrainingTime = int(6e4)

C1 = dde.Variable(1.0)
C2 = dde.Variable(1.0)
alpha = 3.0
beta = 1.2

#Model PDE
def pde(x, u):
    du_xx = dde.grad.hessian(u, x, j=0)
    du_yy = dde.grad.hessian(u, x, j=1)
    du_t = dde.grad.jacobian(u, x, j=2)
    return (
        du_t
        - du_xx
        - du_yy
        -1.0*(C2 - 2.0 - 2.0*C1)
        )

#Boundary segmentation - flexibility for future examples
def r_boundary(X,on_boundary):
    x,y,t = X
    return on_boundary and np.isclose(x,1)
def l_boundary(X,on_boundary):
    x,y,t = X
    return on_boundary and np.isclose(x,0)
def up_boundary(X,on_boundary):
    x,y,t = X
    return on_boundary and np.isclose(y,1)
def down_boundary(X,on_boundary):
    x,y,t = X
    return on_boundary and np.isclose(y,0)
def boundary_initial(_, on_initial):
    x,y,t = X
    return on_initial and np.isclose(t, 0)

def init_func(X):
    x,y,t = X[:, 0:1], X[:, 1:2],X[:,2:3]
    return  1.0 + x**2 + alpha*(y**2)

#Chosen manufactured solution
def func(X):
    x,y,t = X[:, 0:1], X[:,1:2],X[:,2:3]
    return 1.0 + x**2 + alpha*(y**2) + beta*t

geom = dde.geometry.Rectangle(xmin=[0,0],xmax=[1,1])
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#Define boundary conditions with integration domain, function and location
bc1 = dde.DirichletBC(geomtime, func, l_boundary)
bc2 = dde.DirichletBC(geomtime, func, r_boundary)
bc3 = dde.DirichletBC(geomtime, func, up_boundary)
bc4 = dde.DirichletBC(geomtime, func, down_boundary)

#Genral Bounday/Initial Condition SetUp
ic = dde.icbc.IC(geomtime, init_func, lambda _, on_initial: on_initial)
bc = dde.icbc.DirichletBC(geomtime,func, lambda _, on_boundary: on_boundary)

x_obs = geomtime.random_points(nObs)

y_obs = func(x_obs)
y_train = dde.icbc.PointSetBC(x_obs, y_obs, component=0)


#Data Terms (Physical and/or Empirical) to be minimized
data1 = dde.data.TimePDE(
    geomtime,
    pde,
    [bc1, bc2, bc3, bc4, ic, y_train],
    num_domain=400,
    num_boundary=100,
    num_initial=200,
    anchors=x_obs,
    solution= func,
    num_test=500
)

data2 = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic, y_train],
    num_domain=nDom,
    num_boundary=nBound,
    num_initial=nInit,
    anchors=x_obs,
    solution= func,
    num_test=nTest
)

#Network Architecture parameteres
net = dde.nn.FNN(NNarcht, ActFunc, InitWeights)
#net.apply_output_transform(lambda x,y: abs(y))
model = dde.Model(data2, net)
variable = dde.callbacks.VariableValue(
    [C1, C2], period=1000, filename='variables.dat'
    )

#Training hyperparameters
# [PDE Loss, BC1 loss, BC2 loss, BC3 loss, BC4 loss, IC Loss, Data Loss]
lossWeights = [1,1,1,1]
#loss_weights = [10, 1, 1, 1, 1, 1, 10]
checker = dde.callbacks.ModelCheckpoint("Checkpoint_Model/Inverse_model_Adam", save_better_only=True, period=1000)

Training_Flag = str(input('Load or Train Model? [load | train] : '))
try:
    assert(Training_Flag == 'train' or Training_Flag == 'load')
    if Training_Flag == 'train':
        #model.compile("adam", lr= 1e-3, external_trainable_variables = [C1, C2], loss_weights=lossWeights)
        #losshistory, train_state = model.train(epochs=TrainingTime, callbacks = [checker,variable])
            
        model.compile("L-BFGS", dde.optimizers.set_LBFGS_options(maxcor=100),external_trainable_variables = [C1, C2], loss_weights=lossWeights)
        losshistory, train_state = model.train(epochs = TrainingTime*0.5, callbacks = [variable], model_save_path='Checkpoint_Model/Inverse_model_BFGS')
    else: 
        optim = str(input('Desired Optimizer: '))
        cpt = int(input('Desired Checkpoint : '))
        if optim == 'Adam':
           model.compile("adam", lr= 1e-3, external_trainable_variables = [C1, C2], loss_weights=lossWeights)
           model.restore('Checkpoint_Model/Inverse_model_Adam-'+str(cpt)+'.ckpt', verbose=1)
        else:
            model.compile("L-BFGS", dde.optimizers.set_LBFGS_options(maxcor=50),external_trainable_variables = [C1, C2], loss_weights=lossWeights)
            model.restore('Checkpoint_Model/Inverse_model_BFGS-'+str(cpt)+'.ckpt', verbose=1)

except AssertionError as AE:
        print(AE)
        print("Loading last stable model")
        model.restore('Checkpoint_Model/Inverse_model_Adam-30000.ckpt', verbose=1)


Plot_Flag = True
while(Plot_Flag):

    nelx = 100
    nely = 100
    timesteps = 101
    x = np.linspace(0,1,nelx+1)
    y = np.linspace(0,1,nely+1)
    t = np.linspace(0,T,timesteps)
    delta_t = t[1] - t[0]
    xx,yy = np.meshgrid(x,y)


    x_ = np.zeros(shape = ((nelx+1) * (nely+1),))
    y_ = np.zeros(shape = ((nelx+1) * (nely+1),))
    for c1,ycor in enumerate(y):
        for c2,xcor in enumerate(x):
            x_[c1*(nelx+1) + c2] = xcor
            y_[c1*(nelx+1) + c2] = ycor

    user_Time = float(input(f'Desired Time[s]: '))
    time = user_Time
    t_ = np.ones((nelx+1) * (nely+1),) * (time)
    X = np.column_stack((x_,y_))
    X = np.column_stack((X,t_))
    T_Pred = model.predict(X)
    T_Pred = T_Pred.reshape(T_Pred.shape[0],)
    T_Pred = T_Pred.reshape(nelx+1,nely+1)

    #X = np.column_stack((x_,y_))
    #X = np.column_stack((X,t_))
    T_Sol = func(X)
    T_Sol = T_Sol.reshape(T_Sol.shape[0],)
    T_Sol = T_Sol.reshape(nelx+1,nely+1)

    Error_T = np.sqrt((T_Pred - T_Sol)**2)
    T_Sol = T_Sol/np.max(T_Sol)
    T_Pred = T_Pred/np.max(T_Pred)

    fig = plt.figure(figsize=(12,6))
    fig.suptitle('Time '+str(round(time,2))+'[s]', fontsize=14)
    fig = plt.subplot(131)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolor(xx, yy, T_Sol,cmap = 'seismic', vmin=0.)
    plt.colorbar()

    fig = plt.subplot(132)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolor(xx, yy, T_Pred,cmap = 'seismic', vmin=0.)
    plt.colorbar()

    fig = plt.subplot(133)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pcolor(xx, yy, Error_T,cmap = 'Blues')
    plt.colorbar()

    plt.savefig('Plots/Inverse.png')
    plt.show()
    