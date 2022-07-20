import deepxde as dde
import numpy as np

a = dde.Variable(1.0)
b = dde.Variable(1.0)
c = dde.Variable(1.0)

geom = dde.geometry.TimeDomain(0, 1)

def Lorenz_system(t, w):
    x, y, z = w[:, 0:1], w[:, 1:2], w[:, 2:]
    dx_t = dde.grad.jacobian(w, t, i=0)
    dy_t = dde.grad.jacobian(w, t, i=1)
    dz_t = dde.grad.jacobian(w, t, i=2)
    return [
        dx_t - a * (y - x),
        dy_t - x * (b - z) - y, 
        dz_t - x * y - c * z,
    ]

ic1 = dde.icbc.IC(geom, lambda X: 1, lambda _, on_initial: on_initial, component=0)
ic2 = dde.icbc.IC(geom, lambda X: 1, lambda _, on_initial: on_initial, component=1)
ic3 = dde.icbc.IC(geom, lambda X: 1, lambda _, on_initial: on_initial, component=2)

def gen_traindata():
    data = np.load("/home/Alejandro/Gitlab/pinns/Lorenz_EQSyst/data.npz")
    return data["t"], data["w"]

observe_t, ob_w = gen_traindata()
observe_x = dde.icbc.PointSetBC(observe_t, ob_w[:, 0:1], component=0)
observe_y = dde.icbc.PointSetBC(observe_t, ob_w[:, 1:2], component=1)
observe_z = dde.icbc.PointSetBC(observe_t, ob_w[:, 2:3], component=2)

data = dde.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_x, observe_y, observe_z],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)

net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[a, b, c])
variable = dde.callbacks.VariableValue(
  [a, b, c], period=600, filename="variables.dat"
)

losshistory, train_state = model.train(epochs=60000, callbacks=[variable])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
