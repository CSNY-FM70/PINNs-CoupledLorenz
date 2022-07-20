import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create an image of the Lorenz attractor.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-lorenz-attractor/
# Christian Hill, January 2016.
# Updated, January 2021 to use scipy.integrate.solve_ivp.

user = str(input('Solve coupled system?[y|n]: '))
if user == 'y':
    Coupled_Flag = True
else:
    Coupled_Flag = False
WIDTH, HEIGHT, DPI = 1000, 750, 100
T = 30.
t0 = 0.0
# Lorenz paramters and initial conditions.
state0 = [-8., 7.0, 27.0]
state1 = [13.561, 10.162, 36.951, 3.948, 4.187, 11.531]
sigma, rho1, beta, eps, rho2 = 10., 28., (8./3.), 1.05, 12

time = np.arange(t0, T, 1e-4)

def lorenz(t, X, sigma, rho, beta):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

def coupled_lorenz(t, X, sigma, rho1, beta, eps, rho2):
    """Couple Lorenz Attractor"""
    x1, y1, z1, x2, y2, z2 = X

    x1_dot = sigma*(y1 - x1)
    y1_dot = rho1*x1 - y1 - x1*z1
    z1_dot = -beta*z1 + x1*y1
    
    x2_dot = sigma*(y2 - x2) + 0.1*(7.3*x1 + np.cos(x1))
    y2_dot = rho2*x2 - y2 -x2*z2 + 0.1*(1.2*y1 + 0.5*np.arctan(y1))
    z2_dot = -beta*z2 + x2*y2 + 0.1*(3.5*z1 + 0.9*np.exp(z1))

    return x1_dot,y1_dot,z1_dot,x2_dot,y2_dot,z2_dot


if Coupled_Flag:
    soln = solve_ivp(coupled_lorenz, (t0, T), state1, args=(sigma, rho1, beta, eps, rho2),dense_output=True)
    x1, y1, z1, x2, y2, z2 = soln.sol(time)
    observed_y = np.array([x1,y1,z1,x2,y2,z2])

else:
    states = odeint(lorenz, state0, time,args=(sigma, rho1, beta),tfirst=True)
    x1,y1,z1 = states[:,0], states[:,1], states[:,2]
    observed_y = states.copy()

sections = 100
padFactor = int(time.shape[0]/sections)
sample_t = time[1:-1:padFactor].copy()
sample_t[-1] = time[-1].copy()
sample_y = observed_y[1:-1:padFactor].copy()
sample_y[-1] = observed_y[-1].copy()


# Interpolate solution onto the time grid, t.

# Plot the Lorenz attractor using a Matplotlib 3D projection.
fig = plt.figure()
n = len(time)
s = int(n/20.0)     #3 segmented in 12 is [0., 0.25,....]
cmap = plt.cm.turbo
if Coupled_Flag:
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,2,2,projection='3d')
else:
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    

for i in range(0,n-s,s):
    ax1.plot(x1[i:i+s+1], y1[i:i+s+1], z1[i:i+s+1], color='b', alpha=0.8)
    if Coupled_Flag:
        ax2.plot(x2[i:i+s+1], y2[i:i+s+1], z2[i:i+s+1], color='r', alpha=0.8)

i+=s
ax1.plot(x1[i:i+s+1], y1[i:i+s+1], z1[i:i+s+1], color='b', alpha=0.8)
if Coupled_Flag:
    ax2.plot(x2[i:i+s+1], y2[i:i+s+1], z2[i:i+s+1], color='r', alpha=0.8)
#ax.plot(sample_y[:,0], sample_y[:,1], sample_y[:,2], 'k.')

plt.show()
