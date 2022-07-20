import numpy as np
import matplotlib.pyplot as plt

epochs = '5e4'
ptype = 'Res'
Init = ['GlorotN', 'GlorotU', 'HeN', 'HeU', 'LecunN', 'LecunU']

nobs = []
nDom = []
try:
    test = np.loadtxt(str(epochs)+'_'+Init[int(np.random.uniform(high=5.0))]+'.txt')
    nObs = test[:,1].copy()

    test = np.loadtxt(ptype+'_'+str(epochs)+'_'+Init[int(np.random.uniform(high=5.0))]+'.txt')
    nDom = test[:,0].copy()
except KeyboardInterrupt:
    print("Computation aborted by user.")
except AssertionError as AE:
    print("Assertion:")
    print(AE)
except Warning as W:
    print("Warning:")
    print(W)
except BaseException as E:
    print("Exception:")
    print(E)

L2 = np.zeros((len(Init), len(nDom)))
for i,init in enumerate(Init):
    openPath = str(epochs)+'_'+init+'.txt'
    file_5e4 = np.loadtxt(openPath, delimiter=' ')
    L2[i,:] = file_5e4[:,2]

mean_L2 = np.mean(L2, axis=0)
std_L2 = np.std(L2, axis=0) 

L2_Res = np.zeros((len(Init), len(nDom)))
for i,init in enumerate(Init):
    openPath = ptype+'_'+str(epochs)+'_'+init+'.txt'
    file_5e4R = np.loadtxt(openPath, delimiter=' ')
    L2_Res[i,:] = file_5e4R[:,2]

mean_L2R = np.mean(L2_Res, axis=0)
std_L2R = np.std(L2_Res, axis=0)

L2_Res5e5 = np.zeros((len(Init), len(nDom)))
for i,init in enumerate(Init):
    openPath = ptype+'_5e5_'+init+'.txt'
    file_5e5R = np.loadtxt(openPath, delimiter=' ')
    L2_Res5e5[i,:] = file_5e5R[:,2]

mean_L2R5e5 = np.mean(L2_Res5e5, axis=0)
std_L2R5e5 = np.std(L2_Res5e5, axis=0)

fig = plt.figure(figsize=(18,9))
ax1 = fig.add_subplot(121)
ax1.set_xscale('log')
ax1.set_xlabel(r'$n_{res},n_{data}$')
ax1.set_ylabel(r'$\mathbb{E}[L^2_{rel}]$')
ax1.plot(nDom[:-1], mean_L2[:-1], '-ob', label='Data Points')
ax1.fill_between(nDom[:-1], mean_L2[:-1] - std_L2[:-1], mean_L2[:-1] + std_L2[:-1], alpha=0.2)
ax1.plot(nDom[:-1], mean_L2R[:-1], '-o',color='orange', label='Residual Points')
ax1.fill_between(nDom[:-1], mean_L2R[:-1] - std_L2R[:-1], mean_L2R[:-1] + std_L2R[:-1], alpha=0.2)
ax1.legend(loc="upper right", borderaxespad=0.1)
        

ax2 = fig.add_subplot(122)
ax2.set_xscale('log')
ax2.set_xlabel(r'$n_{res}$')
ax2.set_ylabel(r'$\mathbb{E}[L^2_{rel}]$')
ax2.plot(nDom, mean_L2R5e5, '-ob', label='epochs = 5e5, lr= 1e-4')
ax2.fill_between(nDom, mean_L2R5e5 - std_L2R5e5, mean_L2R5e5 + std_L2R5e5, alpha=0.2)
ax2.plot(nDom, mean_L2R, '-o', color='orange', label='epochs = 5e4, lr= 1e-3')
ax2.fill_between(nDom, mean_L2R - std_L2R, mean_L2R + std_L2R, alpha=0.2)
ax2.legend(loc="upper right", borderaxespad=0.1)

fig.tight_layout()
fig.savefig('GenError.png')
plt.show()
