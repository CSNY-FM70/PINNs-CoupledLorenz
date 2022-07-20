import matlab.engine
import matplotlib.pyplot as plt


def gen_traindata(num_points, Parameters, initValue, tZero=0.0, tEnd=3.0, eps=1e-4,
                  SolFlag=False, PlotFlag=False):
    """ Set SolFlag to true if matlab API for python enabled
    """
    eng = matlab.engine.start_matlab()
            
    driver = matlab.double(Parameters[0])
    response = matlab.double(Parameters[1])
    coupling = matlab.double(Parameters[2])

    initV = matlab.double(initValue)
    t_span = matlab.double(np.arange(tZero, tEnd, eps).tolist())

    time, states = eng.LorenzSolver(
        driver, response, coupling, initV, t_span, eps, nargout=2)

    time = np.array(time)
    states = np.array(states)

    padFactor = int(time.shape[0]/num_points)
    sample_t = time[1:-1:padFactor].copy()
    sample_t[-1] = time[-1].copy()
    sample_y = states[1:-1:padFactor].copy()
    sample_y[-1] = states[-1].copy()

    np.savez('./data', t=time, w=states, ts=sample_t, ws=sample_y)

    data = np.load('data.npz')
    time, states, sample_t, sample_y    \
        = data['t'], data['w'], data['ts'], data['ws']

    padFactor = int(time.shape[0]/num_points)
    sample_t = time[1:-1:padFactor].copy()
    sample_t[-1] = time[-1].copy()
    sample_y = states[1:-1:padFactor].copy()
    sample_y[-1] = states[-1].copy()

    if PlotFlag:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(states[:, 0], states[:, 1], states[:, 2], 'b-', lw=0.5)
        #ax1.plot(sample_y[:,0], sample_y[:,1], sample_y[:,2], 'k.', lw=0.1)
        ax1.set_xlabel("X Axis", fontsize=10)
        ax1.set_ylabel("Y Axis", fontsize=10)
        ax1.set_zlabel("Z Axis", fontsize=10)
        ax1.set_title("Driver System", fontsize=18)

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot(states[:, 3], states[:, 4], states[:, 5], 'r-', lw=0.5)
        #ax2.plot(sample_y[:,3], sample_y[:,4], sample_y[:,5], 'k.', lw=0.1)
        ax2.set_xlabel("X Axis", fontsize=10)
        ax2.set_ylabel("Y Axis", fontsize=10)
        ax2.set_zlabel("Z Axis", fontsize=10)
        ax2.set_title("Response System", fontsize=18)

        fig2 = plt.figure()
        ax3 = fig2.add_subplot(3, 2, 1)
        ax3.plot(time, states[:, 0], 'b')
        ax3.set_xlabel("Time", fontsize=10)
        ax3.set_ylabel("X-Coord", fontsize=10)

        ax4 = fig2.add_subplot(3, 2, 3)
        ax4.plot(time, states[:, 1], 'b')
        ax4.set_xlabel("Time", fontsize=10)
        ax4.set_ylabel("Y-Coord", fontsize=10)

        ax5 = fig2.add_subplot(3, 2, 5)
        ax5.plot(time, states[:, 2], 'b')
        ax5.set_xlabel("Time", fontsize=10)
        ax5.set_ylabel("Z-Coord", fontsize=10)

        ax6 = fig2.add_subplot(3, 2, 2)
        ax6.plot(time, states[:, 3], 'r')
        ax6.set_xlabel("Time", fontsize=10)
        ax6.set_ylabel("X-Coord", fontsize=10)

        ax7 = fig2.add_subplot(3, 2, 4)
        ax7.plot(time, states[:, 4], 'r')
        ax7.set_xlabel("Time", fontsize=10)
        ax7.set_ylabel("Y-Coord", fontsize=10)

        ax8 = fig2.add_subplot(3, 2, 6)
        ax8.plot(time, states[:, 5], 'r')
        ax8.set_xlabel("Time", fontsize=10)
        ax8.set_ylabel("Z-Coord", fontsize=10)

        plt.show()

    # Modify coupling Eqs.
    #eng.edit('LorenzSolver', nargout=0)

    return time, states, sample_t, sample_y

if __name__ =='__main__':
    Physics = [0.0, 3.0, [10., 28., (8./3.)], [10., 12., (8./3.)],
               [0.1, 0.1, 0.1], [13.561, 10.162, 36.951, 3.948, 4.187, 11.531]]

    t0, T, DriverPar, RespPar, CoupPar, InitV = Physics
    Param = [DriverPar, RespPar, CoupPar]

    _,_,_,_ = gen_traindata(int(1e5),Param,InitV,tEnd=10.)