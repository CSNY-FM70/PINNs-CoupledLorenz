import deepxde as dde
import numpy as np
from deepxde.backend import tf

#import matlab.engine
import matplotlib.pyplot as plt


def gen_traindata(num_points, Parameters, initValue, tZero=0.0, tEnd=3.0, eps=1e-4,
                  SolFlag=False, PlotFlag=False, DataPath='./data.npz', Forecast = False):
    """ Set SolFlag to true if matlab API for python enabled
    """
    if SolFlag:
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

        np.savez('./data', t=time, w=states, ts=sample_t, ws=sample_y)

    else:
        data = np.load(DataPath)
        if DataPath == './data.npz':
            time, states, sample_t, sample_y    \
                = data['t'], data['w'], data['ts'], data['ws']
        else:
            time, states   \
                = data['t'], data['w']
            time = time[:,np.newaxis]

    
    if Forecast:
        data_span = 0.5
        t_max = time.shape[0]*data_span
        padFactor = int(t_max/num_points)
        sample_t = time[1:int(t_max)-1:padFactor].copy()
        sample_t[-1] = time[int(t_max)].copy()
        sample_y = states[1:int(t_max)-1:padFactor].copy()
        sample_y[-1] = states[int(t_max)].copy()
    else:
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
        ax3.plot(sample_t, sample_y[:,0],'k.')
        ax3.set_xlabel("Time", fontsize=10)
        ax3.set_ylabel("X-Coord", fontsize=10)

        ax4 = fig2.add_subplot(3, 2, 3)
        ax4.plot(time, states[:, 1], 'b')
        ax4.plot(sample_t, sample_y[:,1],'k.')
        ax4.set_xlabel("Time", fontsize=10)
        ax4.set_ylabel("Y-Coord", fontsize=10)

        ax5 = fig2.add_subplot(3, 2, 5)
        ax5.plot(time, states[:, 2], 'b')
        ax5.plot(sample_t, sample_y[:,2],'k.')
        ax5.set_xlabel("Time", fontsize=10)
        ax5.set_ylabel("Z-Coord", fontsize=10)

        ax6 = fig2.add_subplot(3, 2, 2)
        ax6.plot(time, states[:, 3], 'r')
        ax6.plot(sample_t, sample_y[:,3],'k.')
        ax6.set_xlabel("Time", fontsize=10)
        ax6.set_ylabel("X-Coord", fontsize=10)

        ax7 = fig2.add_subplot(3, 2, 4)
        ax7.plot(time, states[:, 4], 'r')
        ax7.plot(sample_t, sample_y[:,4],'k.')
        ax7.set_xlabel("Time", fontsize=10)
        ax7.set_ylabel("Y-Coord", fontsize=10)

        ax8 = fig2.add_subplot(3, 2, 6)
        ax8.plot(time, states[:, 5], 'r')
        ax8.plot(sample_t, sample_y[:,5],'k.')
        ax8.set_xlabel("Time", fontsize=10)
        ax8.set_ylabel("Z-Coord", fontsize=10)

        plt.show()

    # Modify coupling Eqs.
    #eng.edit('LorenzSolver', nargout=0)

    return time, states, sample_t, sample_y


def main(Physics, ApproxErr, SampErr, OptErr, 
         SolFlag=False, PlotFlag=False, BFGS_Flag=False,
         Forecast = False, datPath='./data.npz'):

    try:
        assert(len(Physics) == 6)
        assert(isinstance(Physics[0], (float, int)))
        assert(isinstance(Physics[1], (float, int)))
        assert(len(Physics[2]) == 3)
        assert(len(Physics[3]) == 3)
        assert(len(Physics[4]) == 3)
        assert(len(Physics[5]) == 6)
        t0, T, DriverPar, RespPar, CoupPar, InitV = Physics
        Param = [DriverPar, RespPar, CoupPar]

        assert(len(ApproxErr) == 3)
        assert(ApproxErr[0][0] == 1)
        assert(ApproxErr[0][-1] == 6)
        assert(isinstance(ApproxErr[1], str))
        assert(isinstance(ApproxErr[2], str))
        NNarcht, ActFunc, InitWeights = ApproxErr

        assert(len(SampErr) == 4)
        assert(isinstance(SampErr[0], int))
        assert(isinstance(SampErr[1], int))
        assert(isinstance(SampErr[2], int))
        assert(isinstance(SampErr[3], int))
        nDom, nBound, nObs, nTest = SampErr

        assert(len(OptErr) == 5)
        assert(isinstance(OptErr[0], int))
        assert(isinstance(OptErr[1], float))
        assert(len(OptErr[2]) == 6)
        assert(len(OptErr[3]) == 6)
        assert(len(OptErr[4]) == 6)
        TrainingTime, LearnRate, weights_PDE, weights_IC, weights_OBS = OptErr
        lossWeights = weights_PDE + weights_IC + weights_OBS

    except AssertionError as AE:
        print('\nError Reading in Parameter Lists, format should be as follows:\n ')
        print(
            'Physics Parameters: \nfloat: t0, float: T, list[3xfloat]: DriverPar, list[3xfloat]: RespPar, list[3xfloat]: CoupPar, list[6xfloat]: InitV0\n')
        print(
            'Approx. Error Parameters: \nlist[1,WxT,6]: NNArcht, str: ActFunc, str: InitWeights\n.')
        print('Sample Error Parameters: \nint: nDom, int: nBound, int: nObs, int: nTest\n')
        print(
            'Optimization Error Parameters: \nint: epochs, float: learning rate, list[6xfloat]: Driver_LossWeights, list[6xfloat]: Response_LossWeights\n')

        print('Settings reverted to base problem(See Section 5 of report)')
        Phys = [0.0, 3.0, [10., 28., (8./3.)], [10., 12., (8./3.)],
                [0.1, 0.1, 0.1], [13.561, 10.162, 36.951, 3.948, 4.187, 11.531]]
        t0, T, DriverPar, RespPar, CoupPar, InitV = Phys
        Param = [DriverPar, RespPar, CoupPar]

        # NNarcht, ActFunc, InitWeights
        AErr = [[1] + [100]*4 + [6], 'swish', 'Glorot uniform']
        NNarcht, ActFunc, InitWeights = AErr

        # nDom, nBound, nObs, nTest
        SErr = [400, 2, 100, 400]
        nDom, nBound, nObs, nTest = SErr

        # TrainingTime(epochs), LearnRate, Residual Loss, Initial Value Loss, Data Loss
        OErr = [60000, 1e-3, [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1], [1e2, 1e2, 1e2, 1e2, 1e2, 1e2]]
        TrainingTime, LearnRate, weights_PDE, weights_IC, weights_OBS = OErr
        lossWeights = weights_PDE + weights_IC + weights_OBS

    C1 = dde.Variable(1.0)
    C2 = dde.Variable(1.0)
    C3 = dde.Variable(1.0)
    C4 = dde.Variable(1.0)
    C5 = dde.Variable(1.0)
    C6 = dde.Variable(1.0)

    R1 = dde.Variable(1.0)
    R2 = dde.Variable(1.0)
    R3 = dde.Variable(1.0)

    def Coupled_lorenz_system(x, y):

        y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        y4, y5, y6 = y[:, 3:4], y[:, 4:5], y[:, 5:6]

        f1 = 7.3*y1 + tf.math.cos(y1)
        f2 = 1.2*y2 + 0.5*tf.math.atan(y2)
        f3 = 3.5*y3 + 0.9*tf.math.exp(-y3)

        dy1_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy2_x = dde.grad.jacobian(y, x, i=1, j=0)
        dy3_x = dde.grad.jacobian(y, x, i=2, j=0)

        dy4_x = dde.grad.jacobian(y, x, i=3, j=0)
        dy5_x = dde.grad.jacobian(y, x, i=4, j=0)
        dy6_x = dde.grad.jacobian(y, x, i=5, j=0)

        return [
            dy1_x - C1 * (y2 - y1),
            dy2_x - y1 * (C2 - y3) + y2,
            dy3_x - y1 * y2 + C3 * y3,

            dy4_x - C4 * (y5 - y4) - R1*f1,
            dy5_x - y4 * (C5 - y6) + y5 - R2*f2,
            dy6_x - y4 * y5 + C6 * y6 - R3*f3,
        ]

    def boundary(_, on_initial):
        return on_initial

    geom = dde.geometry.TimeDomain(t0, T)

    # Initial conditions
    ic1 = dde.icbc.IC(geom, lambda X: InitV[0], boundary, component=0)
    ic2 = dde.icbc.IC(geom, lambda X: InitV[1], boundary, component=1)
    ic3 = dde.icbc.IC(geom, lambda X: InitV[2], boundary, component=2)
    ic4 = dde.icbc.IC(geom, lambda X: InitV[3], boundary, component=3)
    ic5 = dde.icbc.IC(geom, lambda X: InitV[4], boundary, component=4)
    ic6 = dde.icbc.IC(geom, lambda X: InitV[5], boundary, component=5)

    # Get the train data
    # Set SolFlag to true if matlab API for python enabled
    sampled_t, sampled_y, observe_t, ob_y = gen_traindata(
        nObs, Param, InitV, tEnd=T, PlotFlag=PlotFlag, SolFlag=SolFlag,
        Forecast= Forecast, DataPath=datPath)
    observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
    observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
    observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)
    observe_y3 = dde.icbc.PointSetBC(observe_t, ob_y[:, 3:4], component=3)
    observe_y4 = dde.icbc.PointSetBC(observe_t, ob_y[:, 4:5], component=4)
    observe_y5 = dde.icbc.PointSetBC(observe_t, ob_y[:, 5:6], component=5)

    data = dde.data.PDE(
        geom,
        Coupled_lorenz_system,
        [ic1, ic2, ic3, ic4, ic5, ic6,
         observe_y0, observe_y1, observe_y2,
         observe_y3, observe_y4, observe_y5],
        num_domain=nDom,
        num_boundary=nBound,
        anchors=observe_t,
        num_test=nTest
    )

    net = dde.nn.FNN(NNarcht, ActFunc, InitWeights)

    model = dde.Model(data, net)
    variable = dde.callbacks.VariableValue(
        [C1, C2, C3, C4, C5, C6, R1, R2, R3], period=1000, filename="variables.dat"
    )
    model.compile("adam", lr=LearnRate, external_trainable_variables=[C1, C2, C3, C4, C5, C6, R1, R2, R3],
                  loss_weights=lossWeights)

    def print_modelSpec():
        print(
            f'\nNN Archt : {NNarcht} | ActFunc : {ActFunc} | Initializer : {InitWeights}')
        print(
            f'Residual Points : {nDom} | Bound Points : {nBound} | Obs Points : {observe_t.shape[0]} | Test Points : {nTest}')
        print(f'Epochs : {TrainingTime} | Learning Rate : {LearnRate}')
        print(
            f'Residual Weights : {weights_PDE} | Init. Cond. Weights : {weights_IC} | Data Weights : {weights_OBS} \n')

    print_modelSpec()
    losshistory, train_state = model.train(epochs=TrainingTime, callbacks=[variable])
    print_modelSpec()

    if BFGS_Flag:
        model.compile("L-BFGS", dde.optimizers.set_LBFGS_options(maxcor=50),
                      external_trainable_variables=[C1, C2, C3, C4, C5, C6, R1, R2, R3], loss_weights=lossWeights)
        # ,model_save_path='Models/Lorenz/Inverese_Simple_BFGS')
        losshistory, train_state = model.train(
            epochs=TrainingTime*0.5, callbacks=[variable])

    y_predict = model.predict(sampled_t)
    L2_relError = dde.metrics.l2_relative_error(sampled_y, y_predict)

    if PlotFlag:
        # Plot# Plot
        fig1 = plt.figure()
        fig1.suptitle(str(Param)+'Reference Solution.', fontsize=18)

        ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
        ax1.plot(sampled_y[:, 0], sampled_y[:, 1],
                 sampled_y[:, 2], 'k-', label="Data Trace", lw=0.5)
        ax1.plot(ob_y[:, 0], ob_y[:, 1], ob_y[:, 2], 'k.',
                 label="Obs. Data #"+str(observe_t.shape[0]), lw=0.1)
        ax1.set_xlabel("X Axis", fontsize=5)
        ax1.set_ylabel("Y Axis", fontsize=5)
        ax1.set_zlabel("Z Axis", fontsize=5)
        ax1.set_title("Driver System", fontsize=10)

        ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
        ax2.plot(sampled_y[:, 3], sampled_y[:, 4],
                 sampled_y[:, 5], 'k-', label="Data Trace", lw=0.5)
        ax2.plot(ob_y[:, 3], ob_y[:, 4], ob_y[:, 5], 'k.',
                 label="Obs. Data #"+str(observe_t.shape[0]), lw=0.1)
        ax2.set_xlabel("X Axis", fontsize=5)
        ax2.set_ylabel("Y Axis", fontsize=5)
        ax2.set_zlabel("Z Axis", fontsize=5)
        ax2.set_title("Response System", fontsize=10)

        fig3 = plt.figure()
        fig3.suptitle(str(Param)+'PINN Solution', fontsize=18)

        ax3 = fig3.add_subplot(1, 2, 1, projection='3d')
        ax3.plot(y_predict[:, 0], y_predict[:, 1],
                 y_predict[:, 2], 'b', label='Driver', lw=0.5)
        ax3.set_xlabel("X Axis", fontsize=5)
        ax3.set_ylabel("Y Axis", fontsize=5)
        ax3.set_zlabel("Z Axis", fontsize=5)
        ax3.set_title("Driver System", fontsize=10)

        ax4 = fig3.add_subplot(1, 2, 2, projection='3d')
        ax4.plot(y_predict[:, 3], y_predict[:, 4],
                 y_predict[:, 5], 'r', label='Response', lw=0.5)
        ax4.set_xlabel("X Axis", fontsize=5)
        ax4.set_ylabel("Y Axis", fontsize=5)
        ax4.set_zlabel("Z Axis", fontsize=5)
        ax4.set_title("Response System", fontsize=10)

        fig2 = plt.figure()
        ax5 = fig2.add_subplot(3, 2, 1)
        ax5.plot(sampled_t, sampled_y[:, 0], 'k', label="Data Trace")
        ax5.plot(sampled_t, y_predict[:, 0], 'b--', label='PINN Sol.')
        ax5.set_title("X1 - Convective Intensity", fontsize=10)
        ax6 = fig2.add_subplot(3, 2, 2)
        ax6.plot(sampled_t, sampled_y[:, 3], 'k', label="Data Trace")
        ax6.plot(sampled_t, y_predict[:, 3], 'r--', label='PINN Sol')
        ax6.set_title("X2 - Convective Intensity", fontsize=10)

        ax7 = fig2.add_subplot(3, 2, 3)
        ax7.plot(sampled_t, sampled_y[:, 1], 'k')
        ax7.plot(sampled_t, y_predict[:, 1], 'b--')
        ax7.set_title("Y1 - Horizontal Temperature Variation", fontsize=10)
        ax8 = fig2.add_subplot(3, 2, 4)
        ax8.plot(sampled_t, sampled_y[:, 4], 'k')
        ax8.plot(sampled_t, y_predict[:, 4], 'r--')
        ax8.set_title("Y2 - Horizontal Temperature Variation", fontsize=10)

        ax9 = fig2.add_subplot(3, 2, 5)
        ax9.plot(sampled_t, sampled_y[:, 2], 'k')
        ax9.plot(sampled_t, y_predict[:, 2], 'b--')
        ax9.set_title("Z1 - Vertical Temperature Variation", fontsize=10)
        ax10 = fig2.add_subplot(3, 2, 6)
        ax10.plot(sampled_t, sampled_y[:, 5], 'k')
        ax10.plot(sampled_t, y_predict[:, 5], 'r--')
        ax10.set_title("Z1 - Vertical Temperature Variation", fontsize=10)

        fig1.tight_layout()
        fig1.legend(loc="lower left", borderaxespad=0.1)
        fig2.tight_layout()
        fig2.legend(loc="lower left", borderaxespad=0.1)
        fig3.tight_layout()
        fig3.legend(loc="lower left", borderaxespad=0.1)

        plt.show()

    return model, sampled_y, sampled_t


if __name__ == "__main__":
   # t0, T, DriverPar, RespPar, CoupPar, InitV0
    Time = input('Desired End-Time [10,25] : ')
    Forecast_Flag = str(input('Forecast [y|n] : '))
    Forecast_Flag = 'y' == Forecast_Flag
    if Forecast_Flag:
        f_flag = True
        Domain_Flag = input('Outside trained domain [y|n] : ')
        Domain_Flag = 'y' == Domain_Flag
        if Domain_Flag:
            T_End = 0.5*float(Time)
        else:
            T_End = float(Time)    
    else:
        f_flag = False
        T_End = float(Time)
    physics = [0.0, T_End, [10., 28., (8./3.)], [10., 12., (8./3.)],
               [0.1, 0.1, 0.1], [13.561, 10.162, 36.951, 3.948, 4.187, 11.531]]

    # NNarcht, ActFunc, InitWeights
    approxErr = [[1] + [128]*3 + [6], 'swish', 'Glorot normal']

    # nDom, nBound, nObs, nTest
    sampErr = [600, 2, 200, 200]

    # TrainingTime(epochs), LearnRate, Residual Loss, Initial Value Loss, Data Loss
    res_wlist = (1e0*np.full(6, 1)).tolist()
    ic_wlist = (1e0*np.full(6, 1)).tolist()
    data_wlist = (1e2*np.full(6, 1)).tolist()
    optErr = [int(1e5), 1e-3, res_wlist, ic_wlist, data_wlist]

    pinn, y_plot, t_plot = main(physics, approxErr, sampErr, optErr,
                   PlotFlag=True, BFGS_Flag=True, Forecast=f_flag,
                        datPath='./data_T'+str(Time)+'_1e5.npz')
    y_pred = pinn.predict(t_plot)
    L2_relError = dde.metrics.l2_relative_error(y_plot, y_pred)
    print(f'L2 Error  :{L2_relError}')
