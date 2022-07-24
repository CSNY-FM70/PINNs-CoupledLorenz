# PINNs-CoupledLorenz
Physics-Informed Neural Netwroks are deep learning models augmented with any empirically validated rule or 
other domain expertise as prior information(e.g. governing physical laws of time-dependant dynamics of a system)
to act as a regularization agent, not only amplifying data information content but constraining the space of admissible
solutions to a managable size.

![PINNs](/Plots/PINNs.PNG)

This work applies the PINN framework to a unilaterally coupled 2-stage Lorenz system in order to study its limitations
under deterministic chaos at varying spatio-temporal scales. The study focuses on three types of errors arising from
the chosen netwrok architecture ($\epsilon_{app}$), the observed data information content and quantity ($\epsilon_{data}$)
and the complexity of the loss-functional to be optimized ($\epsilon_{opt}$).

$$\frac{dx_1}{dt} = \sigma_d(y_1-x_1) \quad \frac{dx_2}{dt} = \sigma_r(y_2-x_2) + \mu_1 f_1(u(t))$$

$$\frac{dy_1}{dt} = x_1(\rho_d-z_1) -y_1 \quad \frac{dy_2}{dt} = x_2(\rho_r - z_2) - y_2 + \mu_2 f_2(u(t))$$

$$\frac{dz_1}{dt} = x_1 y_1 - \beta_d z_1 \quad \frac{dz_2}{dt} = x_2 y_2 - \beta_r z_2 + \mu_3 f_3(u(t))$$

![PINNs](/Plots/error_analysis.PNG)

![Reference System - Driver(b) & Response(r)](/CoupledLorenz/General_Plots/Coupled_System_Sol.png)
![PINN(6x128) Solution Accuracy - Driver(b) & Response(r)](/CoupledLorenz/General_Plots/CLI_T256x128.png)