initV = [13.561,10.162,36.951,3.948,4.187,11.531];
DriverPar = [10., 28., (8./3.)];
ResponsePar = [10., 12., (8./3.)];
CoupPar = [0.1, 0.1, 0.1];
TEnd =25.;
t0 = 0.;
numDat = 1e5;
delta = (TEnd - t0)/numDat;
Time =(t0:delta:TEnd)';
[Time, States] = LorenzSolv(DriverPar, ResponsePar, CoupPar, initV, Time);

x1 = States(:,1);
y1 = States(:,2);
z1 = States(:,3);
x2 = States(:,4);
y2 = States(:,5);
z2 = States(:,6);

subplot(1,2,1);
plot3(x1,y1,z1,'b');
title('Driver');
xlabel('X'); ylabel('Y'); zlabel('Z');


subplot(1,2,2);
plot3(x2,y2,z2,'r')
title('Response');
xlabel('X'); ylabel('Y'); zlabel('Z');

writematrix(Time,'t010.dat')
writematrix(States,'sol010.dat')


function [T,X] = LorenzSolv(DriverPar, RespPar, CouplingPar, initV, T, eps)
% LORENZ Function generates the lorenz attractor of the prescribed values
% of parameters rho, sigma, beta
%
%   [X,Y,Z] = LORENZ(RHO,SIGMA,BETA,INITV,T,EPS)
%       X, Y, Z - output vectors of the strange attactor trajectories
%       RHO     - Rayleigh number
%       SIGMA   - Prandtl number
%       BETA    - parameter
%       INITV   - initial point
%       T       - time interval
%       EPS     - ode solver precision
%
% Example.
%        [X Y Z] = lorenz(28, 10, 8/3);
%        plot3(X,Y,Z);
if nargin<4
  error('MATLAB:lorenz:NotEnoughInputs','Not enough input arguments.'); 
end

if nargin<6
  eps = 0.000001;
%   T = [0 3];
%   initV = [13.561,10.162,36.951,3.948,4.187,11.531];
end

sigma = DriverPar(1); rho = DriverPar(2); beta = DriverPar(3);

sigma_hat = RespPar(1); rho_hat = RespPar(2); beta_hat = RespPar(3); 

mux = CouplingPar(1); muy = CouplingPar(2); muz = CouplingPar(3);

options = odeset('RelTol',eps,'AbsTol',[eps eps eps/10 eps eps eps/10]);
[T,X] = ode45(@(T,X) F(T, X, sigma, rho, beta, sigma_hat, rho_hat, beta_hat, mux,muy,muz), ... 
                T, initV);%, options);

% x1 = X(:,1);
% y1 = X(:,2);
% z1 = X(:,3);
% x2 = X(:,4);
% y2 = X(:,5);
% z2 = X(:,6);
% 
% subplot(1,2,1);
% plot3(x1,y1,z1,'b');
% title('Driver');
% xlabel('X'); ylabel('Y'); zlabel('Z');
% 
% 
% subplot(1,2,2);
% plot3(x2,y2,z2,'r')
% title('Response');
% xlabel('X'); ylabel('Y'); zlabel('Z');
    return
end

function dx = F(T, X, sigma, rho, beta, sigma_hat, rho_hat, beta_hat, mux,muy,muz)
% Evaluates the right hand side of the Lorenz system
% x' = sigma*(y-x)
% y' = x*(rho - z) - y
% z' = x*y - beta*z
% typical values: rho = 28; sigma = 10; beta = 8/3;

    f1 = @(x) 7.3*x(1) + cos(x(1));
    f2 = @(x) 1.2*x(2) + 0.5*atan(x(2));
    f3 = @(x) 3.5*x(3) + 0.9*exp(-x(3));

    dx = zeros(6,1);
    dx(1) = sigma*(X(2) - X(1)) ;
    dx(2) = X(1)*(rho - X(3)) - X(2);
    dx(3) = X(1)*X(2) - beta*X(3);
    
    dx(4) = sigma_hat*(X(5) - X(4)) + mux*feval(f1,X);
    dx(5) = X(4)*(rho_hat - X(6)) - X(5) + muy*feval(f2,X);
    dx(6) = X(4)*X(5) - beta_hat*X(6) + muz*feval(f3,X);
    
    return
end
