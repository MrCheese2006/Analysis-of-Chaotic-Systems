import Lorenz_Attractor_Analysis_Code as lrnz
import numpy as np


# Initial Conditions

# choose your own initial conditions
x_0 = 0.9
y_0 = 0
z_0 = 0
init = np.array([x_0, y_0, z_0])

# pre-set initial conditions for each system
init_lorenz = np.array([1, 1, 1])
init_rf = np.array([-1, 0, 0.5])
init_chen = np.array([-10, 0, 37])
init_henon = np.array([0, 0])
init_m_chua = np.array([0, 0, 0])
init_duffing = np.array([0, 0, 0])

init = init_lorenz

# step
dt = 0.01
num_steps_to_stop = 100000 # the number of steps that the simulation goes through before stopping

params_custom = [] # choose your own parameters

# pre-set parameters for each system
params_lorenz = [10, 28, 8/3] # sigma, rho, Beta
params_rf = [0.14, 0.1] # a, b
params_chen = [35, 3, 28] # a, b, c
params_henon = [1.4, 0.3] # a, b
params_m_chua = [10.814, 14.0, 1.3, 0.11, 8, np.pi] # alpha, Beta, a, b, c, d. Note that c is odd => d = 0, c is even => d = pi
params_duffing = [1, -1, 0.2, 0.3, 1] # aplha, Beta, gamma, delta, omega 

# choose which parameter set you could like to use while modelling
params = params_lorenz

# Commands:
runtime = 1 # get runtime
system = lrnz.lorenz # choose system to model
# modelling methods:
EM = 0
improved_EM = 0
RK4 = 1
RK8 = 0
model_henon = 0
# plotting:
plot_all = 0
plot = 0
plot_xy = 0
plot_xz = 0
plot_yz = 0
# analysis:
sensitive_dependance, disturbance = 1, 0.0001
orbit_sep, method, sub_method, d0 = 0, lrnz.runge_kutta, lrnz.runge_kutta_4, 1e-8
GS = 0
Poincare = 0
discard = 100
modelling_error = 0
error_comparison, log_scale = 0, 0

lrnz.run(init, dt, num_steps_to_stop, params, runtime, system, EM, improved_EM, RK4, RK8, model_henon, plot_all, plot, plot_xy, 
         plot_xz, plot_yz, sensitive_dependance, disturbance, orbit_sep, method, sub_method, d0, GS, Poincare, discard, modelling_error, error_comparison, log_scale)
