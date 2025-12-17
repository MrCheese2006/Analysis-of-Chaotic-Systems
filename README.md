# Lorenz Attractor and other Chaotic System Modelling and Analysis

## Basic Overview

This code provides some analysis and modelling tools for various systems of ordinary differential equations. It currently includes implementations for the Lorenz, Rabinovich-Fabrikant, Chen, Hénon, Modified Chua and Duffing systems. Basic knowledge of Python is recommended for use of this code. In order to run the code, the following libraries need to be installed: ```numpy```, ```matplotlib```, and ```time```. This code supports Python versions 3.12 - 3.14.

The code has the following implementations:
- Modelling in 3-dimensional space (where applicable), as well as their projections on the xy, xz and yz planes. The following modelling methods are implemented:
    - Euler's Method (EM)
    - Improved Euler's Method (IEM)
    - Runge-Kutta 4 (RK4)
    - Runge-Kutta 8 (RK8)
- Plotting a Poincaré map on a user-defined plane. Only constant planes are supported.
- Determining the spectrum of Lyapunov exponents of a system with the Modified Gram-Schmidt Orthonormalization method
- Determining the maximal Lyapunov exponent using the orbit separation method
- Calculate the average Lyapunov spectrum or maximal value for the respective methods above
- Using Richardson Extrapolation to determine the approximate running error. This can be used in a comparison of error for the various modelling methods
- Plotting the difference in systems with a small perturbation to the initial condition in the $x$ direction to visually depict the deterministic chaotic behaviour these systems can have.

## Code Use Instructions

All of the code is run and managed from the ```run.py``` file through a series of command sections. This is where the initial conditions, system parameters, number of steps and step size are chosen. It is also where the commands are placed for which analysis method to use. This is done by making the desired command variable equal to 1. Each analysis method has some additional parameters that need to be specified, some of while are repeated across methods. These repeated parameters are placed below the commands under the comment block ```Additional Commands for the noted analysis methods```. The required use of these parameters will be discussed in more detail for each analysis method.

#### Initial Conditions
Initial conditions (IC's) are required for all analysis methods except for computing the average Lyapunov spectrum or maximal value. Pre-set IC's are available for use for each system, or one can create their own custom IC's. The chosen IC must be specified in the ```init``` variable. For example, if one wanted to use the initial conditions for the Lorenz system or use custom parameters, they would set ```init = init_lorenz``` or ```init = init_custom```, respectively.

#### Parameters
Parameters are required for all analysis methods, and have a similar functionality to the IC's. Pre-set or custom parameters are available for use and are specified in the same way as the IC's are. Note that each system has a set amount of parameters (see below), so one must be aware of how many parameters to use for their desired system.

Parameters for each system with some traditional values:
- Lorenz: $(\sigma, \text{ } \rho, \text{ } \beta) = (10, \text{ } 28, \text{ } 8/3)$
- Rabinovich-Fabrikant: $(a, \text{ } b) = (0.1, \text{ } 0.1)$ 
- Chen: $(a, \text{ } b, \text{ } c) = (35, \text{ } 3, \text{ } 28)$
- Hénon: $(a, \text{ } b) = (1.4, \text{ } 0.3)$
- Modified Chua: $(\alpha, \text{ } \beta, \text{ } a, \text{ } b, \text{ } c, \text{ } d) = (10.814, \text{ } 14.0, \text{ } 1.3, \text{ } 0.11, \text{ } 8, \text{ } \pi)$
    - Note that if $c\in \{0, \mathbb{Z}^{+}\}$ is even, then $d = \pi$. If $c$ is odd, then  $d=0$.
- Duffing: $(\alpha, \text{ } \beta, \text{ } \gamma, \text{ } \delta, \text{ } \omega) = (1, \text{ } -1, \text{ } 0.2, \text{ } 0.3, \text{ } 1)$

All of the above traditional parameters are hard-coded into the code already.

#### Step Size, Number of Steps and Runtime

The step size specifies how large of a step one would like while modelling the chosen system. It is also required in all analysis functions. A smaller step size makes the model closer to the exact value but won't travel as far into time. This value can be changed with the ```dt``` variable in the code.

The number of steps determines how long your code runs for and how far into time the model is run for, depending on the step size. Note that a large number of steps will result in longer computation times. The number of steps is set by the variable ```num_steps_to_stop```.

Runtime is a variable that can be set to ```1``` or ```0``` to print out the computation time for modelling and/or analysis. It does not affect the functionality of these methods, and is only present for ones interest.

#### Modelling

In order to model a system the code requires the use of the following command sections: System, Modelling methods and Plotting.

1. System: 
    The system section consists of a variable ```system``` which allows one to choose the system that they would like to model. Each system has its own function that must be called, and are specified as follows:
    - Lorenz: ```lrnz.lorenz```
    - Rabinovich-Fabrikant: ```lrnz.rabinovich_fabrikant```
    - Chen: ```lrnz.chen```
    - Hénon: ```lrnz.henon```
    - Modified Chua: ```lrnz.modified_chua```
    - Duffing: ```lrnz.duffing``` <br> <br>

2. Modelling Methods:
    The modelling methods section of the ```run.py``` file has the following commands available:

    ```python
    EM = 0
    improved_EM = 0
    RK4 = 0
    RK8 = 0
    model_henon = 0
    ```

    In order to model a desired system one can choose their desired method by placing a ```1``` next to the variable for the chosen method. For the Hénon system, set ```model_henon = 1``` and all others to ```0```. Only choose one modelling method at a time. <br> <br>

3. Plotting: 
    The plotting section of the code determines what is plotted. The code supports plotting in 3D and projections on the $xy$, $xz$ and $yz$ axes. The following commands are available:

    ```python
    plot_all = 0
    plot = 0
    plot_xy = 0
    plot_xz = 0
    plot_yz = 0
    ```

    To plot all the graphs, set ```plot_all = 1```. Setting ```plot = 1``` and all others to zero plots only the 3D model. The other three commands plot their respective projections to the orthogonal axes. The commands can be combined as one wishes.

For example, if one wishes to plot the Modified Chua System in 3D and with the $xy$ projection, using the RK4 modelling method they would input the following commands, with their desired ```dt``` and ```num_steps_to_stop```:

```python
# Choose which system you would like to model
system = lrnz.modified_chua

# Modelling methods:
EM = 0
improved_EM = 0
RK4 = 1
RK8 = 0
model_henon = 0

# Plotting:
plot_all = 0
plot = 1
plot_xy = 1
plot_xz = 0
plot_yz = 0
```

### Analysis

The following analysis tools are implemented in the code:

1. Sensitive Dependence
2. Lyapunov Exponents
    - Orbit Separation
    - Modified Gram-Schmidt Orthonormalization
    - Average Lyapunov Exponent and Uncertainties
3. Poincaré Maps
4. Modelling Error

Each of the above tools are explained in their own section below. Note that the code was developed primarily for use with the Lorenz system, so analysis functionality with the other systems was not sufficiently tested. Additionally, overlap of variables may result in errors if multiple analysis methods are run at once, so only doing one at a time is recommended.

#### Sensitive Dependence

The commands for the sensitive dependence section of the code are as follows:

```python
sensitive_dependence = 0
disturbance = 0.0001
```

Setting ```sensitive_dependence = 1``` results in the system being modelled twice - once with the initial conditions set in the Initial Conditions command section, and another with the same initial conditions, but the initial x value is disturbed by the amount ```disturbance```. A plot of the differences between the 2 models is generated and saved under ```figures/analysis/Sensitive_Dependence/sensitive_dependence_{system name}.png``` in the working directory. One can modify the ```disturbance``` value to investigate how slight changes in initial conditions results in large changes in final values in deterministic chaotic systems. To calculated the sensitive dependence, the system must be manually modelled using the tools outlined earlier.

#### Lyapunov Exponents

Two different methods are implemented to determine the Lyapunov exponent(s) of a system: Orbit separation and the Modified Gram-Schmidt Orthonormalization Method. As previously mentioned, Orbit Separation determines the maximal Lyapunov exponent and the modified Gram-Schmidt Orthonormalization Method determines the spectrum of Lyapunov exponents. Additionally, using either of these methods, one can determine the average Lyapunov exponent, along with its error. The command section for these analysis tools are as follows: 

```python
# Orbit separation:
orbit_sep = 0
plot_running = 'y' # y or n

# Modified Gram-Schmidt Orthonormalization Method
GS = 0

# Average Lyapunov Exponent and Uncertainties
average_lyapunov = 1
lyapunov_method = lrnz.orbsep
num_iterations = 1000

# Additional commands for the noted analysis methods
method = lrnz.runge_kutta # Orbit Separation and Average Lyapunov
sub_method = lrnz.runge_kutta_4 # Orbit Separation and Average Lyapunov
d0 = 1e-8 # orbit separation and Average Lyapunov
discard = 100 # Orbit Separation, Gram-Schmidt and Average Lyapunov
```

The final section, ```# Additional commands for the noted analysis methods```, contains commands used in the noted analysis methods. They were combined into one section to avoid variable overlap. Each section is explained below.

**Orbit Separation:**

In order to determine the maximal Lyapunov exponent using Orbit separation, one must use the ```orbit_sep``` (set ```orbit_sep = 1```), ```plot_running```, ```method```, ```sub_method```, ```d0``` and ```discard``` commands. One must also specify some initial conditions and the system the wish to model, as discussed earlier.

```plot_running``` determines if one would like to plot the running maximal Lyapunov exponent and accepts ```'y'``` (yes) or ```'n'``` (no) as inputs. 

The ```method``` and ```sub_method``` variables determine what modelling method is used. The possibilities are as follows:

```python
method, sub_method = lrnz.eulers_method, None
method, sub_method = lrnz.improved_eulers_method, None
method, sub_method = lrnz.runge_kutta, lrnz.runge_kutta_4
method, sub_method = lrnz.runge_kutta, lrnz.runge_kutta_8
```

Only the RK methods need a sub-method. 

Orbit Separation uses a disturbance to calculate the Lyapunov exponent. This is represented by the variable ```d0```.

In order to avoid calculation errors, an initial set of values should be discarded from the modelling. ```discard = x``` means that ```x``` values will be discarded. 

It is recommended to leave the ```d0``` and ```discard``` values unchanged.

The maximal Lyapunov exponent will be printed to the terminal, and if ```plot_running = 'y'``` then the generated figure will be saved to ```figures/analysis/Lyapunov_Exponents/figures/analysis/Orbit_Separation_Running_Lyapunov_Values_{system_name}.png```

**Modified Gram-Schmidt Orthonormalization Method:**

Using the Modified Gram-Schmidt Orthonormalization Method requires setting ```GS = 1```, as well as fully modelling the desired system, as discussed earlier. As with Orbit Separation, the ```discard``` variable discards an initial set of values from the calculation. It is recommended to leave this unchanged. The spectrum of Lyapunov exponents will be printed to the terminal.

**Average Lyapunov:**

To determine the Average Lyapunov exponent, set ```orbit_sep = 0```, ```GS = 0``` and ```average_lyapunov = 1```. The other  commands used are ```lyapunov_method```, ```num_iterations```, ```method```, ```sub_method```, ```d0``` and ```discard```. Setting ```lyapunov_method = lrnz.orbsep``` or ```lyapunov_method = lrnz.GS_ortho``` determines the average Lyapunov value(s) using the Orbit Separation Method or the Modified Gram-Schmidt Orthonormalization Method, respectively. ```num_iterations``` specifies the number of times the Lyapunov values are calculated, enabling the average and error calculations. The other commands have the same functionality as earlier described, however all need to be set, regardless of the choice of ```lyapunov_method```. The parameters, step size and number of steps also need to be specified, however the initial conditions do **not**. The average Lyapunov exponent(s) and their error will be printed to the terminal. If ```lyapunov_method = lrnz.orbsep``` is chosen, the running exponent values will not be plotted.

#### Poincaré Maps

The command section for this analysis tool is the following:

```python
# Poincare Map
Poincare = 0
```

To plot a Poincaré Map for a system, set ```Poincare = 1```. Before running, the commands for a model of the system need to be set up. This includes setting the parameters, initial conditions, step size, number of steps, ```system``` and a modelling method. After running the code, two prompts will appear in the terminal. The first one is ```"Input axis of intersecting plane (x, y or z): " ```, and the second one is ```"Input equation for constant plane of intersection: "```. The first input gets the user to choose what axis they want their intersecting plane to be on. The inputs $(x, y, z) \text{ or } (X, Y, Z)$ for the standard axes of $\mathbb{R}^3$ are accepted. The second input gets the user to input the point along the specified axis at which the plane will be constructed. As previously mentioned, this code only provides support for constant planes. For example, if the inputs to the first and second prompts were respectively chosen as ```z``` and ```25```, a Poincaré map would be produced of the intersections of the model with the plane $z = 25$. A 3D plot of the model and the specified plane is also produced. All plots will be saved to ```figures/analysis/Poincare_Maps/Poincare_Map_{system_name}_{axis}``` in the working directory. A text file of the intersection points will also be created and saved to ```figures/analysis/Poincare_Maps/Intersection_Points/poincare_intersections_{system_name}_{axis}={plane}.txt``` in the working directory. ```axis``` and ```plane``` are the user inputs.

#### Modelling Error

The command section for this analysis tool is as follows:

```python
# Modelling Error
modelling_error = 0
error_comparison = 0
log_scale = 0
```

In order to calculate the modelling error of a single model, set ```modelling_error = 1```. Before running, the commands for a model of the system need to be set up. This includes setting the parameters, initial conditions, step size, number of steps, ```system``` and a modelling method. By setting ```error_comparison = 1```, a plot of the error for each of the modelling methods, EM, IEM, RK4 and RK8, will be generated to facilitate error comparison. Before running, the commands for a model of the system need to be set up. This includes setting the parameters, initial conditions, step size, number of steps and ```system```. A modelling method does not need to be chosen. All plots will be produced and saved to ```figures/analysis/Modelling_Error``` in the working directory. In either of the above cases, setting ```log_scale = 1``` plots the figures on a logarithmic scale.



Thats it! Enjoy the code! :)
