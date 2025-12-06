# Lorenz Attractor and other Chaotic System Modelling and Analysis

## Basic Overview

This code provides some analysis and modelling tools for various systems of ordinary differential equations. It currently includes implementations for the Lorenz, Rabinovich Fabrikant, Chen, Hénon, Modified Chua and Duffing systems. Basic knowledge of Python is recommended for use of this code.

The code has the following implmentations:
- Modelling in 3-dimensional space (where applicable), as well as their projections on the xy, xz and yz planes. The following modelling methods are implemented:
    - Eulers Method (EM)
    - Improved Eulers Method (IEM)
    - Runge-Kutta 4 (RK4)
    - Runge-Kutta 8 (RK8)
- Plotting a poincare map on a user-defined plane. Only constant planes are supported.
- Determining the spectrum of Lyapunov exponents of a system with the Modified Gram-Schmidt Orthonormalization method
- Determining the maximal Lyapunov exponent using the orbit separation method
- Calculate the average Lyapunov spectrum or maximal value for the respective methods above
- Using Richardson Extrapolation to determine the approximate running error. This can be used in a comparison of error for the various modelling methods
- Plotting the difference in systems with a small purturbation to the initial condition in the \(x\) direction to visually depict the deterministic chaotic behaviour these systems can have.

## Code Use Instructions

A video demonstrating how to use each implementation of the code can be found here: INSERT LINK. Text instructions are below. 

All of the code is run and managed from the ```run.py``` file. This is where the initial conditions, system parameters, number of steps and step size are chosen. This is also where the commands are placed for which analysis method to use. This is done by making the desired command variable equal to 1. Each analysis method has some additional parameters that need to be specified, some of while are repeated across methods. These repeated parameters are placed below the commands under the comment block ```Additional Commands for the noted analysis methods```. The required use of these parameters will be discussed in more detail for each analysis method.

**Initial Conditions:**
Initial conditions (IC's) are required for all analysis methods except for computing the average Lyapunov spectrum or maximal value. Pre-set IC's are availible for use for each system, or one can create their own custom IC's. The chosen IC must be specified in the ```init``` variable. For example, if one wanted to use the initial conditions for the lorenz system or use custom parameters, they would set ```init = init_lorenz``` or ```init = init_custom```, respectively.

**Parameters:**
Parameters are required for all analysis methods, and have a similar functionality to the IC's. Pre-set or custom parameters are availible for use and are specified in the same way as the IC's are. Note that each system has a set amount of paramters (see below), so one must be aware of how many parameters to use for their desired system.

Parameters for each system:
- Lorenz: 3 (\(\sigma\), \(\rho\), \(\beta\))