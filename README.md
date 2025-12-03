# Lorenz Attractor and other Chaotic System Modelling and Analysis

## Basic Overview

This code provides some analysis and modelling tools for various systems of ordinary differential equations. It currently includes implementations for the Lorenz, Rabinovich Fabrikant, Chen, Hénon, Modified Chua and Duffing systems. 

The code has the following implmentations:
- Modelling in 3-dimensional space (where applicable), as well as their projections on the xy, xz and yz planes. The following modelling methods are implemented:
    - Eulers Method (EM)
    - Improved Eulers Method (IEM)
    - Runge-Kutta 4 (RK4)
    - Runge-Kutta 8 (RK8)
- Plotting a poincare map on a user-defined plane. Only constant planes are supported.
- Determining the spectrum of Lyapunov exponents of a system with the Modified Gram-Schmidt Orthonormalization method
- Determining the maximal Lyapunov exponent using the orbit separation method
- Using Richardson Extrapolation to determine the approximate running error. This can be used in a comparison of error for the various modelling methods
- Plotting the difference in systems with a small purturbation to the initial condition in the \(x\) direction to visually depict the deterministic chaotic behaviour these systems can have.