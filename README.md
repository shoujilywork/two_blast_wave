# Blast Wave Interaction Simulation

A high-performance CFD solver for simulating blast wave interactions using OpenMP and CUDA parallelization.

## Features
- Solves 2D Euler equations for compressible flow
- Lax-Friedrichs flux splitting
- Hybrid OpenMP+CUDA parallel implementation
- RK4 time integration for stability

## Installation
### Prerequisites
- CUDA Toolkit ≥ 11.0
- OpenMP-enabled compiler
- Eigen3 library

README: Two Blast Wave Interaction Simulation
This program simulates the ​​interaction of two blast waves​​ in a 2D domain, solving the Euler equations with high-order numerical methods. Below are the key features:

​​Physical Model​​
Solves the ​​Euler equations​​ for compressible flow, modeling blast wave propagation and interaction.
Uses ​​Lax-Friedrichs flux splitting​​ and high order scheme for shock-capturing.
Initial conditions (e.g., u = 1.86, p = 0.99) are set for typical blast wave dynamics.
​​Two-Wave Interaction Mechanism​​
The split_lf function computes fluxes in both ​​x and y directions​​, enabling multi-dimensional wave interactions.
The simulation captures ​​collisions between blast waves​​ and their reflection from boundaries.
​​Application Scenarios​​
The computational grid (65×65, domain x∈[0,2], y∈[0,1.1]) is optimized for small-scale blast experiments.
A ​​V-shaped boundary​​ (tan35 slope) represents blast wave interaction with inclined surfaces

​​Numerical Methods​​

​​Explicit time integration (RK4)​​ for stable time stepping.
​​High-resolution scheme for derivative and discontinuities
Initial pressure (p = 5/7) matches the ​​Friedlander waveform​​ theory for blast overpressure.

​​Conclusion​​

This code simulates ​​2D blast wave collisions and their interaction with geometric boundaries​​, serving as a benchmark for ​​computational dynamics​​.
