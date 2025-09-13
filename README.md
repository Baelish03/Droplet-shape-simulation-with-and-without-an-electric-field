# Water Droplet in an Electric Field

This repository contains the simulation codes developed for my undergraduate thesis in Physics at the University of Padua.  
The project investigates the **shape and dynamics of a water droplet**, first in absence of external forces, and then under the influence of a **dipole electric field**.  

The study combines concepts from **capillarity, electrostatics, and fluid dynamics**, with numerical methods for solving equilibrium equations, computing curvature, and simulating droplet deformation.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Physics Background](#physics-background)  
   - [Droplet without Electric Field](#droplet-without-electric-field)  
   - [Droplet in an Electric Field](#droplet-in-an-electric-field)  
3. [Numerical Methods](#numerical-methods)  
4. [Features](#features)  
5. [Example Results](#example-results)  
6. [Requirements](#requirements)  
7. [Usage](#usage)  
8. [Repository Structure](#repository-structure)  
9. [Acknowledgments](#acknowledgments)  
10. [License](#license)  

---

## Introduction

Water droplets are a classical system for studying the interplay between **surface tension** and external stresses.  
A free droplet on a rigid surface tends to have a fixed contact angle.
When an electric field is applied, however, additional stresses appear on the droplet surface due to the interaction between the conductive drop and the field.  

This project explores both regimes:  
- **Without electric field:** equilibrium shape determined only by hydrostatic and surface tension.  
- **With electric field:** deformation due to electrostatic pressure, leading to prolate shapes and dynamic evolution.

---

## Physics Background

### Droplet Without Electric Field

A static droplet in free space is governed by the **Laplace pressure balance**:

$\Delta p = \alpha (\kappa_1 + \kappa_"2)$

where  
- $\Delta p$ = pressure difference across the interface,  
- $\alpha$ = surface tension coefficient,  
- $\kappa_i$ = mean curvature of the interface on principal planes.  

For a generic droplet with principal radius $R_1$ and $R_2$:

$\kappa_i = \frac{1}{R_i}, \quad \Delta p = \alpha \left( \frac{1}{R_1} + \frac{1}{R_2} \right)$

This provides the baseline against which deformations in external fields can be measured.

---

### Droplet in an Electric Field

When a droplet is subjected to a uniform electric field $\mathbf{E}$, additional stresses arise from the **Maxwell stress tensor**:

$T_{ij} = \varepsilon_0 \left( E_i E_j - \frac{1}{2} E^2 \delta_{ij} \right)$

The normal component of the electric stress contributes an effective **electrostatic pressure**:

$p_\text{elec} = \tfrac{1}{2}\,\epsilon_0\,E^2$

The pressure balance in a **droplet section** then becomes:

$p_\text{elec} + \alpha \kappa_1 + p_0 + p_\text{hyd} = 0$

where:
- $p_0$ is a correction pressure to conserve volume,
- $p_{hyd} = \rho g h$ is Stevino's law.

The competition between capillary forces and electrostatic stresses results in prolate deformation. 

---

## Numerical Methods

The simulation framework is written in **Python** and implements the following methods:

- **Curvature calculation** in polar coordinates  
- **Volume conservation** via Simpson’s integration rule  
- **Electrostatic pressure evaluation** on the droplet surface  
- **Exact sparse linear solvers** (`pypardiso`) for large Poisson problems
- **Fast sparse linear solvers** (`pyAMG`)  
- **Time evolution** of droplet deformation (dynamic regime)  
- **Visualization** with Matplotlib  

---

## Features

- Baseline calculation of spherical droplet shape without field  
- Static equilibrium shape in presence of an external electric field  
- Pressure distribution along the interface  
- Volume conservation checks  
- Dynamic evolution of droplet deformation  
- Automatic generation of plots and numerical output  

---

## Results
- Without electric field:
  - shape of the droplet
  - execution time
- With electric field:
  - time evolution of droplet shape under increasing field strengths 
  - execution time
  - electrostatic pressure distribution along the surface  
  - plot of curvature 

---

## Requirements

- Python ≥ 3.9  
- NumPy  
- SciPy  
- Matplotlib  
- [pypardiso](https://github.com/haasad/PyPardiso) (for exact sparse matrix solvers)
- [pyAMG](https://github.com/pyamg/pyamg) (for fast sparse matrix solvers)

Install dependencies with:

```bash
pip install numpy scipy matplotlib pypardiso pyamg
```
