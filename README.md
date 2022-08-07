endstate_correction
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/wiederm/endstate_correction/workflows/CI/badge.svg)](https://github.com/wiederm/endstate_correction/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/wiederm/endstate_correction/branch/main/graph/badge.svg)](https://codecov.io/gh/wiederm/endstate_correction/branch/main)


Endstate correction from MM to QML potential

## Theory

Following an arbritary thermodynamic cycle to calculate a free energy for a given molecular system at a certain level of theory, we can perform endstate corrections at the nodes of the thermodynamic cycle to a desired target level of theory.
In this work we have performed the endstate corrections using equilibrium free energy calculations, non-equilibrium (NEQ) work protocolls and free energy perturbation (FEP).

### Equilibrium free energy endstate corrections
In equilibrium free energy calculations samples are drawn from the Boltzmann distrubtion at specific interpolation states between thermodynamic states (in our specific case: different energetic descriptions of the molecular system, i.e. the source level of theory and the target level of theroy) and, given sufficient overlap of its pdfs, a free energy can be estimated. This protocoll is expensive (it needs iid samples at each lambda state connecting the Boltzmann distribution at the endstates) but also reliable and accureate (with low variance).

### Non equilibrium work protocol 
Non-equilibrium work protocolls, and the fluctuation theorems connecting nonequilibrium driven processes to equilibrium properties, can be used to estimate free energy differences between different levels of theory efficiently.
A specific NEQ protocol typically consists of a series of perturbation kernel $\alpha_t(x,y)$ and a propagation kernel $\kappa_t(x,y)$, which are used in an alternating pattern to drive the system out of equilibrium.
Each perturbation kernel $\alpha$ drives an alchemical coupling parameter $\lambda$, and each propagation kernel $\kappa$ propagates the coordinates of the system at fixed $\lambda$ according to a defined MD process.
The free energy difference can then be recovered using either using the Jarzynski equation (if initial conformations to seed the NEQ protocoll are only drawn from $\pi(x, \lambda=0)$ and the NEQ protocoll perturbations only from $\lambda=0$ to $\lambda=1$) or the Crooks' fluctuation theorem (if samples to seed the NEQ protocoll are drawn from $\pi(x, \lambda=0)$ and $\pi(x, \lambda=1)$ and the perturbation kernels are set for a bidirectinoal protocol.

### Free energy perturbation (FEP)

Here, we define FEP as a special case of the NEQ protocol (and the Jarzynski equation) in which the protocol consists only of a single perturbation kernel $\alpha_t(x,y)$, without a propagation kernel. $\alpha_t(x,y)$ perturbates the alchemical DOF from one 'endstate', without any intermediate states, to another 'endstate'. 
In the limiting cases of infinitely fast switching the Jarzynski equality reduced to the well-known FEP equation:
$e^{-\beta \Delta F} = \langle e^{−β[E(x,\lambda=1)− E(x,\lambda=0)]} \rangle_{\lambda=0}$.
$\langle \rangle_{\lambda=0}$ indicate that samples are drawn from the equilibrium distribution $\pi(\lambda=0, x)$.

## Installation

We recommend setting up a new python environment with `python=3.9` and installing the packages defined here using `mamba`: https://github.com/wiederm/endstate_correction/blob/main/devtools/conda-envs/test_env.yaml.
This package can be installed using:
`pip install git+https://github.com/wiederm/endstate_correction.git`.

## How to use this package

### 

### Copyright

Copyright (c) 2022, Sara Tkaczyk, Johannes Karwounopoulos & Marcus Wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
