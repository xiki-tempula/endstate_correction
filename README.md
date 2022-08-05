endstate_correction
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/wiederm/endstate_correction/workflows/CI/badge.svg)](https://github.com/wiederm/endstate_correction/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/wiederm/endstate_correction/branch/main/graph/badge.svg)](https://codecov.io/gh/wiederm/endstate_correction/branch/main)


Endstate correction from MM to QML potential

## Theory

Following an arbritary thermodynamic cycle to calculate a free energy for a given molecular system at a certain level of theory, we can perform endstate corrections at the nodes of the thermodynamic cycle to a desired target level of theory.
In this work we have performed the endstate corrections using equilibrium free energy calculations and NEQ work protocolls.
### Equilibrium free energy endstate corrections
In equilibrium free energy calculations samples are drawn from the Boltzmann distrubtion at specific interpolation states between thermodynamic states (in our specific case: different energetic descriptions of the molecular system, i.e. the source level of theory and the target level of theroy) and, given sufficient overlap of its pdfs, a free energy can be estimated. This protocoll is expensive (it needs iid samples at each lambda state connecting the Boltzmann distribution at the endstates) but also reliable and accureate (with low variance).

### Non equilibrium work protocol 
Non-equilibrium work protocolls, and the fluctuation theorems connecting nonequilibrium driven processes to equilibrium properties, can be used to estimate free energy differences between different levels of theory efficiently. 
A specific NEQ protocol typically consists of a perturbation and a propagation kernel, which are used to drive the system out of equilibrium.  

### Copyright

Copyright (c) 2022, Sara Tkaczyk, Johannes Karwounopoulos & Marcus Wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
