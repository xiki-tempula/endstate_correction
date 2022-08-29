endstate_correction
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/wiederm/endstate_correction/workflows/CI/badge.svg)](https://github.com/wiederm/endstate_correction/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/wiederm/endstate_correction/branch/main/graph/badge.svg)](https://codecov.io/gh/wiederm/endstate_correction/branch/main)
[![Github release](https://badgen.net/github/release/wiederm/endstate_correction)](https://github.com/florianj77/wiederm/endstate_correction/)
[![GitHub license](https://img.shields.io/github/license/wiederm/endstate_correction?color=green)](https://github.com/wiederm/endstate_correction/blob/main/LICENSE)
[![GH Pages](https://github.com/wiederm/endstate_correction/actions/workflows/build_page.yaml/badge.svg)](https://github.com/wiederm/endstate_correction/actions/workflows/build_page.yaml)
[![CodeQL](https://github.com/wiederm/endstate_correction/actions/workflows/codeql.yml/badge.svg)](https://github.com/wiederm/endstate_correction/actions/workflows/codeql.yml)

[![docs stable](https://img.shields.io/badge/docs-stable-5077AB.svg?logo=read%20the%20docs)](https://wiederm.github.io/endstate_correction/)

[//]: <[![GitHub forks](https://img.shields.io/github/forks/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/network)>
[//]: <[![Github tag](https://badgen.net/github/tag/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/tags/)>
[//]: <[![GitHub issues](https://img.shields.io/github/issues/wiederm/endstate_correction?style=flat)](https://github.com/wiederm/endstate_correction/issues)>
[//]: <[![GitHub stars](https://img.shields.io/github/stars/wiederm/endstate_correction)](https://github.com/wiederm/endstate_correction/stargazers)>


Endstate correction from MM to QML potential

# Theory

Following an arbritary thermodynamic cycle to calculate a free energy for a given molecular system at a certain level of theory, we can perform endstate corrections at the nodes of the thermodynamic cycle to a desired target level of theory.
In this work we have performed the endstate corrections using equilibrium free energy calculations, non-equilibrium (NEQ) work protocols and free energy perturbation (FEP).

![TD_cycle](https://user-images.githubusercontent.com/64199149/183875405-be049fa2-7ba7-40ba-838f-e2d43c4801f4.PNG)


## Equilibrium free energy endstate corrections
In equilibrium free energy calculations samples are drawn from the Boltzmann distrubtion at specific interpolation states between thermodynamic states (in our specific case: different energetic descriptions of the molecular system, i.e. the source level of theory and the target level of theroy) and, given sufficient overlap of its pdfs, a free energy can be estimated. This protocol is expensive (it needs iid samples at each lambda state connecting the Boltzmann distribution at the endstates) but also reliable and accureate (with low variance).
![EQ](https://user-images.githubusercontent.com/64199149/183875892-239d53f0-4caf-4bd6-8f37-349448af7d01.PNG)

## Non-equilibrium work protocol 
Non-equilibrium work protocols, and the fluctuation theorems connecting non-equilibrium driven processes to equilibrium properties, can be used to estimate free energy differences between different levels of theory efficiently.
A specific NEQ protocol typically consists of a series of perturbation kernel $\alpha_t(x,y)$ and a propagation kernel $\kappa_t(x,y)$, which are used in an alternating pattern to drive the system out of equilibrium.
Each perturbation kernel $\alpha$ drives an alchemical coupling parameter $\lambda$, and each propagation kernel $\kappa$ propagates the coordinates of the system at fixed $\lambda$ according to a defined MD process.
The free energy difference can then be recovered using either the Jarzynski equation (if initial conformations to seed the NEQ protocol are only drawn from $\pi(x, \lambda=0)$ and the NEQ protocol perturbations only from $\lambda=0$ to $\lambda=1$) or the Crooks' fluctuation theorem (if samples to seed the NEQ protocol are drawn from $\pi(x, \lambda=0)$ and $\pi(x, \lambda=1)$ and the perturbation kernels are set for a bidirectinoal protocol).

## Free energy perturbation (FEP)

Here, we define FEP as a special case of the NEQ protocol (and the Jarzynski equation) in which the protocol consists only of a single perturbation kernel $\alpha_t(x,y)$, without a propagation kernel. $\alpha_t(x,y)$ perturbates the alchemical DOF from one 'endstate', without any intermediate states, to another 'endstate'. 
In the limiting cases of infinitely fast switching the Jarzynski equality reduces to the well-known FEP equation:
$e^{-\beta \Delta F} = \langle e^{−β[E(x,\lambda=1)− E(x,\lambda=0)]} \rangle_{\lambda=0}$.
$\langle \rangle_{\lambda=0}$ indicate that samples are drawn from the equilibrium distribution $\pi(x, \lambda=0)$.

# Installation

We recommend setting up a new python environment with `python=3.9` and installing the packages defined here using `mamba`: https://github.com/wiederm/endstate_correction/blob/main/devtools/conda-envs/test_env.yaml.
This package can be installed using:
`pip install git+https://github.com/wiederm/endstate_correction.git`.

# How to use this package

We have prepared two scripts that should help to use this package, both are located in `endstate_correction/scripts`.
We will start by describing the use of the `sampling.py` script and then discuss the `perform_correction.py` script.

# A typical NEQ workflow

## Generate the equilibrium distribution $\pi(x, \lambda=0)$

In order to perform a NEQ work protocol, we need samples drawn from the equilibrium distribution from which we initialize our trial moves.
If samples are not already available, the `sampling.py` script provides and easy way to obtain these samples.

We start by setting up an openMM system object. Here, we use CHARMM parameter and files, but any other supported parameter set and files can be used. We start by defining a `CharmmPsfFile`, `PDBFile` and `CharmmParameterSet`:  

```
psf = CharmmPsfFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.psf")
pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.pdb")
params = CharmmParameterSet(
    f"{parameter_base}/{system_name}/charmm-gui/unk/unk.rtf",
    f"{parameter_base}/{system_name}/charmm-gui/unk/unk.prm",
    f"{parameter_base}/toppar/top_all36_cgenff.rtf",
    f"{parameter_base}/toppar/par_all36_cgenff.prm",
    f"{parameter_base}/toppar/toppar_water_ions.str",
)
```
and then we define the atoms that should be perturbed using the coupling parameter $\lambda$ with
```
ml_atoms = [atom.index for atom in chains[0].atoms()]
```
Depending if all atoms in your system are included in the `ml_atoms` list or only a subset, you can set up your QML or QML/MM simulation object using 

```
sim = create_charmm_system(psf=psf, parameters=params, env=env, ml_atoms=ml_atoms)
```
That is everything you need to define to run the equilibrium sampling. 
The parameters defining the number of samples to save and time interval between samples can be defined in the script.
Keep in mind that if you want to perform bidirectional FEP you need to sample at $\pi(x, \lambda=0)$ and $\pi(x, \lambda=1)$. 
This can be controlled by setting the number using the variable `nr_lambda_states`.

## Perform unidirectional NEQ from $\pi(x, \lambda=0)$

The endstate correction can be performed using the script `perform_correction.py`.
Again, we need to initialize the simulation object as described above.
To perform a specific endstate correction we need to define a protocol (some standard protocols are shown here https://github.com/wiederm/endstate_correction/blob/main/endstate_correction/tests/test_endstate_correction.py) with:
```
fep_protocoll = Protocoll(
    method="NEQ",
    direction="unidirectional",
    sim=sim,
    trajectories=[mm_samples],
    nr_of_switches=400,
    neq_switching_length=5_000, # in fs
)
```
.
This protocol is then passed to the actual function performing the protocol: `perform_endstate_correction(fep_protocoll)`.

## Analyse results of an unidirection NEQ protocol

To analyse the results generated by `r = perform_endstate_correction(...)` pass the return value to     `plot_endstate_correction_results(system_name, r, "results_neq_unidirectional.png")` and results will be plotted and printed.
An example is shown e.g. here:
https://github.com/wiederm/endstate_correction/blob/74b20882d1420884d1b014a656ed5727d1a159c9/endstate_correction/tests/test_analysis.py#L58.

### Copyright

Copyright (c) 2022, Sara Tkaczyk, Johannes Karwounopoulos & Marcus Wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
