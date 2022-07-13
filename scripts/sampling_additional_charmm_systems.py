# general imports
import os, sys
import pickle

import numpy as np
import openmm as mm
from openmm import unit
import torch
from endstate_rew.constant import collision_rate, stepsize, temperature, jctc_systems
from endstate_rew.system import generate_samples
from openmm.app import (
    CharmmParameterSet,
    CharmmPsfFile,
    NoCutoff,
    PME,
    PDBFile,
    Simulation,
    DCDReporter,
)
from openmmml import MLPotential
import json

### set number of CPU threads used by pytorch
num_threads = 2
torch.set_num_threads(num_threads)


def read_box(psf, filename):
    try:
        sysinfo = json.load(open(filename, 'r'))
        boxlx, boxly, boxlz = map(float, sysinfo['dimensions'][:3])
    except:
        for line in open(filename, 'r'):
            segments = line.split('=')
            if segments[0].strip() == "BOXLX": boxlx = float(segments[1])
            if segments[0].strip() == "BOXLY": boxly = float(segments[1])
            if segments[0].strip() == "BOXLZ": boxlz = float(segments[1])
    psf.setBox(boxlx*unit.angstroms, boxly*unit.angstroms, boxlz*unit.angstroms)
    return psf


###################
# parse command line arguments
print("Simulating jctc system")
run_id = int(sys.argv[1])
assert run_id > 0
system_id = int(sys.argv[2])
system_name = jctc_systems[system_id]
vac = True
###################
###########################################
###########################################
potential = MLPotential("ani2x")

import endstate_rew
package_path = endstate_rew.__path__[0]

base = f"/home/mwieder/endstate_rew/data/jctc_data/{system_name}/"
parameter_base = f"{package_path}/data/jctc_data"
###################
ff = "charmmff"  # "openff" #"charmmff"  # openff
n_samples = 5_000
n_steps_per_sample = 1_000
n_lambdas = 2
platform = "CUDA"
###################
###################
os.makedirs(f"{base}/sampling_{ff}/run{run_id:0>2d}", exist_ok=True)
print('saving to {base}/sampling_{ff}/run{run_id:0>2d}')
print(f"{system_name=}")
print(f"{run_id=}")
print(f"{ff=}")
print(f"{platform=}")
print(f"{n_lambdas=}")

lambs = np.linspace(0, 1, n_lambdas)
assert len(lambs) == n_lambdas
assert lambs[0] == 0.0
assert lambs[-1] == 1.0
###################
# generate simulation
if vac:
    psf = CharmmPsfFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.psf")
    pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.pdb")
else:
    psf = CharmmPsfFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/vac.psf")
    pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/vac.pdb")

params = CharmmParameterSet(
    f"{parameter_base}/{system_name}/charmm-gui/unk/unk.rtf",
    f"{parameter_base}/{system_name}/charmm-gui/unk/unk.prm",
    f"{parameter_base}/toppar/top_all36_cgenff.rtf",
    f"{parameter_base}/toppar/par_all36_cgenff.prm",
    f"{parameter_base}/toppar/toppar_water_ions.str",
)
if vac:
    mm_system = psf.createSystem(params, nonbondedMethod=NoCutoff)
else:
    psf = read_box(psf, f'{parameter_base}/{system_name}/charmm-gui/input.config.dat')
    mm_system = psf.createSystem(params, nonbondedMethod=PME)

chains = list(psf.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]
print(f'{ml_atoms=}')
potential = MLPotential("ani2x")
ml_system = potential.createMixedSystem(
    psf.topology, mm_system, ml_atoms, interpolate=True
)

integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)
platform = mm.Platform.getPlatformByName(platform)
sim = Simulation(psf.topology, ml_system, integrator, platform=platform)
sim.context.setVelocitiesToTemperature(temperature)

###################
# perform lambda protocoll
for lamb in lambs:
    print(f"{lamb=}")
    # set lambda
    sim.context.setParameter("lambda_interpolate", lamb)
    # set coordinates
    sim.context.setPositions(pdb.positions)
    # collect samples
    sim.reporters.append(
        DCDReporter(
            f"{base}/sampling_{ff}/run{run_id:0>2d}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}.dcd",
            n_steps_per_sample,
        )
    )

    samples = generate_samples(
        sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
    )
    # save samples
    pickle.dump(
        samples,
        open(
            f"{base}/sampling_{ff}/run{run_id:0>2d}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}.pickle",
            "wb",
        ),
    )
    print(
        f"traj dump to: {base}/sampling_{ff}/run{run_id:0>2d}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}.pickle"
    )
