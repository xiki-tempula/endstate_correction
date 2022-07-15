# general imports
import json
import os
import sys

import endstate_rew
import numpy as np
import openmm as mm
import torch
from endstate_rew.constant import collision_rate, jctc_systems, stepsize, temperature
from endstate_rew.system import generate_samples
from openmm import unit
from openmm.app import (
    PME,
    CharmmParameterSet,
    CharmmPsfFile,
    DCDReporter,
    NoCutoff,
    PDBFile,
    Simulation,
)
from openmmml import MLPotential

package_path = endstate_rew.__path__[0]

### set number of CPU threads used by pytorch
num_threads = 2
torch.set_num_threads(num_threads)


def read_box(psf, filename):
    try:
        sysinfo = json.load(open(filename, "r"))
        boxlx, boxly, boxlz = map(float, sysinfo["dimensions"][:3])
    except:
        for line in open(filename, "r"):
            segments = line.split("=")
            if segments[0].strip() == "BOXLX":
                boxlx = float(segments[1])
            if segments[0].strip() == "BOXLY":
                boxly = float(segments[1])
            if segments[0].strip() == "BOXLZ":
                boxlz = float(segments[1])
    psf.setBox(boxlx * unit.angstroms, boxly * unit.angstroms, boxlz * unit.angstroms)
    return psf


###################
# parse command line arguments
print("Simulating jctc system")
run_id = int(sys.argv[1])
assert run_id > 0
system_id = int(sys.argv[2])
system_name = jctc_systems[system_id]
vac = False
###################
###########################################
###########################################
potential = MLPotential("ani2x")

base = f"/data/shared/projects/endstate_rew/jctc_data/{system_name}/"
parameter_base = f"{package_path}/data/jctc_data"
###################
ff = "charmmff"  # "openff" #"charmmff"  # openff
n_samples = 5_0  # 00
n_steps_per_sample = 1_000
n_lambdas = 2
platform = "CUDA"
###################
###################
os.makedirs(f"{base}/sampling_{ff}/run{run_id:0>2d}", exist_ok=True)
print("saving to {base}/sampling_{ff}/run{run_id:0>2d}")
print(f"{system_name=}")
print(f"{run_id=}")
print(f"{ff=}")
print(f"{platform=}")
print(f"{n_lambdas=}")
print(f"{vac=}")

lambs = np.linspace(0, 1, n_lambdas)
assert len(lambs) == n_lambdas
assert lambs[0] == 0.0
assert lambs[-1] == 1.0
###################
# generate simulation
if vac:
    psf = CharmmPsfFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/vac.psf")
    pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/vac.pdb")
else:
    psf = CharmmPsfFile(
        f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.psf"
    )
    pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.pdb")

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
    psf = read_box(psf, f"{parameter_base}/{system_name}/charmm-gui/input.config.dat")
    mm_system = psf.createSystem(params, nonbondedMethod=PME)

chains = list(psf.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]
print(f"{ml_atoms=}")
potential = MLPotential("ani2x")
ml_system = potential.createMixedSystem(
    psf.topology, mm_system, ml_atoms, interpolate=True
)

integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)
platform = mm.Platform.getPlatformByName(platform)
sim = Simulation(psf.topology, ml_system, integrator, platform=platform)


###################
# perform lambda protocoll
for lamb in lambs:
    print(f"{lamb=}")
    if vac:
        trajectory_file = f"{base}/sampling_{ff}/run{run_id:0>2d}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}_vacuum.dcd"
    else:
        trajectory_file = f"{base}/sampling_{ff}/run{run_id:0>2d}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}_waterbox.dcd"

    print(f"Trajectory saved to: {trajectory_file}")
    # set lambda
    sim.context.setParameter("lambda_interpolate", lamb)
    # set coordinates
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature)
    # collect samples
    sim.reporters.append(
        DCDReporter(
            trajectory_file,
            n_steps_per_sample,
        )
    )

    samples = generate_samples(
        sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
    )
    sim.reporters.clear()
