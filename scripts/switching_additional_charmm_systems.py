# general imports
import pickle
import sys
import os
from os import path
import numpy as np
import torch
from endstate_rew.neq import perform_switching
from openmm.app import (
    PME,
    CharmmParameterSet,
    CharmmPsfFile,
    NoCutoff,
    PDBFile,
    Simulation,
)
from openmmml import MLPotential
from endstate_rew.constant import collision_rate, jctc_systems, stepsize, temperature
from glob import glob
import json
from openmm import unit
import endstate_rew
import openmm as mm
import mdtraj

package_path = endstate_rew.__path__[0]


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


num_threads = 2
torch.set_num_threads(num_threads)
env = "vacuum"  # waterbox
assert env in ("waterbox", "vacuum")
###########################################################################################
# define run
print("Simulating jctc system")
system_id = int(sys.argv[1])
system_name = jctc_systems[system_id]
# choose ff and working directory
ff = "charmmff"  # openff
platform = "CUDA"
###########################################
potential = MLPotential("ani2x")

base = f"/data/shared/projects/endstate_rew/jctc_data/{system_name}/"
parameter_base = f"{package_path}/data/jctc_data"
###################
# equilibrium samples
n_samples = 5_000
n_steps_per_sample = 1_000
#############
# NEQ
switching_length = 5_001
nr_of_switches = 500
#############
save_traj = True
mm_to_qml_traj_filename = f"{base}/switching_{ff}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_qml_endstate_nr_samples_{nr_of_switches}_switching_length_{switching_length}_{env}.pickle"
qml_to_mm_traj_filename = f"{base}/switching_{ff}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_mm_endstate_nr_samples_{nr_of_switches}_switching_length_{switching_length}_{env}.pickle"
#############

print(f"{ff=}")
print(f"{system_name=}")

mm_to_qml_filename = f"{base}/switching_{ff}/{system_name}_neq_ws_from_mm_to_qml_{nr_of_switches}_{switching_length}.pickle"
qml_to_mm_filename = f"{base}/switching_{ff}/{system_name}_neq_ws_from_qml_to_mm_{nr_of_switches}_{switching_length}.pickle"

if path.isfile(mm_to_qml_filename) and path.isfile(qml_to_mm_filename):
    print("All work values have already been calculated.")
    sys.exit()


# create folder
os.makedirs(f"{base}/switching_{ff}", exist_ok=True)
print(f"Generate directory: {base}/switching_{ff}")

###########################################################################################
###########################################################################################
# generate mol
# generate simulation
if env == "vacuum":
    top = f"{parameter_base}/{system_name}/charmm-gui/openmm/vac.psf"
    psf = CharmmPsfFile(top)
    pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/vac.pdb")
elif env == "waterbox":
    top = f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.psf"
    psf = CharmmPsfFile(top)
    pdb = PDBFile(f"{parameter_base}/{system_name}/charmm-gui/openmm/step3_input.pdb")
else:
    raise RuntimeError()

params = CharmmParameterSet(
    f"{parameter_base}/{system_name}/charmm-gui/unk/unk.rtf",
    f"{parameter_base}/{system_name}/charmm-gui/unk/unk.prm",
    f"{parameter_base}/toppar/top_all36_cgenff.rtf",
    f"{parameter_base}/toppar/par_all36_cgenff.prm",
    f"{parameter_base}/toppar/toppar_water_ions.str",
)

if env == "vacuum":
    mm_system = psf.createSystem(params, nonbondedMethod=NoCutoff)
elif env == "waterbox":
    psf = read_box(psf, f"{parameter_base}/{system_name}/charmm-gui/input.config.dat")
    mm_system = psf.createSystem(params, nonbondedMethod=PME)
else:
    raise RuntimeError()

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
###########################################################################################
# load samples for lambda=0. , the mm endstate
mm_samples = []
mm_sample_files = glob(
    f"{base}/sampling_{ff}/run*/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000_{env}.dcd"
)
nr_of_runs = len(mm_sample_files)

for samples in mm_sample_files:
    mm_samples.extend(
        mdtraj.load_dcd(
            samples,
            top=top,
        ).xyz
        * unit.nanometer
    )

assert len(mm_samples) == nr_of_runs * n_samples
###########################################################################################
# load samples for lambda=1. , the qml endstate
qml_samples = []
qml_sample_files = glob(
    f"{base}/sampling_{ff}/run*/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000_{env}.dcd"
)
nr_of_runs = len(qml_sample_files)

for samples in qml_sample_files:
    qml_samples.extend(
        mdtraj.load_dcd(
            samples,
            top=top,
        ).xyz
        * unit.nanometer
    )

assert len(qml_samples) == nr_of_runs * n_samples
###########################################################################################
# MM endstate
# perform switching only if file does not already exist
if not path.isfile(mm_to_qml_filename):
    # define lambda space
    lambs = np.linspace(0, 1, switching_length)
    # perform NEQ from MM to QML
    ws_from_mm_to_qml, mm_to_qml_samples = perform_switching(
        sim,
        lambdas=lambs,
        samples=mm_samples,
        nr_of_switches=nr_of_switches,
        save_traj=save_traj,
    )
    # dump work values
    print(ws_from_mm_to_qml)
    pickle.dump(ws_from_mm_to_qml, open(mm_to_qml_filename, "wb"))

    if save_traj:
        # save qml endstate samples
        pickle.dump(mm_to_qml_samples, open(mm_to_qml_traj_filename, "wb+"))
        print(f"traj dump to: {mm_to_qml_traj_filename}")
else:
    print(f"Already calculated: {mm_to_qml_filename}")

###########################################################################################
# QML endstate
# # perform switching only if file does not already exist
if not path.isfile(qml_to_mm_filename):
    # define lambda space
    lambs = np.linspace(1, 0, switching_length)
    # perform NEQ from QML to MM
    ws_from_qml_to_mm, qml_to_mm_samples = perform_switching(
        sim,
        lambdas=lambs,
        samples=qml_samples,
        nr_of_switches=nr_of_switches,
        save_traj=save_traj,
    )
    # dump work values
    pickle.dump(ws_from_qml_to_mm, open(qml_to_mm_filename, "wb+"))
    print(ws_from_qml_to_mm)

    if save_traj:
        # save MM endstate samples
        pickle.dump(qml_to_mm_samples, open(qml_to_mm_traj_filename, "wb+"))
        print(f"traj dump to: {qml_to_mm_traj_filename}")
else:
    print(f"Already calculated: {qml_to_mm_filename}")
