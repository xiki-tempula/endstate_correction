# general imports
import pickle
import sys
import os
from os import path
import numpy as np
import torch
from endstate_rew.neq import perform_switching
from endstate_rew.system import (
    generate_molecule,
    initialize_simulation_with_charmmff,
    initialize_simulation_with_openff,
)
from endstate_rew.constant import zinc_systems
from glob import glob

num_threads = 2
torch.set_num_threads(num_threads)

###########################################################################################
# define run
if len(sys.argv) == 3:
    print("Simulating zink system")
    run_id = sys.argv[1]
    zink_id = int(sys.argv[2])
    name, smiles = zinc_systems[zink_id]
elif len(sys.argv) == 2:
    name, smiles = zinc_systems[1]
    run_id = sys.argv[1]
else:
    raise RuntimeError("Run_id needs to be provided")

# choose ff
ff = "charmmff"  # openff
w_dir = f"/data/shared/projects/endstate_rew/{name}/"
# equilibrium samples
n_samples = 5_000
n_steps_per_sample = 1_000
#############
# NEQ
switching_length = 5_001
nr_of_switches = 10
#############

print(f"{ff=}")
print(f"{name=}")
print(f"{smiles=}")
print(f"{run_id=}")

mm_to_qml_filename = f"{w_dir}/switching_{ff}/{name}_neq_ws_from_mm_to_qml_{nr_of_switches}_{switching_length}_{run_id}.pickle"
qml_to_mm_filename = f"{w_dir}/switching_{ff}/{name}_neq_ws_from_qml_to_mm_{nr_of_switches}_{switching_length}_{run_id}.pickle"

if path.isfile(mm_to_qml_filename) and path.isfile(qml_to_mm_filename):
    sys.exit()

# create folder
os.makedirs(f"{w_dir}/switching_{ff}", exist_ok=True)
print(f"Generate directory: {w_dir}/switching_{ff}")

###########################################################################################
###########################################################################################
# generate mol
if ff == "openff" and smiles:
    molecule = generate_molecule(forcefield=ff, smiles=smiles)
elif ff == "charmmff" and smiles and not name:
    raise RuntimeError("Charmff can not be used with SMILES input")
else:
    molecule = generate_molecule(forcefield=ff, name=name, base="../data/hipen_data")

# initialize simulation depending on ff keyword
if ff == "openff":
    sim = initialize_simulation_with_openff(
        molecule=molecule, w_dir=f"/data/shared/projects/endstate_rew/{name}/"
    )
elif ff == "charmmff":
    sim = initialize_simulation_with_charmmff(molecule=molecule, zinc_id=name)
###########################################################################################

# load samples
mm_samples = []
mm_sample_files = glob(
    f"/data/shared/projects/endstate_rew/{name}/sampling_{ff}/run*/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000.pickle"
)
nr_of_runs = len(mm_sample_files)

for samples in mm_sample_files:
    mm_samples.extend(pickle.load(open(samples, "rb")))

assert len(mm_samples) == nr_of_runs * n_samples

qml_samples = []
qml_sample_files = glob(
    f"/data/shared/projects/endstate_rew/{name}/sampling_{ff}/run*/{name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000.pickle"
)
nr_of_runs = len(qml_sample_files)

for samples in qml_sample_files:
    qml_samples.append(pickle.load(open(samples, "rb")))

assert len(qml_samples) == nr_of_runs * n_samples

# perform switching only if file does not already exist
if not path.isfile(mm_to_qml_filename):
    # define lambda space
    lambs = np.linspace(0, 1, switching_length)
    # perform NEQ from MM to QML
    ws_from_mm_to_qml = perform_switching(
        sim, lambdas=lambs, samples=mm_samples, nr_of_switches=nr_of_switches
    )
    # dump work values
    pickle.dump(ws_from_mm_to_qml, open(mm_to_qml_filename, "wb+"))

# perform switching only if file does not already exist
if not path.isfile(qml_to_mm_filename):
    # define lambda space
    lambs = np.linspace(1, 0, switching_length)
    # perform NEQ from QML to MM
    ws_from_qml_to_mm = perform_switching(
        sim, lambdas=lambs, samples=qml_samples, nr_of_switches=nr_of_switches
    )
    # dump work values
    pickle.dump(ws_from_qml_to_mm, open(qml_to_mm_filename, "wb+"))
