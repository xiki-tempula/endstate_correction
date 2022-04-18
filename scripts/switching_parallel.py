# general imports
import pickle
import sys
import os
from os import path
import numpy as np
import torch
from endstate_rew.neq import perform_switching
from endstate_rew.system import generate_molecule, initialize_simulation
from endstate_rew.constant import zinc_systems

num_threads = 2
torch.set_num_threads(num_threads)

###########################################################################################
# define run
if len(sys.argv) == 3:
    print("Simulating zink system")
    zink_id = int(sys.argv[1])
    name, smiles = zinc_systems[zink_id]
    run_id = sys.argv[2]
elif len(sys.argv) == 2:
    name = "2cle"
    smiles = "ClCCOCCCl"
    run_id = sys.argv[1]
else:
    raise RuntimeError()

print(f"{name=}")
print(f"{smiles=}")
print(f"{run_id=}")


# equilibrium samples
n_samples = 5_000
n_steps_per_sample = 2_000
#############
# NEQ
switching_length = 10_001
nr_of_switches = 10
#############
w_dir = f"/data/shared/projects/endstate_rew/{name}/"
run = "run01"

mm_to_qml_filename = f"{w_dir}/switching/{run}/{name}_neq_ws_from_mm_to_qml_{nr_of_switches}_{switching_length}_{run_id}.pickle"
qml_to_mm_filename = f"{w_dir}/switching/{run}/{name}_neq_ws_from_qml_to_mm_{nr_of_switches}_{switching_length}_{run_id}.pickle"

###########################################################################################
###########################################################################################
molecule = generate_molecule(smiles)
sim = initialize_simulation(molecule)
###########################################################################################
# create folder
os.makedirs(f"{w_dir}/switching/{run}", exist_ok=True)
print(f"Generate directory: {w_dir}/switching/{run}")

# load samples
mm_samples = pickle.load(
    open(
        f"{w_dir}/sampling/{run}/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle",
        "rb",
    )
)
qml_samples = pickle.load(
    open(
        f"{w_dir}/sampling/{run}/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle",
        "rb",
    )
)

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
