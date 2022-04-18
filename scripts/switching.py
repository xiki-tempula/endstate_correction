# general imports
import pickle
import os, sys

import numpy as np
import torch
from endstate_rew.neq import perform_switching
from endstate_rew.system import generate_molecule, initialize_simulation
from pymbar import BAR, EXP
from endstate_rew.constant import zinc_systems

num_threads = 4
torch.set_num_threads(num_threads)

###########################################################################################
if len(sys.argv) > 1:
    print("Simulating zink system")
    zink_id = int(sys.argv[1])
    name, smiles = zinc_systems[zink_id]
else:
    name = "2cle"
    smiles = "ClCCOCCCl"

print(name)
print(smiles)

n_samples = 5_000
n_steps_per_sample = 2_000
#############
# NEQ
switching_length = 5_001
nr_of_switches = 500
###########################################################################################
molecule = generate_molecule(smiles)
sim = initialize_simulation(molecule)
###########################################################################################
# create folder
w_dir = f"/data/shared/projects/endstate_rew/{name}/switching/run01/"
os.makedirs(w_dir, exist_ok=True)

# load samples
mm_samples = pickle.load(
    open(f"../data/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle", "rb")
)
qml_samples = pickle.load(
    open(f"../data/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle", "rb")
)

# define lambda space
lambs = np.linspace(0, 1, switching_length)
# perform NEQ from MM to QML
ws_from_mm_to_qml = perform_switching(
    sim, lambdas=lambs, samples=mm_samples, nr_of_switches=nr_of_switches
)
# dump work values
pickle.dump(
    ws_from_mm_to_qml,
    open(
        f"{w_dir}/{name}_neq_dws_from_mm_to_qml_{nr_of_switches}_{switching_length}.pickle",
        "wb+",
    ),
)

# define lambda space
lambs = np.linspace(1, 0, switching_length)
# perform NEQ from QML to MM
ws_from_qml_to_mm = perform_switching(
    sim, lambdas=lambs, samples=qml_samples, nr_of_switches=nr_of_switches
)
# dump work values
pickle.dump(
    ws_from_mm_to_qml,
    open(
        f"{w_dir}/{name}_neq_dws_from_qml_to_mm_{nr_of_switches}_{switching_length}.pickle",
        "wb+",
    ),
)

print(f"Crooks' equation: {BAR(ws_from_mm_to_qml, ws_from_qml_to_mm)}")
print(f"Jarzynski's equation: {EXP(ws_from_mm_to_qml)}")

# instantenious swichting (FEP)
switching_length = 2
lambs = np.linspace(0, 1, switching_length)
ws_from_mm_to_qml_inst = perform_switching(
    sim, lambdas=lambs, samples=mm_samples, nr_of_switches=nr_of_switches
)
lambs = np.linspace(1, 0, switching_length)
ws_from_qml_to_mm_inst = perform_switching(
    sim, lambdas=lambs, samples=qml_samples, nr_of_switches=nr_of_switches
)
print(f"FEP: From MM to QML: {EXP(ws_from_mm_to_qml_inst)}")
print(f"FEP: From MM to QML: {EXP(ws_from_qml_to_mm_inst)}")
print(f"BAR: {BAR(ws_from_mm_to_qml_inst, ws_from_qml_to_mm_inst)}")
