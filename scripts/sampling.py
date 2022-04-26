# general imports
import os
import pickle
import sys

import torch
from endstate_rew.constant import zinc_systems
from endstate_rew.system import (
    collect_samples,
    generate_molecule,
    initialize_simulation,
    create_charmm_system,
    initialize_simulation_charmm,
)

### define units
num_threads = 2
torch.set_num_threads(num_threads)

###################
###################
if len(sys.argv) > 1:
    print("Simulating zink system")
    zink_id = int(sys.argv[1])
    name, smiles = zinc_systems[zink_id]
else:
    name = "2cle"
    smiles = "ClCCOCCCl"
###################
###################
ff = "openff"
run_id = "03"
###################
###################
print(zink_id)
print(name)
print(smiles)
print(run_id)
assert ff == "openff" or ff == "charmmff"
n_samples = 5_000
n_steps_per_sample = 2_000
###################
# generate mol
molecule = generate_molecule(smiles)
# initialize working directory
w_dir = f"/data/shared/projects/endstate_rew/{name}/sampling_{ff}/run{run_id}/"
os.makedirs(w_dir, exist_ok=True)

# select a random conformation
from random import randint

conf_id = randint(0, molecule.n_conformers - 1)
print(f"select conf_id: {conf_id}")

if ff == "openff":
    sim = initialize_simulation(
        molecule,
        at_endstate="MM",
        platform="CPU",
        w_dir=f"/data/shared/projects/endstate_rew/{name}/",
        conf_id=conf_id,
    )
elif ff == "charmmff":
    sim = initialize_simulation_charmm(zinc_id=name, at_endstate="MM", platform="CPU")
else:
    raise RuntimeError("Either openff or charmmff. Abort.")

mm_samples = collect_samples(
    sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
)
pickle.dump(
    mm_samples,
    open(f"{w_dir}/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle", "wb+"),
)

if ff == "openff":
    sim = initialize_simulation(
        molecule,
        at_endstate="QML",
        platform="CPU",
        w_dir=f"/data/shared/projects/endstate_rew/{name}/",
        conf_id=conf_id,
    )
elif ff == "charmmff":
    sim = initialize_simulation_charmm(zinc_id=name, at_endstate="QML", platform="CPU")
else:
    raise RuntimeError("Either openff or charmmff. Abort.")

qml_samples = collect_samples(
    sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
)
pickle.dump(
    qml_samples,
    open(f"{w_dir}/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle", "wb+"),
)
