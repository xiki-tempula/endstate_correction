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
)

### define units
num_threads = 2
torch.set_num_threads(num_threads)

###################
if len(sys.argv) > 1:
    print("Simulating zink system")
    zink_id = int(sys.argv[1])
    name, smiles = zinc_systems[zink_id]
else:
    name = "2cle"
    smiles = "ClCCOCCCl"
run_id = 2
###################
print(zink_id)
print(name)
print(smiles)
print(run_id)

n_samples = 5_000
n_steps_per_sample = 2_000
###################
molecule = generate_molecule(smiles)
w_dir = f"/data/shared/projects/endstate_rew/{name}/sampling/run{run_id}/"
os.makedirs(w_dir, exist_ok=True)

# select a random conformation
from random import randint

conf_id = randint(0, molecule.n_conformers - 1)
print(f"select conf_id: {conf_id}")


sim = initialize_simulation(
    molecule,
    at_endstate="MM",
    platform="CPU",
    w_dir=f"/data/shared/projects/endstate_rew/{name}/",
    conf_id=conf_id,
)
mm_samples = collect_samples(
    sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
)
pickle.dump(
    mm_samples,
    open(f"{w_dir}/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle", "wb+"),
)

sim = initialize_simulation(
    molecule,
    at_endstate="QML",
    platform="CPU",
    w_dir=f"/data/shared/projects/endstate_rew/{name}/",
    conf_id=conf_id,
)
qml_samples = collect_samples(
    sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
)
pickle.dump(
    qml_samples,
    open(f"{w_dir}/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle", "wb+"),
)
