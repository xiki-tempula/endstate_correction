# general imports
import pickle
import sys, os
import torch
from endstate_rew.system import (
    collect_samples,
    create_charmm_system,
    initialize_simulation_charmm
)
from endstate_rew.constant import zinc_systems

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

print(name)
print(smiles)

n_samples = 5_000
n_steps_per_sample = 2_000

###################

system = create_charmm_system(name)
w_dir = f"/data/shared/projects/endstate_rew/{name}/sampling_charmmff/run01/"
os.makedirs(w_dir, exist_ok=True)

sim = initialize_simulation_charmm(zinc_id=name, at_endstate="MM", platform="CPU")
mm_samples = collect_samples(
    sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
)
pickle.dump(
    mm_samples,
    open(f"{w_dir}/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle", "wb+"),
)

sim = initialize_simulation_charmm(zinc_id=name, at_endstate="QML", platform="CPU")
qml_samples = collect_samples(
    sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample
)
pickle.dump(
    qml_samples,
    open(f"{w_dir}/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle", "wb+"),
)
