# general imports
import pickle
import sys, os
import torch
from endstate_rew.system import (collect_samples, generate_molecule,
                                 initialize_simulation)
from endstate_rew.constant import zinc_systems

### define units
num_threads = 4
torch.set_num_threads(num_threads)

###################
if len(sys.argv) > 1:
    print('Simulating zink system')
    zink_id = int(sys.argv[1])
    name, smiles = zinc_systems[zink_id]
else:
    name = '2cle'
    smiles = 'ClCCOCCCl'

print(name)
print(smiles)

n_samples = 5_000
n_steps_per_sample = 2_000

###################
molecule = generate_molecule(smiles)

os.makedirs(f'../data/{name}', exist_ok=True)

sim = initialize_simulation(molecule, at_endstate='MM', platform='CPU')
mm_samples = collect_samples(sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample)
pickle.dump(mm_samples, open(f'../data/{name}/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle', 'wb+'))

sim = initialize_simulation(molecule, at_endstate='QML', platform='CPU')
qml_samples = collect_samples(sim, n_samples=n_samples, n_steps_per_sample=n_steps_per_sample)
pickle.dump(qml_samples, open(f'../data/{name}/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle', 'wb+'))
