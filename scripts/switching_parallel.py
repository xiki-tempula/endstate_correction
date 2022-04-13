# general imports
import pickle
import sys
import os

import numpy as np
import torch
from endstate_rew.neq import perform_switching
from endstate_rew.system import generate_molecule, initialize_simulation

num_threads = 4
torch.set_num_threads(num_threads)

###########################################################################################
# define samples
name = 'methanol'
smiles = 'CO'
n_samples = 2_000
n_steps_per_sample = 1_000
#############
# NEQ
switching_length = 5_001
nr_of_switches = 10
###########################################################################################
###########################################################################################
molecule = generate_molecule(smiles)
sim = initialize_simulation(molecule)
###########################################################################################

# load samples
mm_samples = pickle.load(open(f'../data/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle', 'rb'))
qml_samples = pickle.load(open(f'../data/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle', 'rb'))

# create folder
w_dir = f'{name}_neq_switching'
os.makedirs(f'../data/{w_dir}', exist_ok=True)

# define lambda space
lambs = np.linspace(0,1,switching_length)
# perform NEQ from MM to QML
ws_from_mm_to_qml = perform_switching(sim,lambdas=lambs, samples=mm_samples,nr_of_switches=nr_of_switches)
# dump work values
pickle.dump(ws_from_mm_to_qml, open(f'../data/{w_dir}/{name}_neq_ws_from_mm_to_qml_{name}_{nr_of_switches}_{switching_length}_{sys.argv[1]}.pickle', 'wb+'))

# define lambda space
lambs = np.linspace(1,0,switching_length)
# perform NEQ from QML to MM
ws_from_qml_to_mm = perform_switching(sim,lambdas=lambs,samples=qml_samples,nr_of_switches=nr_of_switches)
# dump work values
pickle.dump(ws_from_qml_to_mm, open(f'../data/{w_dir}/{name}_neq_ws_from_qml_to_mm_{name}_{nr_of_switches}_{switching_length}_{sys.argv[1]}.pickle', 'wb+'))
