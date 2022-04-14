from typing import Tuple
from endstate_rew.system import generate_molecule, initialize_simulation
from openmm.app import Simulation
import pickle

def load_system_and_samples(name:str, smiles:str)->Tuple[Simulation, list,list]:
    # initialize simulation and load pre-generated samples
    
    n_samples = 2_000
    n_steps_per_sample = 1_000
    ###########################################################################################
    molecule = generate_molecule(smiles)
    sim = initialize_simulation(molecule)
    
    samples_mm = pickle.load(open(f'data/{name}/sampling/{name}_mm_samples_{n_samples}_{n_steps_per_sample}.pickle', 'rb'))
    samples_qml = pickle.load(open(f'data/{name}/sampling/{name}_qml_samples_{n_samples}_{n_steps_per_sample}.pickle', 'rb'))

    return sim, samples_mm, samples_qml



def test_neq_switchging():
    
    # load simulation and samples for 2cle
    sim, samples_mm, samples_qml = load_system_and_samples(name='2cle', smiles='ClCCOCCCl')
    
    